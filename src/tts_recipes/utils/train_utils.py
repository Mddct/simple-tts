# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import contextlib
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
import yaml
from llama_recipes.model_checkpointing import (
    save_fsdp_model_checkpoint_full, save_model_and_optimizer_sharded,
    save_model_checkpoint, save_optimizer_checkpoint, save_peft_checkpoint)
from llama_recipes.policies import bfSixteen, fpSixteen, get_llama_wrapper
from llama_recipes.utils.flop_utils import FlopMeasure
from llama_recipes.utils.memory_utils import MemoryTrace
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from tqdm import tqdm
from transformers import LlamaTokenizer


def set_tokenizer_params(tokenizer: LlamaTokenizer):
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"


@contextlib.contextmanager
def profile(cfg, local_rank=None):
    use_profiler: bool = cfg.use_profiler
    use_flop_counter: bool = cfg.flop_counter
    if use_flop_counter and use_profiler:
        raise ValueError("Cannot use both profiler and flop counter")
    if use_profiler:
        # profiler needs a warmup stage to get the accurate profiling results
        wait_step, warmup_step, active_step = 1, 2, 3
        min_step = wait_step + warmup_step + active_step + 1
        if cfg.max_train_step > 0 and cfg.max_train_step < min_step:
            raise ValueError(
                f"pytorch profiler requires at least {min_step} train steps to finish the warm-up and recording stage, {wait_step} for wait_step, {warmup_step} for warmup_step, {active_step} for profiling step, please increase the max_train_step, current max_train_step {cfg.max_train_step}"
            )
        print(
            f"pytorch profiling is activated and results will be saved in {cfg.profiler_dir}"
        )
        with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=wait_step,
                                                 warmup=warmup_step,
                                                 active=active_step,
                                                 repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    cfg.profiler_dir),
                profile_memory=True,
                with_stack=False,
                with_flops=True,
                record_shapes=True,
        ) as torch_profiler:
            yield torch_profiler
    elif use_flop_counter:
        if cfg.max_train_step > 0 and cfg.max_train_step <= cfg.flop_counter_start:
            raise ValueError(
                f"flop counter requires at least {cfg.flop_counter_start + 1} train steps, please increase the max_train_step, current max_train_step {cfg.max_train_step}"
            )
        with FlopMeasure(rank=local_rank,
                         warmup_step=cfg.flop_counter_start) as flop_counter:
            yield flop_counter
    else:
        torch_profiler = contextlib.nullcontext()
        yield None


def train(model,
          train_dataloader,
          eval_dataloader,
          optimizer,
          lr_scheduler,
          gradient_accumulation_steps,
          train_config,
          fsdp_config=None,
          local_rank=None,
          rank=None,
          writer=None):
    """
    Trains the model on the given dataloader

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predicitons
        writer: 

    Returns: results dictionary containing average training and validation perplexity and loss
    """
    # Create a gradient scaler for fp16
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler()
    autocast = torch.cuda.amp.autocast if train_config.use_fp16 else nullcontext
    if train_config.save_metrics:
        if not os.path.exists(train_config.output_dir):
            os.makedirs(train_config.output_dir, exist_ok=True)
        train_step_perplexity = []
        train_step_loss = []

    checkpoint_times = []
    total_train_steps = 0
    max_steps_reached = False  # Flag to indicate max training steps reached
    # Start the training loop
    for epoch in range(1):
        print(f"Starting epoch {epoch}/{train_config.num_epochs}")
        print(f"train_config.max_train_step: {train_config.max_train_step}")
        # stop when the maximum number of training steps is reached
        if max_steps_reached:
            break
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            total_loss = 0.0
            with profile(train_config, local_rank) as profile_context:
                for step, batch in enumerate(train_dataloader):
                    total_train_steps += 1
                    # stop when the maximum number of training steps is reached
                    if train_config.max_train_step > 0 and total_train_steps > train_config.max_train_step:
                        max_steps_reached = True
                        if not train_config.enable_fsdp or local_rank == 0:
                            print(
                                "max training steps reached, stopping training, total train steps finished: ",
                                total_train_steps - 1)
                        break
                    for key in batch.keys():
                        if train_config.enable_fsdp:
                            if isinstance(batch['key'], torch.Tensor):
                                batch[key] = batch[key].to(local_rank)
                        else:
                            if isinstance(batch['key'], torch.Tensor):
                                batch[key] = batch[key].to('cuda:0')
                    with autocast():
                        loss = model(**batch).loss
                    total_loss += loss.detach().float()
                    loss = loss / gradient_accumulation_steps
                    if train_config.use_fp16:
                        # if fp16 is enabled, use gradient scaler to handle gradient update
                        scaler.scale(loss).backward()
                        if (
                                step + 1
                        ) % gradient_accumulation_steps == 0 or step == len(
                                train_dataloader) - 1:
                            if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                                scaler.unscale_(optimizer)
                                if train_config.enable_fsdp:
                                    model.clip_grad_norm_(
                                        train_config.
                                        gradient_clipping_threshold)
                                else:
                                    torch.nn.utils.clip_grad_norm_(
                                        model.parameters(), train_config.
                                        gradient_clipping_threshold)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                    else:
                        # regular backpropagation when fp16 is not used
                        loss.backward()
                        if (
                                step + 1
                        ) % gradient_accumulation_steps == 0 or step == len(
                                train_dataloader) - 1:
                            if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                                if train_config.enable_fsdp:
                                    model.clip_grad_norm_(
                                        train_config.
                                        gradient_clipping_threshold)
                                else:
                                    torch.nn.utils.clip_grad_norm_(
                                        model.parameters(), train_config.
                                        gradient_clipping_threshold)
                            optimizer.step()
                            optimizer.zero_grad()
                    lr_scheduler.step()

                    if train_config.use_profiler or train_config.flop_counter:
                        profile_context.step()
                    if train_config.flop_counter and profile_context.is_done():
                        TFlops = profile_context.get_flops_per_sec() / 1e12
                    if writer and rank == 0:
                        writer.write_scalars(
                            epoch + step, {
                                "train/loss":
                                loss.detach().float(),
                                "train/pplx":
                                float(torch.exp(loss.detach().float())),
                            })
                    if epoch + step // train_config.val_per_steps == 0:
                        if not train_config.enable_fsdp or rank == 0:
                            memtrace.print_stats()
                        if train_config.run_validation:
                            evaluation(model, train_config, eval_dataloader,
                                       local_rank, writer, step)
                    if epoch + step // train_config.save_per_steps == 0:
                        checkpoint_start_time = time.perf_counter()
                        if train_config.enable_fsdp:
                            dist.barrier()
                        save_model_optimizer(train_config, fsdp_config, model,
                                             optimizer, rank, step)

                        if train_config.enable_fsdp:
                            dist.barrier()

                        checkpoint_end_time = time.perf_counter(
                        ) - checkpoint_start_time
                        checkpoint_times.append(checkpoint_end_time)
                    if epoch + step // train_config.log_per_steps == 0:
                        ppl = float(torch.exp(loss).detach().float())
                        if train_config.enable_fsdp:
                            if local_rank == 0:
                                print(
                                    f"Train Step={step} {TFlops=} {ppl=} {loss=}"
                                )
                        else:
                            print(
                                f"Train Step={step} {TFlops=} {ppl=} {loss=}")

        if not train_config.enable_fsdp or rank == 0:
            memtrace.print_stats()

    checkpoint_start_time = time.perf_counter()
    # Saving the results every epoch to plot later
    if train_config.enable_fsdp and not train_config.use_peft and rank == 0:
        save_train_params(train_config, fsdp_config, rank)


def evaluation(
    model,
    train_config,
    eval_dataloader,
    local_rank,
    writer=None,
    step_offset=0,
):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions

    Returns: eval_ppl, eval_epoch_loss
    """
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])
    model.eval()
    eval_loss = 0.0  # Initialize evaluation loss
    total_eval_steps = 0
    num_samples = torch.tensor(0).cuda()
    with MemoryTrace() as memtrace:
        for step, batch in enumerate(
                tqdm(eval_dataloader,
                     colour="green",
                     desc="evaluating Epoch",
                     dynamic_ncols=True)):
            total_eval_steps += 1
            # stop when the maximum number of eval steps is reached
            if train_config.max_eval_step > 0 and total_eval_steps > train_config.max_eval_step:
                if not train_config.enable_fsdp or local_rank == 0:
                    print(
                        "max eval steps reached, stopping evaluation, total_eval_steps: ",
                        total_eval_steps - 1)
                break
            bs = 0
            for key in batch.keys():
                if bs == 0:
                    bs = len(batch[key])
                if train_config.enable_fsdp:
                    batch[key] = batch[key].to(local_rank)
                else:
                    batch[key] = batch[key].to('cuda:0')
            num_samples = num_samples + bs
            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                # Forward pass and compute loss
                outputs = model(**batch)
                loss = outputs.loss
                writer.write_scalars(
                    step_offset + step, {
                        "eval/loss": float(loss.detach().float()),
                        "eval/pplx": float(torch.exp(loss).float()),
                    })

                eval_loss += loss.detach().float()
            # Decode predictions and add to evaluation predictions list

    if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_samples, op=dist.ReduceOp.SUM)
    # Compute average loss and perplexity
    eval_epoch_loss = eval_loss / num_samples.cpu().numpy()
    if train_config.enable_fsdp:
        eval_epoch_loss = eval_epoch_loss / world_size
    eval_ppl = torch.exp(eval_epoch_loss)

    # Print evaluation metrics
    if train_config.enable_fsdp:
        if local_rank == 0:
            print(f" {eval_ppl=} {eval_epoch_loss=}")
    else:
        print(f" {eval_ppl=} {eval_epoch_loss=}")

    return eval_ppl, eval_epoch_loss


def freeze_transformer_layers(model, num_layer):
    for i, layer in enumerate(model.model.layers):
        if i < num_layer:
            for param in layer.parameters():
                param.requires_grad = False


def freeze_LLM_only(model):
    """
    Freeze self-attention layers in the language_model. vision_model, multi_modal_projector, and cross-attention layers will be fine-tuned
    """
    for name, param in model.language_model.named_parameters():
        param.requires_grad = False
    for i, layer in enumerate(model.language_model.model.layers):
        if i in model.language_model.model.cross_attention_layers:
            for param in layer.parameters():
                param.requires_grad = True


def check_frozen_layers_peft_model(model):
    for i, layer in enumerate(model.base_model.model.model.layers):
        for name, param in layer.named_parameters():
            print(
                f"Layer {i}, parameter {name}: requires_grad = {param.requires_grad}"
            )


def setup():
    """Initialize the process group for distributed training"""
    dist.init_process_group("nccl")


def setup_environ_flags(rank):
    """Set environment flags for debugging purposes"""
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # This flag will help with CUDA memory fragmentations that can lead into OOM in some cases.
    # Note this is only available in PyTorch Nighlies (as of July 30 2023)
    # os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    if rank == 0:
        print(f"--> Running with torch dist debug set to detail")


def cleanup():
    """Clean up the process group after training"""
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        print(f"Clearing GPU cache for all ranks")
    torch.cuda.empty_cache()


def get_parameter_dtypes(model):
    """Get the data types of model parameters"""
    parameter_dtypes = {}
    for name, parameter in model.named_parameters():
        parameter_dtypes[name] = parameter.dtype
    return parameter_dtypes


def print_model_size(model, config, rank: int = 0) -> None:
    """
    Print model name, the number of trainable parameters and initialization time.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
        init_time_start (float): Initialization start time.
        init_time_end (float): Initialization end time.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        print(f"--> Model {config.model_name}")
        total_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)
        print(
            f"\n--> {config.model_name} has {total_params / 1e6} Million params\n"
        )


def print_frozen_model_status(model, config, rank: int = 0) -> None:
    """
    Print the frozen status of the model's and the number of trainable parameters after frozen.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        trainable_params = sum(p.numel() for p in model.parameters()
                               if p.requires_grad)
        print("After freezing the model:")
        print(
            f"--> {config.model_name} has {trainable_params / 1e6} Million trainable params\n"
        )

        module_states = {}
        # Iterate over all parameters
        for name, param in model.named_parameters():
            # Extract the top-level module name (e.g., "vision_model", "language_model")
            top_module = name.split(".")[0]

            # Initialize a record for the top-level module
            if top_module not in module_states:
                module_states[top_module] = {"frozen": [], "unfrozen": []}

            # Group parameters into frozen or unfrozen
            if param.requires_grad:
                module_states[top_module]["unfrozen"].append(name)
            else:
                module_states[top_module]["frozen"].append(name)

        print("--> Model state after freezing:")
        # Analyze and print the results
        for module, states in module_states.items():
            frozen_params = states["frozen"]
            unfrozen_params = states["unfrozen"]

            if frozen_params and unfrozen_params:
                # Mixed state: both frozen and unfrozen parameters
                print(f"    {module}: Mixed")
            elif frozen_params:
                # All parameters are frozen
                print(f"    {module}: Frozen")
            else:
                # All parameters are unfrozen
                print(f"    {module}: Unfrozen")
        print("")


def get_policies(cfg, rank):
    """Get the policies for mixed precision and fsdp wrapping"""

    verify_bfloat_support = ((torch.version.cuda
                              and torch.cuda.is_bf16_supported()
                              and torch.version.cuda >= "11.0"
                              and dist.is_nccl_available()
                              and nccl.version() >= (2, 10)))

    mixed_precision_policy = None
    wrapping_policy = None

    # Mixed precision
    if cfg.mixed_precision:
        bf16_ready = verify_bfloat_support

        if bf16_ready and not cfg.use_fp16:
            mixed_precision_policy = bfSixteen
            if rank == 0:
                print(
                    f"bFloat16 enabled for mixed precision - using bfSixteen policy"
                )
        elif cfg.use_fp16:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print(f"FP16 enabled")
        else:
            print(
                f"bFloat16 support not present. Using FP32, and not mixed precision"
            )
    wrapping_policy = get_llama_wrapper()
    return mixed_precision_policy, wrapping_policy


def save_train_params(train_config, fsdp_config, rank):
    """
    This function saves the train_config and FSDP config into a train_params.yaml.
    This will be used by converter script in the inference folder to fetch the HF model name or path.
    It also would be hepful as a log for future references.
    """
    # Convert the train_config and fsdp_config objects to dictionaries,
    # converting all values to strings to ensure they can be serialized into a YAML file
    train_config_dict = {
        k: str(v)
        for k, v in vars(train_config).items() if not k.startswith('__')
    }
    fsdp_config_dict = {
        k: str(v)
        for k, v in vars(fsdp_config).items() if not k.startswith('__')
    }
    # Merge the two dictionaries into one
    train_params_dict = {**train_config_dict, **fsdp_config_dict}
    # Construct the folder name (follwoing FSDP checkpointing style) using properties of the train_config object
    folder_name = (train_config.dist_checkpoint_root_folder + "/" +
                   train_config.dist_checkpoint_folder + "-" +
                   train_config.model_name)

    save_dir = Path.cwd() / folder_name
    # If the directory does not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Convert the dictionary to a YAML string
    config_yaml = yaml.dump(train_params_dict, indent=4)
    file_name = os.path.join(save_dir, 'train_params.yaml')

    # Check if there's a directory with the same name as the file
    if os.path.isdir(file_name):
        print(f"Error: {file_name} is a directory, not a file.")
    else:
        # Write the YAML string to the file
        with open(file_name, 'w') as f:
            f.write(config_yaml)
        if rank == 0:
            print(f"training params are saved in {file_name}")


def save_to_json(output_filename, train_step_loss, train_epoch_loss,
                 train_step_ppl, train_epoch_ppl, val_step_loss,
                 val_epoch_loss, val_step_ppl, val_epoch_ppl):
    metrics_data = {
        "train_step_loss": train_step_loss,
        "train_epoch_loss": train_epoch_loss,
        "train_step_perplexity": train_step_ppl,
        "train_epoch_perplexity": train_epoch_ppl,
        "val_step_loss": val_step_loss,
        "val_epoch_loss": val_epoch_loss,
        "val_step_perplexity": val_step_ppl,
        "val_epoch_perplexity": val_epoch_ppl
    }
    with open(output_filename, "w") as f:
        json.dump(metrics_data, f)


def save_model_optimizer(train_config, fsdp_config, model, optimizer, rank,
                         step):
    if train_config.use_peft:
        if train_config.enable_fsdp:
            if rank == 0:
                print(f"we are about to save the PEFT modules")
        else:
            print(f"we are about to save the PEFT modules")
        save_peft_checkpoint(model, train_config.output_dir)
        if train_config.enable_fsdp:
            if rank == 0:
                print(
                    f"PEFT modules are saved in {train_config.output_dir} directory"
                )
        else:
            print(
                f"PEFT modules are saved in {train_config.output_dir} directory"
            )

    else:
        if not train_config.enable_fsdp:
            save_model_checkpoint(model, train_config.output_dir)

        elif fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:
            print(" Saving the FSDP model checkpoint using FULL_STATE_DICT")
            print("=====================================================")
            save_fsdp_model_checkpoint_full(model,
                                            optimizer,
                                            rank,
                                            train_config,
                                            epoch=step)

            if train_config.save_optimizer:
                print(" Saving the FSDP optimizer using FULL_STATE_DICT")
                print("=====================================================")
                save_optimizer_checkpoint(model,
                                          optimizer,
                                          rank,
                                          train_config,
                                          epoch=step)

        elif fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:

            if train_config.save_optimizer:
                print(
                    " Saving the FSDP model checkpoints using SHARDED_STATE_DICT"
                )
                print("=====================================================")
                save_model_and_optimizer_sharded(model,
                                                 rank,
                                                 train_config,
                                                 optim=optimizer)
            else:
                print(
                    " Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT"
                )
                print("=====================================================")
                save_model_and_optimizer_sharded(model, rank, train_config)
