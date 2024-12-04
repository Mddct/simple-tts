# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import random
from warnings import warn

import fire
import numpy as np
import torch
import torch.optim as optim
from peft import PeftModel, get_peft_model
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.optim.lr_scheduler import StepLR
from transformers import (AutoConfig, AutoProcessor, AutoTokenizer,
                          BitsAndBytesConfig, LlamaForCausalLM,
                          MllamaForConditionalGeneration)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mllama.modeling_mllama import (
    MllamaCrossAttentionDecoderLayer, MllamaSelfAttentionDecoderLayer,
    MllamaVisionEncoderLayer)

from tts_recipes.configs import fsdp_config as FSDP_CONFIG
from tts_recipes.configs import quantization_config as QUANTIZATION_CONFIG
from tts_recipes.configs import train_config as TRAIN_CONFIG
from tts_recipes.data.concatenator import ConcatDataset
from tts_recipes.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing
from tts_recipes.utils import fsdp_auto_wrap_policy
from tts_recipes.utils.config_utils import (check_fsdp_config,
                                            generate_dataset_config,
                                            generate_peft_config,
                                            get_dataloader_kwargs,
                                            update_config)
from tts_recipes.utils.dataset_utils import (get_custom_data_collator,
                                             get_preprocessed_dataset)
from tts_recipes.utils.fsdp_utils import hsdp_device_mesh
from tts_recipes.utils.train_utils import (clear_gpu_cache, freeze_LLM_only,
                                           freeze_transformer_layers,
                                           get_policies,
                                           print_frozen_model_status,
                                           print_model_size, setup,
                                           setup_environ_flags, train)


def main(**kwargs):
    # Update the configuration for the training and sharding process
    train_config, fsdp_config = TRAIN_CONFIG(), FSDP_CONFIG()
    update_config((train_config, fsdp_config), **kwargs)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)
    np.random.seed(train_config.seed)

    if train_config.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    # setting quantization configs
    bnb_config = None
    if train_config.quantization:
        if type(train_config.quantization) == type(True):
            warn(
                "Quantization (--quantization) is a boolean, please specify quantization as '4bit' or '8bit'. Defaulting to '8bit' but this might change in the future.",
                FutureWarning,
            )
            train_config.quantization = "8bit"

        if train_config.quantization == "8bit" and train_config.enable_fsdp:
            raise ValueError(
                "8bit quantization is not supported with FSDP, please use 4bit quantization"
            )

        quant_config = QUANTIZATION_CONFIG()
        update_config(quant_config, **kwargs)
        bnb_config = quant_config.create_bnb_config(train_config.quantization)

    # Load the pre-trained model and setup its configuration
    use_cache = False if train_config.enable_fsdp else None
    config = AutoConfig.from_pretrained(train_config.model_name)
    if config.model_type == "QwenMoe":
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            train_config.model_name,
            quantization_config=bnb_config,
            use_cache=use_cache,
            attn_implementation="sdpa"
            if train_config.use_fast_kernels else None,
            device_map=("auto" if train_config.quantization
                        and not train_config.enable_fsdp else None),
            torch_dtype=torch.float16
            if train_config.use_fp16 else torch.bfloat16,
        )
    else:
        raise ValueError(
            f"Model type {config.model_type} is not supported. Please use llama or mllama model."
        )
    # Load the tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(
        train_config.model_name if train_config.tokenizer_name is None else
        train_config.tokenizer_name)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # If there is a mismatch between tokenizer vocab size and embedding matrix,
    # throw a warning and then expand the embedding matrix
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print(
            "WARNING: Resizing the embedding matrix to match the tokenizer vocab size."
        )
        model.resize_token_embeddings(len(tokenizer))

    print_model_size(model, train_config,
                     rank if train_config.enable_fsdp else 0)

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if (train_config.enable_fsdp and fsdp_config.pure_bf16
            and not train_config.quantization):
        model.to(torch.bfloat16)

    if train_config.use_peft:
        # Load the pre-trained peft model checkpoint and setup its configuration
        if train_config.from_peft_checkpoint:
            model = PeftModel.from_pretrained(
                model, train_config.from_peft_checkpoint, is_trainable=True)
            peft_config = model.peft_config
        # Generate the peft config and start fine-tuning from original model
        else:
            peft_config = generate_peft_config(train_config, kwargs)
            model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    hsdp_device_mesh_plan = None
    if (fsdp_config.hsdp and fsdp_config.sharding_strategy
            == ShardingStrategy.HYBRID_SHARD):
        hsdp_device_mesh_plan = hsdp_device_mesh(
            replica_group_size=fsdp_config.replica_group_size,
            sharding_group_size=fsdp_config.sharding_group_size,
        )
        print("HSDP device mesh is ready")

    # setting up FSDP if enable_fsdp is enabled
    if train_config.enable_fsdp:
        check_fsdp_config(fsdp_config)

        if not train_config.use_peft and train_config.freeze_layers:
            freeze_transformer_layers(model, train_config.num_freeze_layers)
            # print model size and frozen layers after freezing layers
            print_frozen_model_status(model, train_config,
                                      rank if train_config.enable_fsdp else 0)

        if not train_config.use_peft and train_config.freeze_LLM_only and config.model_type == "mllama":
            freeze_LLM_only(model)
            # print model size and frozen layers after freezing layers
            print_frozen_model_status(model, train_config,
                                      rank if train_config.enable_fsdp else 0)

        mixed_precision_policy, wrapping_policy = get_policies(
            fsdp_config, rank)
        # Create the FSDP wrapper for LlamaDecoderLayer in text models
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(
            model, [LlamaDecoderLayer])
        device_id = 0
        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()

        if train_config.freeze_LLM_only:
            use_orig_params = True
        else:
            use_orig_params = False
        model = FSDP(
            model,
            auto_wrap_policy=(my_auto_wrapping_policy
                              if train_config.use_peft else wrapping_policy),
            cpu_offload=(CPUOffload(offload_params=True)
                         if fsdp_config.fsdp_cpu_offload else None),
            mixed_precision=(mixed_precision_policy
                             if not fsdp_config.pure_bf16 else None),
            sharding_strategy=fsdp_config.sharding_strategy,
            device_mesh=hsdp_device_mesh_plan,
            device_id=device_id,
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=((lambda module: module.to_empty(
                device=torch.device("cuda"), recurse=False)) if
                           train_config.low_cpu_fsdp and rank != 0 else None),
            use_orig_params=use_orig_params,
        )
        if fsdp_config.fsdp_activation_checkpointing:
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable()
            apply_fsdp_checkpointing(model)
    elif not train_config.quantization and not train_config.enable_fsdp:
        if torch.cuda.is_available():
            model.to("cuda")
    dataset_config = generate_dataset_config(train_config, kwargs)
    dataset_processer = tokenizer

    # Load and preprocess the dataset for training and validation
    dataset_train = init_ataset(
        dataset_processer,
        dataset_config,
        split="train",
        packing=train_config.batching_strategy,
    )

    dataset_val = init_dataset(
        dataset_processer,
        dataset_config,
        split="test",
        packing=False,
    )

    train_dl_kwargs = get_dataloader_kwargs(train_config, "train")
    # Create DataLoaders for the training and validation dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        **train_dl_kwargs,
    )
    val_dl_kwargs = get_dataloader_kwargs(train_config, "val")
    eval_dataloader = torch.utils.data.DataLoader(
        dataset_val,
        **val_dl_kwargs,
    )
    # Initialize the optimizer and learning rate scheduler
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            model.parameters(),
            lr=train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
            weight_decay=train_config.weight_decay,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)
    train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config,
        fsdp_config if train_config.enable_fsdp else None,
        local_rank if train_config.enable_fsdp else None,
        rank if train_config.enable_fsdp else None,
    )


if __name__ == "__main__":
    fire.Fire(main)
