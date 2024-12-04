import inspect
from dataclasses import asdict

import torch
import torch.distributed as dist
from peft import AdaptionPromptConfig, LoraConfig, PrefixTuningConfig
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.utils.data import DistributedSampler
from transformers import default_data_collator
from transformers.data import DataCollatorForSeq2Seq
from tts_recipes.configs import (datasets, llama_adapter_config, lora_config,
                                 prefix_config, train_config)
from tts_recipes.data.sampler import (DistributedLengthBasedBatchSampler,
                                      LengthBasedBatchSampler)
from tts_recipes.datasets import DATASET_PREPROC


def update_config(config, **kwargs):
    if isinstance(config, (tuple, list)):
        for c in config:
            update_config(c, **kwargs)
    else:
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
            elif "." in k:
                # allow --some_config.some_param=True
                config_name, param_name = k.split(".")
                if type(config).__name__ == config_name:
                    if hasattr(config, param_name):
                        setattr(config, param_name, v)
                    else:
                        # In case of specialized config we can warn user
                        print(
                            f"Warning: {config_name} does not accept parameter: {k}"
                        )
            elif isinstance(config, train_config):
                print(f"Warning: unknown parameter {k}")


def generate_peft_config(train_config, kwargs):
    configs = (lora_config, llama_adapter_config, prefix_config)
    peft_configs = (LoraConfig, AdaptionPromptConfig, PrefixTuningConfig)
    names = tuple(c.__name__.rstrip("_config") for c in configs)

    if train_config.peft_method not in names:
        raise RuntimeError(
            f"Peft config not found: {train_config.peft_method}")

    if train_config.peft_method == "prefix":
        raise RuntimeError(
            "PrefixTuning is currently not supported (see https://github.com/meta-llama/llama-recipes/issues/359#issuecomment-2089350811)"
        )

    if train_config.enable_fsdp and train_config.peft_method == "llama_adapter":
        raise RuntimeError(
            "Llama_adapter is currently not supported in combination with FSDP (see https://github.com/meta-llama/llama-recipes/issues/359#issuecomment-2089274425)"
        )

    config = configs[names.index(train_config.peft_method)]()

    update_config(config, **kwargs)
    params = asdict(config)
    peft_config = peft_configs[names.index(train_config.peft_method)](**params)

    return peft_config


def generate_dataset_config(train_config, kwargs):
    names = tuple(DATASET_PREPROC.keys())

    assert train_config.dataset in names, f"Unknown dataset: {train_config.dataset}"

    dataset_config = {k: v
                      for k, v in inspect.getmembers(datasets)
                      }[train_config.dataset]()

    update_config(dataset_config, **kwargs)

    return dataset_config


def get_dataloader_kwargs(train_config, mode, seed=2024):

    generator = torch.Generator()
    generator.manual_seed(seed)

    kwargs = {}
    kwargs['batch_size'] = None
    kwargs['persistent_workers'] = True
    kwargs[
        'prefetch_fector'] = train_config.train_prefetch if mode == 'train' else train_config.val_prefetch

    kwargs[
        'num_workers'] = train_config.num_workers_dataloader if mode == 'train' else train_config.val_prefetch

    return kwargs


def check_fsdp_config(fsdp_config):
    VALID_TYPES = (StateDictType.SHARDED_STATE_DICT,
                   StateDictType.FULL_STATE_DICT)
    if isinstance(fsdp_config.checkpoint_type, str):
        str_to_obj = {
            "StateDictType.SHARDED_STATE_DICT":
            StateDictType.SHARDED_STATE_DICT,
            "StateDictType.FULL_STATE_DICT": StateDictType.FULL_STATE_DICT,
        }
        if fsdp_config.checkpoint_type in str_to_obj:
            fsdp_config.checkpoint_type = str_to_obj[
                fsdp_config.checkpoint_type]

    if not fsdp_config.checkpoint_type in VALID_TYPES:
        raise ValueError(
            f"Invalid checkpoint_type {fsdp_config.checkpoint_type}")
