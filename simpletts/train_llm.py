import pathlib
from dataclasses import asdict

import transformers
from transformers import AutoModelForCausalLM, Trainer
from transformers.models.qwen2_moe.configuration_qwen2_moe import \
    Qwen2MoeConfig

from simpletts.configs import (DataArguments, TrainingArguments,
                               TTSLLMModelConfig)
from simpletts.datasets.tts_llm_dataset import init_dataset_and_dataloader
from simpletts.llm import TTSLLM
from simpletts.utils.campplus import SpkEmbExtractor


def init_model(model_args, tokenizer):

    config = Qwen2MoeConfig(**asdict(model_args))

    llm = AutoModelForCausalLM.from_config(config)
    print(llm)

    vocab = tokenizer.get_vocab()
    model = TTSLLM(
        config,
        llm,
        # TODO: fix here
        special_tokens={
            "<|startofaudio|>": len(vocab),
            "<|endofaudio|>": len(vocab) + 8192 * 4,
            "pad": tokenizer.pad_token_id,  # tokenizer.pad_token_id
        })

    print(model)
    return model


def init_tokenizer():
    # TODO : fix to use character for Chinese and bpe for English
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        'Qwen/Qwen1.5-MoE-A2.7B')
    # TODO: check pickle issues
    return tokenizer


def init_dataloader(split, train_args, data_args, tokenizer):
    if train_args.campplus_onnx_path == '':
        spk_emb_extractor = None
    else:
        spk_emb_extractor = SpkEmbExtractor(train_args.campplus_onnx_path)
    _, dataloader = init_dataset_and_dataloader(
        data_args.data_path,
        tokenizer,
        train_args.per_device_train_batch_size,
        train_args.dataloader_num_workers,
        train_args.dataloader_prefetch_factor,
        split == 'train',
        train_args.max_steps,
        True,
        spk_emb_extractor,
        seed=2025)
    return dataloader


class CustomTrainer(Trainer):

    def set_train_dataloader(self, dataloader):
        self._train_dataloader = dataloader

    def set_eval_dataloader(self, dataloader):
        self._eval_dataloader = dataloader

    def get_train_dataloader(self):
        if hasattr(self, "_train_dataloader"):
            return self._train_dataloader
        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset=None):
        if hasattr(self, "_eval_dataloader"):
            return self._eval_dataloader
        return super().get_eval_dataloader(eval_dataset)


def main():

    parser = transformers.HfArgumentParser(
        (TTSLLMModelConfig, TrainingArguments, DataArguments))
    (model_args, training_args,
     data_args) = parser.parse_args_into_dataclasses()

    # TODO: tokenizer
    tokenizer = init_tokenizer()
    model = init_model(model_args, tokenizer)

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    train_dataloader = init_dataloader(split='train',
                                       train_args=training_args,
                                       data_args=data_args,
                                       tokenizer=tokenizer)
    eval_dataloader = init_dataloader(split='eval',
                                      train_args=training_args,
                                      data_args=data_args,
                                      tokenizer=tokenizer)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
    )

    trainer.set_train_dataloader(train_dataloader)
    trainer.set_eval_dataloader(eval_dataloader)
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()


if __name__ == '__main__':
    main()
