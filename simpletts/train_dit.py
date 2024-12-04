import pathlib
from dataclasses import asdict

import transformers
from transformers import AutoModelForCausalLM, Trainer
from transformers.models.qwen2_moe.configuration_qwen2_moe import \
    Qwen2MoeConfig

from simpletts.configs import DataArguments, DITModelConfig, TrainingArguments
from simpletts.datasets.tts_dit_dataset import init_dataset_and_dataloader
from simpletts.dit import SimpleDIT
from simpletts.utils.campplus import SpkEmbExtractor


def init_model(model_args):

    config = Qwen2MoeConfig(**asdict(model_args))
    llm = AutoModelForCausalLM.from_config(config)
    print(llm)

    model = SimpleDIT(config, llm)
    print(model)
    return model


def init_dataloader(split, train_args, data_args):
    if train_args.campplus_onnx_path == '':
        spk_emb_extractor = None
    else:
        spk_emb_extractor = SpkEmbExtractor(train_args.campplus_onnx_path)
    _, dataloader = init_dataset_and_dataloader(
        data_args.data_path,
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
        (DITModelConfig, TrainingArguments, DataArguments))
    (model_args, training_args,
     data_args) = parser.parse_args_into_dataclasses()

    model = init_model(model_args)

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    train_dataloader = init_dataloader(split='train',
                                       train_args=training_args,
                                       data_args=data_args)
    eval_dataloader = init_dataloader(split='eval',
                                      train_args=training_args,
                                      data_args=data_args)

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
