from typing import Iterator, Any, Callable
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import LoraConfig, get_peft_model
import sys
import os
import torch

from .dtype import *
from .recipe import load_recipe

TEXT_KEY = "text"
TOKEN_MAX_LENGTH = 2048
TOKEN_MAX_LENGTH = 256

# DATA
def load_datagenerator(config: DataConfig) -> Callable[[], Iterator[Example]]:
    def datagenerator_jsonl_chat() -> Iterator[Example]:
        with open(config.path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                prompt, completion = item["prompt"], item["completion"]
                assert isinstance(prompt, str) and isinstance(completion, str)

                yield {
                    TEXT_KEY: f"{prompt}\n{completion}",
                }
    
    def datagenerator_raw() -> Iterator[Example]:
        for name in os.listdir(config.path):
            path = f"{config.path}/{name}"
            text = open(path).read()
            chunk_size = config.chunk_size
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i+chunk_size]
                yield {
                    TEXT_KEY: chunk,
                }

    if config.dataset_type == "raw":
        return datagenerator_raw
    elif config.dataset_type == "jsonl_chat":
        return datagenerator_jsonl_chat
    else:
        raise RuntimeError(f"ERROR: wrong dataset_type {config.dataset_type}")

def load_dataset(config: DataConfig) -> Dataset:
    return Dataset.from_generator(load_datagenerator(config))

# MODEL and TRAIN

def load_tokenizer(config: ModelConfig) -> Tokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=config.model_path,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_model(config: ModelConfig) -> Model:
    model_kwargs: dict[str, Any] = {}
    if config.model_kwargs is not None:
        model_kwargs.update(config.model_kwargs)
    
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=config.model_path,
        device_map="auto",
        **model_kwargs,
    )
    if config.finetune_mode == "sft":
        return model
    elif config.finetune_mode == "lora":
        lora_kwargs = {
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj", "v_proj"],
            "lora_dropout": 0.05,
            "bias": "none",
            "inference_mode": False,
            "task_type": "CAUSAL_LM",
        }
        if config.lora_kwargs is not None:
            lora_kwargs.update(config.lora_kwargs)

        lora_config = LoraConfig(**lora_kwargs)
        model = get_peft_model(model, lora_config)
        return model
    else:
        raise RuntimeError("ERROR: finetune_mode")

def train(config: TrainConfig, tokenizer: Tokenizer, model: Model, dataset: Dataset):
    def preprocess_function(examples: Examples):
        tokenized = tokenizer(
            examples[TEXT_KEY],
            padding="max_length",
            truncation=True,
            max_length=TOKEN_MAX_LENGTH,
        )

        tokenized["labels"] = tokenized["input_ids"] # for CausalLM
        return tokenized
    
    tokenized_dataset = dataset.map(
        function=preprocess_function,
        batched=True,
    )

    has_cuda = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_available()

    training_args = Seq2SeqTrainingArguments(
        seed=config.seed,
        output_dir=config.output_dir,

        # data
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.batch_size,
        dataloader_prefetch_factor=config.dataloader_prefetch_factor,
        dataloader_num_workers=config.dataloader_num_workers,

        # optimizer
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_grad_norm=config.max_grad_norm,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,

        optim=config.optim,
        adam_beta1=config.adam_beta1,
        adam_beta2=config.adam_beta2,
        adam_epsilon=config.adam_epsilon,

        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,

        # save
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,

        # logging
        logging_strategy=config.logging_strategy,
        logging_steps=config.log_steps,

        # others
        # bf16/tf32 only when CUDA is available; MPS will run in full precision
        bf16=has_cuda,
        tf32=has_cuda,
        # let HF Trainer automatically select MPS/GPU/CPU instead of forcing CPU
        generation_max_length=TOKEN_MAX_LENGTH,
        predict_with_generate=True,
        report_to="none",
        push_to_hub=False,
        remove_unused_columns=False,
        torch_compile=False,
        dataloader_pin_memory=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)


def main():
    config = load_recipe(sys.argv[1])

    dataset = load_dataset(config.data)
    tokenizer = load_tokenizer(config.model)
    model = load_model(config.model)
    train(
        config=config.train,
        tokenizer=tokenizer,
        model=model,
        dataset=dataset,
    )

if __name__ == "__main__":
    main()