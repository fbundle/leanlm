import os
from typing import Any, Iterable, Callable, Literal

import torch
from datasets import Dataset
from pydantic import BaseModel, ConfigDict
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer


Language = str

class Processor(object):
    def marshal_input(self, input_text: str) -> Language:
        raise NotImplementedError

    def unmarshal_input(self, prompt: Language) -> str:
        raise NotImplementedError

    def unmarshal_output(self, completion: Language) -> str:
        raise NotImplementedError

Mode = Literal["debug", "prepare", "train"]

class TrainConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    mode: Mode

    output_dir: str
    processor: Processor
    tokenizer: Any # TODO - change to something that has .encode and .decode
    model: Any # TODO - change to something that has .generate

    reward_func: Callable[[str, str], float]

    batch_size: int
    accumulation_steps: int = 1
    num_generations: int

    max_completion_length: int
    temperature: float
    top_p: float
    min_p: float
    top_k: int

    repetition_penalty: float

    save_steps: int
    train_data: Iterable[str]
    eval_data: list[str]

    deepspeed: str = "conf/ds_zero2.json"

def take(n: int, i: Iterable[Any]) -> Iterable[Any]:
    return (x for _, x in zip(range(n), i))

def train(config: TrainConfig):
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    model, tokenizer = config.model, config.tokenizer

    if config.mode in ["prepare", "debug"]:
        def apply_chat_template(*args, **kwargs):
            raise RuntimeError("GRPO must not use apply_chat_template")

        # prevent TRL from using apply_chat_template
        tokenizer.apply_chat_template = apply_chat_template

        def prepare_generate(generate):
            def helper(*args, **kwargs):
                if "min_new_tokens" not in kwargs:
                    kwargs["min_new_tokens"] = config.max_completion_length
                generate(*args, **kwargs)
            return helper

        # in prepare mode, always generate in full to monitor GPU memory
        model.generate = prepare_generate(model.generate)

        if config.mode == "debug":
            # in debug mode, make everything as small as possible
            config.batch_size = 1
            config.accumulation_steps = 2
            config.num_generations = 2
            config.max_completion_length = 16

        # in prepare or debug mode, train for only 2 accumulation steps
        n = 2 * config.batch_size * config.accumulation_steps
        config.train_data = list(take(n, config.train_data))

    # DATASET

    train_dataset = Dataset.from_generator(map(
        lambda x: {"prompt": config.processor.marshal_input(x)},
        config.train_data,
    ))
    eval_dataset = Dataset.from_list(list(map(
        lambda x: {"prompt": config.processor.marshal_input(x)},
        config.eval_data,
    )))

    has_cuda = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_available()
    use_vllm = False # TODO - enable vllm

    training_args = GRPOConfig(
        output_dir=config.output_dir,
        num_train_epochs=1,

        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.accumulation_steps,
        per_device_eval_batch_size=config.batch_size,
        num_generations=config.num_generations,

        # floating point precision
        bf16=has_cuda or has_mps,
        tf32=has_cuda,

        # logging
        logging_strategy="steps",
        logging_steps=config.save_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        # eval_strategy="steps",
        eval_strategy="no",
        eval_steps=config.save_steps,
        eval_on_start=False,

        # generation
        generation_kwargs={
            "max_new_tokens": config.max_completion_length,

            "temperature": config.temperature,
            "top_p": config.top_p,
            "min_p": config.min_p,
            "top_k": config.top_k,

            "repetition_penalty": config.repetition_penalty,
        },

        use_vllm=use_vllm,
        vllm_mode="colocate",
        vllm_max_model_length=config.max_completion_length,

        gradient_checkpointing=True,
    )

    def reward_func(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        answers = list(map(config.processor.unmarshal_output, completions))
        inputs = list(map(config.processor.unmarshal_input, prompts))

        rewards = [config.reward_func(i, a) for i, a in zip(inputs, answers)]
        return rewards

    trainer = GRPOTrainer(
        args=training_args,
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_func,
        reward_processing_classes=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    for sample in config.eval_data:
        print(sample)

    resume_from_checkpoint = get_last_checkpoint(config.output_dir)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    trainer.save_model(config.output_dir)









