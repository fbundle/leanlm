import unsloth
import os
import shutil
from typing import Any, Iterable, Callable, Literal

import torch
from datasets import Dataset
from pydantic import BaseModel, ConfigDict
from transformers import TrainingArguments, TrainerCallback, TrainerState, TrainerControl

from leanlm.llm_trainer.processor import Processor
from trl import GRPOConfig, GRPOTrainer # type: ignore



type Mode = Literal["prepare", "train"]
ModePrepare: Mode = "prepare"
ModeTrain: Mode = "train"

class TrainConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    mode: Mode

    code_src_list: list[str] | None = []

    output_dir: str
    processor: Processor
    tokenizer: Any # TODO - change to something that has .encode and .decode
    model: Any # TODO - change to something that has .generate

    reward_func: Callable[[str, str, str], float]

    batch_size: int
    accumulation_steps: int = 1
    num_generations: int

    generation_kwargs: dict[str, Any] | None

    save_steps: int
    train_size: int
    train_data: Callable[[int], str]
    eval_data: list[str]

    deepspeed: str | None = None


def take(n: int, i: Iterable[Any]) -> Iterable[Any]:
    return (x for _, x in zip(range(n), i))


def copy_code(output_dir: str, code_src_list: list[str]):
    print(f"copying code {code_src_list}")
    for code_src in code_src_list:
        if not os.path.exists(code_src):
            continue

        code_dst = f"{output_dir}/src/{code_src}"
        if os.path.exists(code_dst):
            shutil.rmtree(code_dst)

        if not os.path.exists(os.path.dirname(code_dst)):
            os.makedirs(os.path.dirname(code_dst))

        shutil.copytree(code_src, code_dst)

class OnSaveCallback(TrainerCallback):
    def __init__(self, callback: Callable[[], None]):
        super().__init__()
        self.callback = callback

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        self.callback()

def train(config: TrainConfig):
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    if config.code_src_list is not None:
        copy_code(config.output_dir, config.code_src_list)

    model, tokenizer = config.model, config.tokenizer

    generation_kwargs = {}
    if config.generation_kwargs is not None:
        generation_kwargs.update(config.generation_kwargs)


    def apply_chat_template(*args, **kwargs):
        raise RuntimeError("GRPO must not use apply_chat_template")

    # prevent TRL from using apply_chat_template
    tokenizer.apply_chat_template = apply_chat_template

    if config.mode == "prepare":
        # in prepare mode, always generate in full to monitor GPU memory
        generation_kwargs["min_new_tokens"] = generation_kwargs["max_completion_length"]

        # in prepare mode, train for at most 2 accumulation steps
        config.train_size = min(
            2 * config.batch_size * config.accumulation_steps,
            config.train_size,
        )

    # DATASET

    def train_generator():
        for i in range(config.train_size):
            yield {"prompt": config.processor.marshal_input(
                config.train_data(i),
            )}


    train_dataset = Dataset.from_generator(train_generator)
    eval_dataset = Dataset.from_list(list(map(
        lambda x: {"prompt": config.processor.marshal_input(x)},
        config.eval_data,
    )))

    has_cuda = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_available()
    use_vllm = False # TODO - enable vllm

    callback: Callable[[], None] = lambda: None

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
        generation_kwargs=generation_kwargs,

        use_vllm=use_vllm,
        vllm_mode="colocate",
        vllm_max_model_length=generation_kwargs.get("max_completion_length", None),

        gradient_checkpointing=True,
    )

    def reward_func(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        answers = list(map(config.processor.unmarshal_output, completions))
        inputs = list(map(config.processor.unmarshal_input, prompts))

        rewards = [config.reward_func(i, r, a) for i, (r, a) in zip(inputs, answers)]
        return rewards

    trainer = GRPOTrainer(
        args=training_args,
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_func, # type: ignore
        reward_processing_classes=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[OnSaveCallback(callback=callback)],
    )

    for sample in config.eval_data:
        print(sample)

    trainer.train()








