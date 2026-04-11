import os
import shutil
from typing import Any, Iterable, Callable, Literal

import torch
from datasets import Dataset
from pydantic import BaseModel, ConfigDict
from transformers import TrainingArguments, TrainerCallback, TrainerState, TrainerControl
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer

from huggingface_hub import login, upload_large_folder

Language = str

class Processor(object):
    def marshal_input(self, input_text: str) -> Language:
        raise NotImplementedError

    def unmarshal_input(self, prompt: Language) -> str:
        raise NotImplementedError

    def unmarshal_output(self, completion: Language) -> str:
        raise NotImplementedError

type Mode = Literal["prepare", "train"]
ModePrepare: Mode = "prepare"
ModeTrain: Mode = "train"

class TrainConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    mode: Mode

    hf_repo: str | None = None
    src_list: list[str] | None = []

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
    train_size: int
    train_data: Callable[[int], str]
    eval_data: list[str]

    deepspeed: str | None = None


def take(n: int, i: Iterable[Any]) -> Iterable[Any]:
    return (x for _, x in zip(range(n), i))


def upload(output_dir: str, repo_id: str, src_list: list[str] | None = None) -> Callable[[], None]:
    def helper():
        print(f"uploading to hf {repo_id}")

        if src_list is not None:
            for code_src in src_list:
                if not os.path.exists(code_src):
                    continue

                code_dst = f"{output_dir}/src/{code_src}"
                if os.path.exists(code_dst):
                    shutil.rmtree(code_dst)

                if not os.path.exists(os.path.dirname(code_dst)):
                    os.makedirs(os.path.dirname(code_dst))

                shutil.copytree(code_src, code_dst)

        login()
        upload_large_folder(
            folder_path=output_dir,
            repo_id=repo_id,
            repo_type="model",
        )

    return helper

class OnSaveCallback(TrainerCallback):
    def __init__(self, callback: Callable[[], None]):
        super().__init__()
        self.callback = callback

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        self.callback()





def train(config: TrainConfig):
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    model, tokenizer = config.model, config.tokenizer

    generation_kwargs = {
        "max_new_tokens": config.max_completion_length,

        "temperature": config.temperature,
        "top_p": config.top_p,
        "min_p": config.min_p,
        "top_k": config.top_k,

        "repetition_penalty": config.repetition_penalty,
    }

    def apply_chat_template(*args, **kwargs):
        raise RuntimeError("GRPO must not use apply_chat_template")

    # prevent TRL from using apply_chat_template
    tokenizer.apply_chat_template = apply_chat_template

    if config.mode == "prepare":
        # in prepare mode, always generate in full to monitor GPU memory
        generation_kwargs["min_new_tokens"] = config.max_completion_length

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

    callbacks: list[TrainerCallback] | None = None
    callback: Callable[[], None] = lambda: None
    if config.hf_repo is not None:
        callback = upload(
            output_dir=config.output_dir,
            repo_id=config.hf_repo,
            src_list=config.src_list,
        )
        callbacks = [OnSaveCallback(callback=callback)]

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
        callbacks=callbacks,
    )

    for sample in config.eval_data:
        print(sample)

    resume_from_checkpoint = get_last_checkpoint(config.output_dir)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    trainer.save_model(config.output_dir)

    callback()








