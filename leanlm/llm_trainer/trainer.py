import os
import shutil
import platform
from typing import Any, Iterable, Callable, Literal

import torch
from datasets import Dataset
from pydantic import BaseModel, ConfigDict
from transformers import TrainingArguments, TrainerCallback, TrainerState, TrainerControl
from transformers.trainer_utils import get_last_checkpoint

from leanlm.llm_trainer.processor import Processor

from trl import GRPOConfig, GRPOTrainer # type: ignore

from dotenv import load_dotenv
load_dotenv()


type Mode = Literal["prepare", "train"]
ModePrepare: Mode = "prepare"
ModeTrain: Mode = "train"

class TrainConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    train_mode: Mode = "train"

    code_src_list: list[str] | None = []

    output_dir: str
    processor: Processor
    tokenizer: Any # TODO - change to something that has .encode and .decode
    model: Any # TODO - change to something that has .generate

    reward_func: Callable[[str, str, str], float]

    batch_size: int
    accumulation_steps: int = 1
    num_generations: int

    generation_kwargs: dict[str, Any] | None = None
    train_config_kwargs: dict[str, Any] | None = None

    save_steps: int
    train_size: int
    train_data: Callable[[int], str]
    eval_data: list[str] | None = None # DEPRECATED

    deepspeed: str | None = None


def take(n: int, i: Iterable[Any]) -> Iterable[Any]:
    return (x for _, x in zip(range(n), i))


def copy_code(output_dir: str, code_src_list: list[str]):
    print(f"copying code {code_src_list}")
    for code_src in code_src_list:
        if not os.path.exists(code_src):
            continue
        code_dst = f"{output_dir}/src/{code_src}"
        shutil.copytree(code_src, code_dst, dirs_exist_ok=True)

class OnSaveCallback(TrainerCallback):
    def __init__(self, callback: Callable[[], None]):
        super().__init__()
        self.callback = callback

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        self.callback()

class GPUMemoryCallback(TrainerCallback):
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if torch.cuda.is_available():
            # Get current and peak memory
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            peak = torch.cuda.max_memory_allocated() / 1024**3

            print(
                f"\n[GPU Memory] Step {state.global_step}: "
                f"Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB | Peak: {peak:.2f}GB"
            )

            # Reset peak memory stats for the next step if you want per-step peak
            # torch.cuda.reset_peak_memory_stats()

def get_hf_info() -> tuple[str, str] | None:
    hf_user = os.environ.get("HF_USER", default=None)
    hf_token = os.environ.get("HF_TOKEN", default=None)
    if hf_user is not None and hf_token is not None:
        return hf_user, hf_token
    return None

def train(config: TrainConfig):
    from accelerate import PartialState
    rank = PartialState().process_index

    # warning
    if config.eval_data is not None:
        print("DEPRECATED - config.eval_data")

    if rank == 0:
        os.makedirs(config.output_dir, exist_ok=True)
        if config.code_src_list is not None:
            copy_code(config.output_dir, config.code_src_list)

    # train
    if platform.system() == "Linux" and platform.machine() == "x86_64":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for training on Linux x86_64 but not found.")

    push_to_hub = False
    hf_model = None
    hf_token = None
    hf_info = get_hf_info()
    if hf_info is not None:
        push_to_hub = True
        hf_user, hf_token = hf_info
        hf_model = hf_user + "/" + os.path.basename(config.output_dir)
    else:
        print("WARNING: not pushing to huggingface")

    model, tokenizer = config.model, config.tokenizer

    generation_kwargs = {}
    if config.generation_kwargs is not None:
        generation_kwargs.update(config.generation_kwargs)
    train_config_kwargs = {}
    if config.train_config_kwargs is not None:
        train_config_kwargs.update(config.train_config_kwargs)


    def apply_chat_template(*args, **kwargs):
        raise RuntimeError("GRPO must not use apply_chat_template")

    # prevent TRL from using apply_chat_template
    tokenizer.apply_chat_template = apply_chat_template

    if config.train_mode == "prepare":
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

    has_cuda = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_available()
    use_vllm = False # TODO - enable vllm

    callback: Callable[[], None] = lambda: None

    training_args = GRPOConfig(
        output_dir=config.output_dir,
        num_train_epochs=1,

        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.accumulation_steps,
        num_generations=config.num_generations,

        # floating point precision
        bf16=has_cuda or has_mps,
        tf32=has_cuda,

        # log and eval
        logging_strategy="steps",
        logging_steps=max(config.save_steps // 10, 1),
        save_strategy="steps",
        save_steps=config.save_steps,
        
        eval_strategy="no",

        # hugging face
        push_to_hub=push_to_hub,
        hub_model_id=hf_model,
        hub_token=hf_token,
        hub_strategy="every_save",
        hub_always_push=True,
        report_to="tensorboard",

        # generation
        generation_kwargs=generation_kwargs,

        use_vllm=use_vllm,
        vllm_mode="colocate",
        vllm_max_model_length=generation_kwargs.get("max_completion_length", None),

        gradient_checkpointing=True,

        **train_config_kwargs,
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
        train_dataset=train_dataset, # type: ignore
        callbacks=[OnSaveCallback(callback=callback), GPUMemoryCallback()],
    )

    trainer.train(resume_from_checkpoint=get_last_checkpoint(config.output_dir))










