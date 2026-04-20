import os
import shutil
import platform
import time
from typing import Any, Callable, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field
from transformers import TrainingArguments, TrainerCallback, TrainerState, TrainerControl
from transformers.trainer_utils import get_last_checkpoint

from leanlm.llm_trainer.dataset import LazyDataset
from leanlm.llm_trainer.processor import Processor

from trl import GRPOConfig, GRPOTrainer # type: ignore

from accelerate import PartialState

from dotenv import load_dotenv
load_dotenv()


def copy_code(output_dir: str, code_src_list: list[str]):
    print(f"copying code {code_src_list}")
    for code_src in code_src_list:
        if not os.path.exists(code_src):
            continue
        code_dst = f"{output_dir}/src/{code_src}"
        shutil.copytree(code_src, code_dst, dirs_exist_ok=True)


type Mode = Literal["prepare", "train"]
ModePrepare: Mode = "prepare"
ModeTrain: Mode = "train"

class TrainConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    train_mode: Mode = "train"

    code_src_list: list[str] | None = []

    output_dir: str
    processor: Processor
    tokenizer: Any # change to something that has .encode and .decode
    model: Any # change to something that has .generate

    reward_func: Callable[[str, str, str], float]
    train_data: LazyDataset[str]


    # per device memory ~ batch_size x num_generations x max_completion_length^\alpha
    per_device_batch_size: int
    num_generations: int
    max_completion_length: int
    # gradient accumulation in every: num_processes x per_device_batch_size x gradient_accumulation_steps
    gradient_accumulation_steps: int = 1


    # on_step_end
    save_every_seconds: int = 1 * 3600    # by default, save every 1 hour
    log_every_seconds: int = 0            # by default, log immediately after step_end

    # others
    deepspeed: str | None = None
    generation_kwargs: dict[str, Any] | None = None
    train_config_kwargs: dict[str, Any] | None = None

    # DEPRECATED
    save_steps: int = Field(1, deprecated="use save_every_seconds instead")
    log_steps: int = Field(1, deprecated="use log_every_seconds instead")


class Callback(TrainerCallback):
    def __init__(self, config: TrainConfig, rank: int):
        self.rank = rank
        self.save_every_seconds = config.save_every_seconds
        self.log_every_seconds = config.log_every_seconds
        

        self.last_save_time = time.time()
        self.last_log_time = time.time()
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # get GPU utilization
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            peak = torch.cuda.max_memory_allocated() / 1024**3

            trainer = kwargs.get("trainer", None)
            if trainer is not None:
                trainer.log({
                    f"gpu/rank_{self.rank}/allocated_gb": allocated,
                    f"gpu/rank_{self.rank}/reserved_gb": reserved,
                    f"gpu/rank_{self.rank}/peak_gb": peak,
                })
            
            # torch.cuda.reset_peak_memory_stats() # we want peak to be accumulative

        # trigger log and save
        current_time = time.time()

        if current_time - self.last_log_time >= self.log_every_seconds:
            control.should_log = True
            self.last_log_time = current_time
        
        if current_time - self.last_save_time >= self.save_every_seconds:
            control.should_save = True
            self.last_save_time = current_time

def get_hf_info(output_dir: str) -> tuple[bool, str, str]:
    hf_user = os.environ.get("HF_USER", default=None)
    hf_token = os.environ.get("HF_TOKEN", default=None)
    if hf_user is None or hf_token is None:
        return False, "", ""
    
    hf_model = hf_user + "/" + os.path.basename(output_dir)
    return True, hf_model, hf_token

def train(config: TrainConfig):
    rank = PartialState().process_index

    if rank == 0:
        os.makedirs(config.output_dir, exist_ok=True)
        if config.code_src_list is not None:
            copy_code(config.output_dir, config.code_src_list)

    # train
    if platform.system() == "Linux" and platform.machine() == "x86_64":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for training on Linux x86_64 but not found.")

    push_to_hub, hf_model, hf_token = get_hf_info(config.output_dir)

    generation_kwargs = {}
    if config.generation_kwargs is not None:
        generation_kwargs.update(config.generation_kwargs)
    train_config_kwargs = {}
    if config.train_config_kwargs is not None:
        train_config_kwargs.update(config.train_config_kwargs)


    def apply_chat_template(*args, **kwargs):
        raise RuntimeError("GRPO must not use apply_chat_template")

    # prevent TRL from using apply_chat_template
    config.tokenizer.apply_chat_template = apply_chat_template

    if config.train_mode == "prepare":
        # in prepare mode, always generate in full to monitor GPU memory
        generation_kwargs["min_new_tokens"] = config.max_completion_length

    # DATASET
    train_dataset = config.train_data.map(
        lambda input_text: {"prompt": config.processor.marshal_input(input_text)}
    )




    has_cuda = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_available()

    callbacks: list[TrainerCallback] = [Callback(config=config, rank=rank)]

    training_args = GRPOConfig(
        output_dir=config.output_dir,
        num_train_epochs=1,
        deepspeed=config.deepspeed,

        per_device_train_batch_size=config.per_device_batch_size,
        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        gradient_accumulation_steps=config.gradient_accumulation_steps,

        # floating point precision
        bf16=has_cuda or has_mps,
        tf32=has_cuda,

        # no eval
        eval_strategy="no",

        # log and save - set a big number as we manually save and log
        save_strategy="epoch",
        logging_strategy="epoch",


        # hugging face
        push_to_hub=push_to_hub,
        hub_model_id=hf_model,
        hub_token=hf_token,
        hub_strategy="every_save",
        hub_always_push=True,
        report_to="tensorboard",

        use_vllm=False, # may change to true in the future
        vllm_mode="colocate",

        gradient_checkpointing=True,

        # others
        generation_kwargs=generation_kwargs,
        **train_config_kwargs,
    )

    def reward_func(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        answers = list(map(config.processor.unmarshal_output, completions))
        inputs = list(map(config.processor.unmarshal_input, prompts))

        rewards = [config.reward_func(i, r, a) for i, (r, a) in zip(inputs, answers)]
        return rewards

    trainer = GRPOTrainer(
        args=training_args,
        model=config.model,
        processing_class=config.tokenizer,
        reward_funcs=reward_func, # type: ignore
        reward_processing_classes=config.tokenizer,
        train_dataset=train_dataset, # type: ignore
        callbacks=callbacks,
    )

    trainer.train(resume_from_checkpoint=get_last_checkpoint(config.output_dir))










