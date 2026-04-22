import platform
import random
import sys
from typing import Literal

import jiwer
from peft import LoraConfig, get_peft_model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import PartialState


import math
import jiwer

from leanlm.env_trainer.dataset import LazyDataset
from leanlm.env_trainer.environment import Action, Env, Seed, StateDelta
from leanlm.env_trainer.rollout import TransformerRolloutModel
from leanlm.env_trainer.trainer import train
from leanlm.env_trainer.trainer_config import Mode, TrainConfig
from leanlm.env_trainer.processor import qwen3_instruct_processor

def get_int(s: str) -> int | None:
    try:
        return int(s)
    except Exception:
        return None

def get_int2(s: str) -> tuple[int, int, bool]:
    try:
        a, b = s.split()
        return int(a), int(b), True
    except Exception:
        return 0, 0, False


class GcdEnv(Env):
    def reset(self, seed: Seed) -> StateDelta:
        a_str, b_str = seed.split()
        a, b = int(a_str), int(b_str)
        self.gcd = math.gcd(a, b)
        self.reward = 0
        self.terminate = False
        return f"""
calculate the GCD of {a} and {b}
every turn, you are able to either output the answer by
ANSWER <answer>
or output 
SUBTRACT <number1> <number2>
I will help you to calculate the difference between two numbers with absolute precision
"""
    
    def step(self, action: Action) -> StateDelta:
        f = lambda x: 1 / (1 + x) # map [0, inf) -> [1, 0)
        parts = action.split("SUBTRACT ")
        if len(parts) >= 2: # detected subtract
            last = parts[-1]
            a, b, ok = get_int2(last)
            if not ok:
                format_reward, answer_reward = 0, 0
                self.reward = format_reward + answer_reward
                self.terminate = True
                return f"subtract_format_error: {last}"
            else:
                format_reward = f(jiwer.cer(f"{a} {b}", last))
                answer_reward = 0
                self.reward = format_reward + answer_reward
                return f"subtract: {a} - {b} = {a - b}"
        
        parts = action.split("ANSWER ")
        if len(parts) >= 2: # detected answer
            self.terminate = True
            last = parts[-1]
            answer = get_int(last)
            if answer is None:
                format_reward, answer_reward = 0, 0
                self.reward = format_reward + answer_reward
                
                return f"answer_format_error: {last}"
            else:
                format_reward = f(jiwer.cer(str(answer), last))
                answer_reward = 1 if answer == self.gcd else 0
                self.reward = format_reward + answer_reward
                if answer_reward == 1:
                    return f"answer_correct: {answer}"
                else:
                    return f"answer_wrong: {answer}"
        
        format_reward, answer_reward = 0, 0
        self.reward = format_reward + answer_reward
        self.terminate = True
        return "format_error"

def load_model_and_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path)
    if tokenizer.padding_side is None:
        tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token # <|im_end|>

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
    )
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        inference_mode=False,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer


def main(train_mode: Mode, uuid: str, debug: bool):
    num_processes = PartialState().num_processes

    # model updates every effective_batch_size
    effective_batch_size = 32

    max_turn_length = 1024
    # per device memory ~ batch_size x num_generations x max_completion_length^\alpha
    # alpha = 2 for usual transformer
    # alpha = 1 for flash attention
    per_device_batch_size = 1
    num_generations = 8
    max_completion_length = 16384

    if debug:
        effective_batch_size = 4
        num_generations = 4
        per_device_batch_size = 1

    gradient_accumulation_steps = effective_batch_size // (per_device_batch_size * num_processes)

    assert effective_batch_size == per_device_batch_size * gradient_accumulation_steps * num_processes

    # train 10000 batches
    train_size = 1000 * effective_batch_size

    


    # train data generation
    # total_num_steps = train_size x num_generations / effective_batch_size
    #       = 8000
    # no_points_per_step = effective_batch_size / num_generations
    
    def f(i: int) -> str:
        a = random.randrange(1000)
        b = random.randrange(1000)
        c = random.randrange(1000)
        return f"{a * b} {b * c}"
    
    data = LazyDataset[str](n=train_size, f=f)

    model_path = "Qwen/Qwen3.5-0.8B"
    output_dir = f"mnt/output/qwen3.5-0.8b-tl{max_turn_length}-cl{max_completion_length}-b{effective_batch_size}-{uuid}-lora-gcd"
    deepspeed = "conf/ds_zero2.json"
    if debug:
        deepspeed = None       

    rule =f"""
every turn, you can output a maximum number of {max_turn_length} tokens
the whole conversation should not last longer than {max_completion_length} tokens
"""


    model, tokenizer = load_model_and_tokenizer(model_path)

    config = TrainConfig(
        mode=train_mode,
        deepspeed=deepspeed,
        output_dir=output_dir,
        processor=qwen3_instruct_processor,
        system_prompt=rule,
        model=TransformerRolloutModel(
            tokenizer=tokenizer,
            model=model, # type: ignore
            generation_kwargs=dict(
                temperature=1.0,
            ),
        ),
        data=data,
        env_factory=GcdEnv,
        max_turn_length=max_turn_length,

        per_device_batch_size=per_device_batch_size,
        
        num_generations=num_generations,
        max_conversation_length=max_completion_length,
        gradient_accumulation_steps=gradient_accumulation_steps,

        generation_kwargs=dict(),
        train_config_kwargs=dict(
            temperature=1.0,
            learning_rate = 1e-6,
            weight_decay = 0.001,
        ),

        save_every_seconds=3600,
        log_every_seconds=0,
    )

    train(config)

if __name__ == "__main__":
    MODE = sys.argv[1]
    UUID = "test"
    if len(sys.argv) >= 3:
        UUID = sys.argv[2]
    if MODE not in ["train", "prepare", "debug"]:
        raise RuntimeError("mode")

    if MODE == "debug":
        main("train", UUID, debug=True)
    else:
        main(MODE, UUID, debug=False) # type: ignore
