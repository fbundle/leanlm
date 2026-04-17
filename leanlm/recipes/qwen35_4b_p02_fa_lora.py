import sys
from typing import Literal

import jiwer
from peft import LoraConfig, get_peft_model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from leanlm.llm_trainer.processor import Qwen3Processor
from ..arithmetic.arithmetic import generate_input, get_expected_output
from ..llm_trainer.trainer import TrainConfig, train, Mode

def load_model_and_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path)
    if tokenizer.padding_side is None:
        tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token # <|im_end|>

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
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

def reward_func(question: str, reason: str, answer: str) -> float:
    expected = get_expected_output(question)
    f = lambda x: 1 / (1 + x) # convert [0, +inf] -> [1, 0]

    # cer reward
    cer = jiwer.cer(expected, answer)
    cer_reward = f(cer)

    # arithmetic reward
    e: int = int(expected)
    a: int | None = None
    try:
        a = int(answer)
    except ValueError:
        pass

    if a is None:
        arith_reward = 0
    else:
        e = max(1, abs(e))
        diff = abs((a - e) / e)
        arith_reward = f(diff)

    
    return cer_reward + arith_reward

type RunMode = Literal["train", "prepare", "debug"]

def main(mode: RunMode, uuid: str):
    # memory ~ batch_size x num_generations x max_completion_length^n
    batch_size = 4
    num_generations = 8
    max_completion_length = 2048

    accumulation_steps = 32 // batch_size
    save_examples = 100 * batch_size * accumulation_steps
    save_steps =  save_examples // (batch_size * accumulation_steps)

    p1, p2, m = 0.2, 0.3, 18

    train_size = 10000 * batch_size * accumulation_steps
    curriculum_length = 600 * batch_size * accumulation_steps
    
    def train_data(i: int) -> str:
        # linear function from 0 -> curriculum_length
        # fixed at curriculum_length onwards
        if i < curriculum_length:
            p = p2 - (p2 - p1) * i / curriculum_length
        else:
            p = p1
    
        return generate_input(p, m)


    model_path = "Qwen/Qwen3.5-4B"
    debug_model_path = "Qwen/Qwen3.5-0.8B"
    output_dir = f"mnt/output/qwen3.5-4b-length{max_completion_length}-p{p1}-{uuid}-lora-calculator"
    code_src_list = ["leanlm"]
    deepspeed = "conf/ds_zero2.json"

    # DEBUG
    if mode == "train":
        train_mode: Mode = "train"
    elif mode == "prepare":
        train_mode: Mode = "prepare"
        print("###### PREPARE MODE #######")
    elif mode == "debug":
        train_mode: Mode = "train"
        print("###### DEBUG MODE #######")

        batch_size = 1
        accumulation_steps = 2
        num_generations = 2

        max_completion_length = 16

        train_size = 1 * batch_size

        model_path = debug_model_path
        output_dir = "mnt/output/test"
        deepspeed = None
    else:
        raise RuntimeError("mode")

    # END DEBUG

    model, tokenizer = load_model_and_tokenizer(model_path)

    config = TrainConfig(
        train_mode=train_mode,

        code_src_list=code_src_list,

        output_dir=output_dir,
        processor=Qwen3Processor(),
        tokenizer=tokenizer,
        model=model,
        reward_func=reward_func,

        batch_size=batch_size,
        accumulation_steps=accumulation_steps,
        num_generations=num_generations,

        generation_kwargs=dict(),
        train_config_kwargs=dict(
            max_completion_length=max_completion_length,
            temperature=1.0,
            learning_rate = 1e-6,
            weight_decay = 0.001,
        ),

        save_steps=save_steps,
        train_size=train_size,
        train_data=train_data,

        deepspeed=deepspeed,
    )

    train(config)

if __name__ == "__main__":
    MODE = sys.argv[1]
    UUID = ""
    if len(sys.argv) >= 3:
        UUID = sys.argv[2]
    if MODE not in ["train", "prepare", "debug"]:
        raise RuntimeError("mode")

    main(MODE, UUID)
