import sys
from typing import Literal

import jiwer
import torch
from transformers import AutoTokenizer, Qwen3_5TextConfig, Qwen3_5ForCausalLM

from leanlm.llm_trainer.processor import Qwen3Processor
from ..arithmetic.arithmetic import generate_input, get_expected_output
from ..llm_trainer.trainer import TrainConfig, train, Mode

def load_model_and_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path)
    if tokenizer.padding_side is None:
        tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token # <|im_end|>

    # frenzy flame - we burn everything
    # rising like a phoenix from the ashes
    model = Qwen3_5ForCausalLM(Qwen3_5TextConfig.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
    ))

    return model, tokenizer

def reward_func(question: str, reason: str, answer: str) -> float:
    expected = get_expected_output(question)
    cer = jiwer.cer(expected, answer)
    return 1 - min(cer, 1.0)


type MainMode = Literal["train", "prepare", "debug"]

def main(main_mode: MainMode):
    # memory ~ batch_size x num_generations x max_completion_length
    batch_size = 4
    num_generations = 4
    max_completion_length = 2048

    accumulation_steps = 32 // batch_size
    save_examples = 10 * batch_size * accumulation_steps
    save_steps =  save_examples // (batch_size * accumulation_steps)

    p, m = 0.3, 18

    train_size = 100000 * batch_size * accumulation_steps
    eval_size = batch_size * accumulation_steps
    eval_data = [generate_input(p, m) for _ in range(eval_size)]

    model_path = "Qwen/Qwen3.5-4B"
    debug_model_path = "Qwen/Qwen3.5-0.8B"
    output_dir = f"mnt/output/qwen3.5-4b-length{max_completion_length}-p{p}-phoenix-calculator"
    code_src_list = ["leanlm"]
    deepspeed = None # only for multi GPUs "conf/ds_zero2.json"

    # DEBUG
    if main_mode == "train":
        mode: Mode = "train"
    elif main_mode == "prepare":
        mode: Mode = "prepare"
        print("###### PREPARE MODE #######")
    elif main_mode == "debug":
        mode: Mode = "train"
        print("###### DEBUG MODE #######")

        batch_size = 1
        accumulation_steps = 2
        num_generations = 2

        max_completion_length = 16

        train_size = 1 * batch_size
        eval_size = 1 * batch_size
        eval_data = [generate_input(p, m) for _ in range(eval_size)]

        model_path = debug_model_path
        output_dir = "mnt/output/test"
        deepspeed = None
    else:
        raise RuntimeError("mode")

    # END DEBUG

    model, tokenizer = load_model_and_tokenizer(model_path)

    config = TrainConfig(
        mode=mode,

        code_src_list=code_src_list,

        output_dir=output_dir,
        processor=Qwen3Processor(),
        tokenizer=tokenizer,
        model=model,
        reward_func=reward_func,

        batch_size=batch_size,
        accumulation_steps=accumulation_steps,
        num_generations=num_generations,

        generation_kwargs=dict(
            max_completion_length=max_completion_length,
            temperature=1.0,
        ),
        train_config_kwargs=dict(
            # beta=0.001, # phoenix has no beta
            learning_rate=5e-5,
            weight_decay=0.001,
        ),

        save_steps=save_steps,
        train_size=train_size,
        train_data=lambda _: generate_input(p, m),
        eval_data=eval_data,

        deepspeed=deepspeed,
    )

    train(config)

if __name__ == "__main__":
    argv = sys.argv[1]
    if argv in ["train", "prepare", "debug"]:
        main(argv) # type: ignore
    else:
        raise RuntimeError("mode")
