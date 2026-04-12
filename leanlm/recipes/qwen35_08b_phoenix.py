import sys
from typing import Any, Literal

import jiwer
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from leanlm.llm_trainer.processor import PhoenixQwen3Processor

from ..arithmetic.arithmetic import generate_input, get_expected_output
from ..llm_trainer.trainer import TrainConfig, train, Mode


class Kwargs:
    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
    def __dict__(self) -> dict[str, Any]:
        return self.kwargs

def reinitialize_model(model):
    for name, param in model.named_parameters():
        if param.dim() > 1:
            torch.nn.init.xavier_uniform_(param)
        else:
            torch.nn.init.zeros_(param)

def load_model_and_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path)
    if tokenizer.padding_side is None:
        tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token # <|im_end|>


    # frenzy flame - we burn everything
    # rising like a phoenix from the ashes
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_config(config)

    return model, tokenizer

def reward_func(question: str, answer: str) -> float:
   expected = get_expected_output(question)
   return - jiwer.cer(expected, answer)

type MainMode = Literal["train", "prepare", "debug"]

def main(main_mode: MainMode):
    # memory ~ batch_size x num_generations x max_completion_length
    batch_size = 1
    num_generations = 8
    max_completion_length = 4096

    accumulation_steps = 32 // batch_size
    save_examples = 100 * batch_size * accumulation_steps
    save_steps =  save_examples // (batch_size * accumulation_steps)

    p, m = 0.3, 18

    train_size = 100000 * batch_size * accumulation_steps
    eval_size = batch_size * accumulation_steps
    eval_data = [generate_input(p, m) for _ in range(eval_size)]

    model_path = "Qwen/Qwen3.5-0.8B"
    output_dir = f"mnt/output/qwen3.5-0.8b-length{max_completion_length}-p{p}-phoenix-calculator"
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

        batch_size = 4
        accumulation_steps = 2
        num_generations = 2

        max_completion_length = 16

        train_size = 1 * batch_size
        eval_size = 5 * batch_size
        eval_data = [generate_input(p, m) for _ in range(eval_size)]


        model_path = "Qwen/Qwen3.5-0.8B"
        output_dir = "mnt/output/qwen3.5-0.8b-lora-calculator"
        code_src_list = ["leanlm"]
        deepspeed = None
    else:
        raise RuntimeError("mode")

    # END DEBUG

    model, tokenizer = load_model_and_tokenizer(model_path)

    config = TrainConfig(
        mode=mode,

        code_src_list=code_src_list,

        output_dir=output_dir,
        processor=PhoenixQwen3Processor(),
        tokenizer=tokenizer,
        model=model,
        reward_func=reward_func,

        batch_size=batch_size,
        accumulation_steps=accumulation_steps,
        num_generations=num_generations,

        generation_kwargs=Kwargs(
            max_completion_length=max_completion_length,
            temperature=0.6,
            top_p=0.95,
            min_p=0.0,
            top_k=20,
            repetition_penalty=1.0,
        ).__dict__(),

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