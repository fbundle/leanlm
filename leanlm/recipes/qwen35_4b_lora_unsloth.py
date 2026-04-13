import sys
from typing import Any, Literal

import jiwer
from mlx_tune import FastLanguageModel  # type: ignore


from leanlm.llm_trainer.processor import Qwen3Processor

from ..arithmetic.arithmetic import generate_input, get_expected_output
from ..llm_trainer.trainer import TrainConfig, train, Mode

class Kwargs:
    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
    def __dict__(self) -> dict[str, Any]:
        return self.kwargs


def load_model_and_tokenizer(model_path: str, max_completion_length: int):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_completion_length,
        dtype="auto",  # For auto-detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit=True,  # Use 4bit quantization to reduce memory usage. Can be False
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,  # Dropout = 0 is currently optimized
        bias="none",  # Bias = "none" is currently optimized
        use_gradient_checkpointing=True,
        random_state=3407,
    )

    return model, tokenizer

def reward_func(question: str, answer: str) -> float:
   expected = get_expected_output(question)
   return - jiwer.cer(expected, answer)

type MainMode = Literal["train", "prepare", "debug"]

def main(main_mode: MainMode):
    # memory ~ batch_size x num_generations x max_completion_length
    batch_size = 4
    num_generations = 8
    max_completion_length = 4096

    accumulation_steps = 32 // batch_size
    save_examples = 100 * batch_size * accumulation_steps
    save_steps =  save_examples // (batch_size * accumulation_steps)

    p, m = 0.3, 18

    train_size = 100000 * batch_size * accumulation_steps
    eval_size = batch_size * accumulation_steps
    eval_data = [generate_input(p, m) for _ in range(eval_size)]

    model_path = "Qwen/Qwen3.5-4B"
    debug_model_path = "Qwen/Qwen3.5-0.8B"
    output_dir = f"mnt/output/qwen3.5-4b-length{max_completion_length}-p{p}-lora-unsloth-calculator"
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
        eval_size = 5 * batch_size
        eval_data = [generate_input(p, m) for _ in range(eval_size)]

        model_path = debug_model_path
        output_dir = "mnt/output/test"
        deepspeed = None
    else:
        raise RuntimeError("mode")

    # END DEBUG

    model, tokenizer = load_model_and_tokenizer(model_path, max_completion_length=max_completion_length)

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