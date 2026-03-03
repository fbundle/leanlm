import os
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model

from .arithmetic import generate_input, match_output


TRAIN_SIZE = 100000
EVAL_SIZE = 100
OUTPUT_DIR = "mnt/output/calculator_qwen3p5_4b_lora"
MODEL_PATH = "Qwen/Qwen3.5-4B"
THINK_END = "</think>"

DEEPSPEED = "conf/ds_zero2.json"

BATCH_SIZE = 32
SAVE_STEPS = 50
MAX_COMPLETION_LENGTH = 512

def get_prompt_from_input_str(input_str: str) -> str:
    # tokenizer.apply_chat_template(
    #   conversation=[{"role": "user", "content": input_str}],
    #   tokenize=False,
    #   add_generation_prompt=True,
    #)
    return f"<|im_start|>user\n{input_str}<|im_end|>\n<|im_start|>assistant\n<think>\n" # qwen3.5

def get_input_str_from_prompt(prompt: str) -> str:
    return prompt.lstrip("<|im_start|>user\n").rstrip("<|im_end|>\n<|im_start|>assistant\n<think>\n") # qwen3.5

def reward_func(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
    # PARSE
    # completion must be in format
    # <reasoning>[SEP]<answer>
    def parse_completion(completion: str) -> str:
        answer = completion.split(THINK_END)[-1] # choose text segment after the last [SEP]
        return answer
    answers = list(map(parse_completion, completions))

    parse_prompt = get_input_str_from_prompt
    inputs = list(map(parse_prompt, prompts))

    # MATCH
    rewards = [match_output(input_str, answer) for input_str, answer in zip(inputs, answers)]
    return rewards

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=MODEL_PATH,
    )
    if tokenizer.padding_side is None:
        tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=MODEL_PATH,
    )
    lora_kwargs = {
        "r": 8,
        "lora_alpha": 16,
        "target_modules": ["q_proj", "v_proj"],
        "lora_dropout": 0.05,
        "bias": "none",
        "inference_mode": False,
        "task_type": "CAUSAL_LM",
    }

    lora_config = LoraConfig(**lora_kwargs)
    model = get_peft_model(model, lora_config)

    return model



def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    tokenizer = load_tokenizer()
    model = load_model()

    def train_generator():
        for _ in range(TRAIN_SIZE):
            yield {"prompt": get_prompt_from_input_str(generate_input())}

    train_dataset = Dataset.from_generator(train_generator)

    eval_data = [{
        "prompt": get_prompt_from_input_str(generate_input())
    } for _ in range(EVAL_SIZE)]
    eval_dataset = Dataset.from_list(eval_data)

    has_cuda = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_available()
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_generations=8,

        # optimizer
        deepspeed=DEEPSPEED if has_cuda else None,
        bf16=has_cuda or has_mps,
        tf32=has_cuda,

        # log and eval
        logging_strategy="steps",
        logging_steps=SAVE_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,

        # generation
        # max_completion_length=38912,
        max_completion_length=1024,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        repetition_penalty=1.0,

        # vllm
        # use_vllm=True,
        # vllm_mode="colocate",

        gradient_checkpointing=True,
    )

    trainer = GRPOTrainer(
        args=training_args,
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_func,
        reward_processing_classes=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print(eval_data)

    trainer.train(resume_from_checkpoint=False)

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
