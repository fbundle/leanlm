import os
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model
import jiwer

from .arithmetic import generate_input, get_expected_output

DEBUG = False


import sys
if len(sys.argv) >= 2 and sys.argv[1] == "DEBUG":
    DEBUG = True



OUTPUT_DIR = "mnt/output/calculator_qwen3_0p6b_lora"
MODEL_PATH = "Qwen/Qwen3-0.6B"

DEEPSPEED = "conf/ds_zero2.json"

BATCH_SIZE = 8
ACCUMULATION_STEPS = 32 // BATCH_SIZE # effective batch size 32
ACCUMULATION_STEPS = 1
if DEBUG:
    BATCH_SIZE = 2

# each sample costs about NUM_GENERATIONS x MAX_COMPLETION_LENGTH

MAX_COMPLETION_LENGTH = 32768
MAX_COMPLETION_LENGTH = 4096
NUM_GENERATIONS = 8
if DEBUG:
    NUM_GENERATIONS = 2

SAVE_STEPS = 50
TRAIN_SIZE = 10000 * BATCH_SIZE
EVAL_SIZE = 8 * BATCH_SIZE

def get_prompt_from_input_str(input_str: str) -> str:
    return f"<|im_start|>user\n{input_str}<|im_end|>\n<|im_start|>assistant\n<think>\n" # qwen3 qwen3.5
    return tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": input_str}],
        tokenize=False,
        add_generation_prompt=True,
    )

def get_input_str_from_prompt(prompt: str) -> str:
    return prompt.lstrip("<|im_start|>user\n").rstrip("<|im_end|>\n<|im_start|>assistant\n<think>\n") # qwen3 qwen3.5


def get_output_str_from_completion(completion: str) -> str:
    # completion must be in format
    # reasoning</think>answer
    output_str = completion.split("</think>")[-1] # choose text segment after the last </think>
    return output_str

def reward_func(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
    answers = list(map(get_output_str_from_completion, completions))
    inputs = list(map(get_input_str_from_prompt, prompts))
    expected_answers = list(map(get_expected_output, inputs))
    
    rewards = [
        - jiwer.cer(expected_answer, answer)
        for expected_answer, answer in zip(expected_answers, answers)
    ]
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
        # attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
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
        gradient_accumulation_steps=ACCUMULATION_STEPS,
        per_device_eval_batch_size=BATCH_SIZE,
        num_generations=NUM_GENERATIONS,

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
        eval_on_start=False,

        # generation config recommented by qwen3.5
        # max_completion_length=262144, # default context length for qwen3.5
        max_completion_length=MAX_COMPLETION_LENGTH,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        # presence_penalty=0.0,
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

    trainer.train(resume_from_checkpoint=True)

    trainer.save_model(OUTPUT_DIR)
    # tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
