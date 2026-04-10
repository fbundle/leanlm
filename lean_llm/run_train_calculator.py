import os
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model
import jiwer


from .arithmetic import generate_input, get_expected_output

RELEASE = True

import sys

if len(sys.argv) >= 2 and sys.argv[1] == "test":
    RELEASE = False

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


TOKEN_TYPE = "qwen3"
OUTPUT_DIR = "mnt/output/qwen3.5-0.8b-lora-calculator"
MODEL_PATH = "Qwen/Qwen3.5-0.8B"
LORA_FT = True

TOKEN_TYPE = "qwen3"
OUTPUT_DIR = "mnt/output/qwen3-0.6b-lora-calculator"
MODEL_PATH = "Qwen/Qwen3-0.6B"
LORA_FT = True

TOKEN_TYPE = "gemma"
OUTPUT_DIR = "mnt/output/gemma-4-E2B-it-lora-calculator"
MODEL_PATH = "google/gemma-4-E2B-it"
LORA_FT = False

TOKEN_TYPE = "qwen2.5_instruct"
OUTPUT_DIR = "mnt/output/qwen2.5-0.5b-lora-calculator"
MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_FT = True

TOKEN_TYPE = "deepseek_r1"
OUTPUT_DIR = "mnt/output/deepseek_r1-1.5b-lora-calculator"
MODEL_PATH = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
LORA_FT = True

TOKEN_TYPE = "qwen2.5_instruct"
OUTPUT_DIR = "mnt/output/qwen2.5-0.5b-lora-calculator"
MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_FT = True

if RELEASE:
    TOKEN_TYPE = "qwen3"
    OUTPUT_DIR = "mnt/output/qwen3.5-4b-lora-calculator"
    MODEL_PATH = "Qwen/Qwen3.5-4B"
    LORA_FT = True

DEEPSPEED = "conf/ds_zero2.json"

BATCH_SIZE = 4 # about 1 for non LORA
ACCUMULATION_STEPS = 8

# each sample costs about NUM_GENERATIONS x MAX_COMPLETION_LENGTH

MAX_COMPLETION_LENGTH = 128
NUM_GENERATIONS = 2

if RELEASE:
    MAX_COMPLETION_LENGTH = 4096
    NUM_GENERATIONS = 8

SAVE_STEPS = 100
TRAIN_SIZE = 100000 * BATCH_SIZE
EVAL_SIZE = 50 * BATCH_SIZE


def get_prompt_from_input_str(input_str: str) -> str:
    if TOKEN_TYPE == "qwen3":
        # qwen 3.5
        return f"<|im_start|>user\n{input_str}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    
    elif TOKEN_TYPE == "gemma":
        # gemma-4-E2B-it
        return f"<bos><|turn>system\n<|think|><turn|>\n<|turn>user\n{input_str}<turn|>\n<|turn>model\n"
    
    elif TOKEN_TYPE == "qwen2.5_instruct":
        return f"<|im_start|>user\n{input_str}<|im_end|>\n<|im_start|>assistant\n"
    
    elif TOKEN_TYPE == "deepseek_r1":
        return f"<｜begin▁of▁sentence｜><｜User｜>{input_str}<｜Assistant｜><think>\n"
    
    else:
        raise NotImplemented

    return tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": input_str}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def get_input_str_from_prompt(prompt: str) -> str:
    if TOKEN_TYPE == "qwen3":
        # qwen 3.5
        return prompt.lstrip("<|im_start|>user\n").rstrip("<|im_end|>\n<|im_start|>assistant\n<think>\n")
    
    elif TOKEN_TYPE == "gemma":
        # gemma-4-E2B-it
        return prompt.lstrip("<bos><|turn>system\n<|think|><turn|>\n<|turn>user\n").rstrip("<turn|>\n<|turn>model\n")
    
    elif TOKEN_TYPE == "qwen2.5_instruct":
        return prompt.lstrip("<|im_start|>user\n").rstrip("<|im_end|>\n<|im_start|>assistant\n")
    
    elif TOKEN_TYPE == "deepseek_r1":
        return prompt.lstrip("<｜begin▁of▁sentence｜><｜User｜>").rstrip("<｜Assistant｜><think>\n")
    
    else:
        raise NotImplemented


def get_output_str_from_completion(completion: str) -> str:
    if TOKEN_TYPE == "qwen3":
        # qwen 3.5
        # completion is in the format
        # reasoning </think> answer <|im_end|>
        completion = completion.split("</think>")[-1]
        completion = completion.split("<|im_end|>")[0]
        return completion
    
    elif TOKEN_TYPE == "gemma":
        # gemma-4-E2B-it
        # completion is in the format
        # <|channel>reasoning<channel|>answer<turn|>
        
        completion = completion.split("<channel|>")[-1]
        completion = completion.split("<turn|>")[0]
        return completion
    
    elif TOKEN_TYPE == "qwen2.5_instruct":
        # qwen 2.5
        # completion is in the format
        # reasoning = answer <|im_end|>

        completion = completion.split("=")[-1]
        completion = completion.split("<|im_end|>")[0]
        return completion
    
    elif TOKEN_TYPE == "deepseek_r1":
        # completion is in the format
        # reasoning</think>answer
        completion = completion.split("</think>")[-1]
        return completion
    
    else:
        raise NotImplemented


def reward_func(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
    answers = list(map(get_output_str_from_completion, completions))
    inputs = list(map(get_input_str_from_prompt, prompts))
    expected_answers = list(map(get_expected_output, inputs))

    rewards = [
        - jiwer.cer(expected_answer, answer)
        for expected_answer, answer in zip(expected_answers, answers)
    ]
    return rewards


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_PATH)
    if tokenizer.padding_side is None:
        tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=MODEL_PATH,
        # attn_implementation="flash_attention_2",
        dtype=torch.float16,
    )
    if not LORA_FT:
        return model, tokenizer

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

    return model, tokenizer

def runtime_error(message: str = "assertion"):
    raise RuntimeError(f"RuntimeError: {message}")
    return f"RuntimeError: {message}"

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    model, tokenizer = load_model_and_tokenizer()

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
    use_vllm = False

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=ACCUMULATION_STEPS,
        per_device_eval_batch_size=BATCH_SIZE,
        num_generations=NUM_GENERATIONS,

        # optimizer
        # deepspeed=DEEPSPEED if has_cuda else None,
        bf16=has_cuda or has_mps,
        tf32=has_cuda,

        # log and eval
        logging_strategy="steps",
        logging_steps=SAVE_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        # eval_strategy="steps",
        eval_strategy="no",
        eval_steps=SAVE_STEPS,
        eval_on_start=False,

        # generation config recommented by qwen3.5
        # max_completion_length=262144, # default context length for qwen3.5
        max_completion_length=MAX_COMPLETION_LENGTH,
        temperature=0.6,
        top_p=0.95,
        min_p=0.0,
        top_k=20,
        # presence_penalty=0.0,
        repetition_penalty=1.0,

        # for testing, using maximum memory
        generation_kwargs={
            "min_new_tokens": MAX_COMPLETION_LENGTH,
        },        

        chat_template_kwargs = {
            "example_key": runtime_error("GRPO must be use apply_chat_template"),
        },

        # vllm - many cuda hardcoded code :(
        use_vllm=use_vllm,
        vllm_mode="colocate",
        vllm_max_model_length=MAX_COMPLETION_LENGTH,

        gradient_checkpointing=True,
    )

    trainer = GRPOTrainer(
        args=training_args,
        model=model,  # type: ignore
        processing_class=tokenizer,
        reward_funcs=reward_func, # type: ignore
        reward_processing_classes=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    for sample in eval_data:
        print(sample)

    resume_from_checkpoint = get_last_checkpoint(OUTPUT_DIR)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    trainer.save_model(OUTPUT_DIR)
    # tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
