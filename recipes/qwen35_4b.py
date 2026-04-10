import jiwer
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM

from arithmetic.arithmetic import generate_input, get_expected_output
from llm_trainer.trainer import Processor, Language, TrainConfig, train


class Qwen3Processor(Processor):
    def __init__(self):
        super().__init__()

    def marshal_input(self, input_text: str) -> Language:
        return "<|im_start|>user\n" + input_text + "<|im_end|>\n<|im_start|>assistant\n<think>\n"

    def unmarshal_input(self, prompt: Language) -> str:
        return prompt.lstrip("<|im_start|>user\n").rstrip("<|im_end|>\n<|im_start|>assistant\n<think>\n")

    def unmarshal_output(self, completion: Language) -> str:
        completion = completion.split("</think>")[-1]
        completion = completion.split("<|im_end|>")[0]
        return completion


def load_model_and_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path)
    if tokenizer.padding_side is None:
        tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        # attn_implementation="flash_attention_2",
        dtype=torch.float16,
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

    return model, tokenizer

def reward_func(question: str, answer: str) -> float:
   expected = get_expected_output(question)
   return - jiwer.cer(expected, answer)

def main():
    model, tokenizer = load_model_and_tokenizer("Qwen/Qwen3.5-4B")
    batch_size = 4
    train_size = 100000 * batch_size
    eval_size = 50 * batch_size
    train_data = (generate_input() for _ in range(train_size))
    eval_data = [generate_input() for _ in range(eval_size)]

    config = TrainConfig(
        mode="debug",

        output_dir="mnt/output/qwen3.5-4b-lora-calculator",
        processor=Qwen3Processor(),
        tokenizer=tokenizer,
        model=model,
        reward_func=reward_func,

        batch_size=batch_size,
        accumulation_steps=8,
        num_generations=8,

        max_completion_length=4096,
        temperature=0.6,
        top_p=0.95,
        min_p=0.0,
        top_k=20,
        repetition_penalty=1.0,

        save_steps=100,
        train_data=train_data,
        eval_data=eval_data,
    )

    train(config)

if __name__ == "__main__":
    main()