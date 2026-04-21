import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from leanlm.grpo_trainer.environment import GcdEnv, GuessEnv
from leanlm.grpo_trainer.processor import Gemma4InstructProcessor, Qwen3InstructProcessor, Qwen3Processor
from leanlm.grpo_trainer.rollout import rollout_once


"""
model_path = "google/gemma-4-E2B-it"
from transformers import  AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.apply_chat_template(
    conversation=[{"role": "user", "content": "hello"}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
"""

def main():
    model_path = "Qwen/Qwen3.5-0.8B"
    processor = Qwen3InstructProcessor()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    with torch.no_grad():
        o = rollout_once(
            tokenizer, model, processor,
            env=GuessEnv(), seed="36",
            max_turn_tokens=64,
            max_conversation_tokens=1024,
        )
        print(o.env_reward)





if __name__ == "__main__":
    main()