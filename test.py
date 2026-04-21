import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from leanlm.grpo_trainer.environment import GcdEnv
from leanlm.grpo_trainer.processor import Qwen3Processor
from leanlm.grpo_trainer.rollout import rollout_once


def main():
    model_path = "Qwen/Qwen3.5-0.8B"
    processor = Qwen3Processor()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    with torch.no_grad():
        o = rollout_once(
            tokenizer, model, processor,
            env=GcdEnv(), seed="18 30",
            max_turn_tokens=64,
            max_tokens=1024,
        )
        print(o)





if __name__ == "__main__":
    main()