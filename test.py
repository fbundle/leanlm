import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from leanlm.grpo_trainer.environment import GcdEnv, GuessEnv
from leanlm.grpo_trainer.rollout import TransformerRolloutModel, rollout_once
from leanlm.grpo_trainer.processor import qwen3_instruct_processor


rule = """
every turn, you can output a maximum number of {max_turn_tokens} tokens
the whole conversation should not last longer than {max_conversation_tokens} tokens
"""

def main():
    model_path = "Qwen/Qwen3.5-0.8B"
    processor = qwen3_instruct_processor

    model = TransformerRolloutModel(
        tokenizer=AutoTokenizer.from_pretrained(model_path),
        model=AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
        ).eval(),
    )

    max_turn_tokens = 128
    max_conversation_tokens = 2048
    system_prompt = rule.format(
        max_turn_tokens=max_turn_tokens,
        max_conversation_tokens=max_conversation_tokens,
    )

    with torch.no_grad():
        o = rollout_once(
            model=model, processor=processor,
            env=GcdEnv(), seed="36",
            system_prompt=system_prompt,
            max_turn_tokens=max_turn_tokens,
            max_conversation_tokens=max_conversation_tokens,
        )
        print(o.env_reward)

if __name__ == "__main__":
    main()