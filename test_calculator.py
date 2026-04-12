import json
import os
from leanlm.llm_engine.api import ChatCompletionGenerateConfig
from leanlm.llm_engine.engine import MlxEngine, TransformerEngine
from peft import PeftModel

from leanlm.llm_trainer.processor import Qwen3Processor

def is_lora_checkpoint(path: str) -> bool:
    return os.path.exists(os.path.join(path, "adapter_config.json"))

def is_mlx_checkpoint(path: str) -> bool: # type: ignore
    if os.path.exists(os.path.join(path, "README.md")):
        mlx: bool = False
        for line in open(os.path.join(path, "README.md")):
            line = line.strip()
            parts = line.split(":")
            if len(parts) == 2:
                if parts[0].strip() == "library_name" and parts[1].strip() == "mlx":
                    mlx = True
                    break
        return mlx

def main():
    to_instruction = Qwen3Processor().marshal_input

    checkpoint_path = "mnt/output_mlx/qwen3.5-4b-length4096-p0.3-lora-calculator-checkpoint-4300"
    checkpoint_path = "mnt/output_mlx/qwen3.5-4b-length4096-p0.3-calculator-checkpoint-1200"
    

    if is_lora_checkpoint(checkpoint_path):
        print("LOADING LORA CHECKPOINT ...")

        adapter_config = json.loads(
            open(os.path.join(checkpoint_path, "adapter_config.json")).read()
        )
        base_model_path = adapter_config["base_model_name_or_path"]

        engine = TransformerEngine(model_path=base_model_path)
        engine.model = PeftModel.from_pretrained(engine.model, checkpoint_path)  # type: ignore
        engine.model = engine.model.to("mps") # type: ignore
    elif is_mlx_checkpoint(checkpoint_path):
        print("LOADING MLX CHECKPOINT ...")
        engine = MlxEngine(checkpoint_path)
    else:
        print("LOADING TRANSFORMER CHECKPOINT ...")
        # full finetuning
        engine = TransformerEngine(checkpoint_path)
        engine.model = engine.model.to("mps") # type: ignore


    question = "1234567890 * 6789012345"
    # answer from deepseek
    # https://chat.deepseek.com/share/t7cawkll4myikz7sq5

    question = "123 * 678"


    chat = engine.chat(messages=to_instruction(question), config=ChatCompletionGenerateConfig(
        max_completion_tokens=4096,
        temperature=0.6,
        top_p=0.95,
        min_p=0.0,
        top_k=20,
        repetition_penalty=1.1,
    ))
    print("-------------------------------------------------")
    for content in chat:
        print(content, end="", flush=True)

if __name__ == "__main__":
    main()