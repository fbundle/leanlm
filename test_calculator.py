import os
from leanlm.llm_engine.api import ChatCompletionGenerateConfig
from leanlm.llm_engine.engine import TransformerEngine
from peft import PeftModel

def is_lora_checkpoint(path: str) -> bool:
    return os.path.exists(os.path.join(path, "adapter_config.json"))

def main():

    model_path = "Qwen/Qwen3.5-4B"
    checkpoint_path = "mnt/output/qwen3.5-4b-length4096-p0.3-calculator/checkpoint-300"
    checkpoint_path = "mnt/output/qwen3.5-4b-length4096-lora-calculator/checkpoint-2800"

    if is_lora_checkpoint(checkpoint_path):
        engine = TransformerEngine(model_path=model_path)
        engine.model = PeftModel.from_pretrained(engine.model, checkpoint_path)  # type: ignore
    else:
        # full finetuning
        engine = TransformerEngine(checkpoint_path)

    engine.model = engine.model.to("mps") # type: ignore


    to_instruction = lambda input_text: "<|im_start|>user\n" + input_text + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
    # to_instruction = lambda input_str: f"<｜begin▁of▁sentence｜><｜User｜>{input_str}<｜Assistant｜><think>\n"

    question = "1234567890 * 6789012345"
    # answer from deepseek
    # https://chat.deepseek.com/share/t7cawkll4myikz7sq5


    chat = engine.chat(messages=to_instruction(question), config=ChatCompletionGenerateConfig(
        max_completion_tokens=4096,
        temperature=0.6,
        top_p=0.95,
        min_p=0.0,
        top_k=20,
        repetition_penalty=1.0,
    ))
    print("-------------------------------------------------")
    for content in chat:
        print(content, end="", flush=True)

if __name__ == "__main__":
    main()