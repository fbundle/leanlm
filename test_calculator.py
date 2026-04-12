from leanlm.llm_engine.api import ChatCompletionGenerateConfig
from leanlm.llm_engine.engine import TransformerEngine
from peft import PeftModel
from transformers.trainer_utils import get_last_checkpoint

def main():

    # engine = MlxEngine("mnt/output_mlx/qwen3.5-4b-length4096-calculator-checkpoint-1000")
    # engine = TransformerEngine("mnt/output/qwen3.5-4b-length4096-calculator/checkpoint-1000")
    engine = TransformerEngine("Qwen/Qwen3.5-4B")
    engine.model = PeftModel.from_pretrained(engine.model, "mnt/output/qwen3.5-4b-length4096-lora-calculator/checkpoint-200")
    engine.model = engine.model.to("mps")

    to_instruction = lambda input_text: "<|im_start|>user\n" + input_text + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
    # to_instruction = lambda input_str: f"<｜begin▁of▁sentence｜><｜User｜>{input_str}<｜Assistant｜><think>\n"

    chat = engine.chat(messages=to_instruction("1234567890 + 6789012345"), config=ChatCompletionGenerateConfig(
        max_completion_tokens=131072,
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