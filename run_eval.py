from typing import Iterator
from threading import Thread
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextIteratorStreamer

from peft import PeftModel

OUTPUT_DIR = "mnt/output/calculator_qwen3_0p6b_lora"
MODEL_PATH = "Qwen/Qwen3-0.6B"

MAX_COMPLETION_LENGTH = 32768
MAX_COMPLETION_LENGTH = 4096


if len(sys.argv) >= 2:
    CHECKPOINT_PATH = sys.argv[1]
else:
    CHECKPOINT_PATH = None


WELCOME = "type your prompt (type 'exit' to quit)"
LOOP_PROMPT = ">>>"

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.padding_side is None:
        tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_model():
    base_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=MODEL_PATH,
        device_map="auto",
        # attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
    )
    if CHECKPOINT_PATH is None:
        return base_model

    print(f"loading {CHECKPOINT_PATH} ...")
    model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
    return model

class StreamerModel:
    def __init__(self, tokenizer, model):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model.eval()

    def generate(self, input_text: str, text_streamer: TextIteratorStreamer):
        input_ids = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        self.model.generate(
            **input_ids,
            streamer=text_streamer,

            # generation config recommented by qwen3.5
            # max_new_tokens=262144, # default context length for qwen3.5
            max_new_tokens=MAX_COMPLETION_LENGTH,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            # presence_penalty=0.0,
            repetition_penalty=1.0,
        )

    def chat(self, input_text: str) -> Iterator[str]:
        text_streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,  # skip the prompt, stream the output only
            skip_special_tokens=True,  # pass into tokenizer.decode, skip EOS for example
        )
        thread = Thread(target=StreamerModel.generate, args=(self, input_text, text_streamer))
        thread.start()

        def streamer() -> Iterator[str]:
            yield from text_streamer
            thread.join()

        return streamer()

def print_some(*args, **kwargs):
    print(*args, **kwargs, end="", flush=True)

def get_prompt_from_input_str(input_str: str) -> str:
    return f"<|im_start|>user\n{input_str}<|im_end|>\n<|im_start|>assistant\n<think>\n" # qwen3.5 4b
    return tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": input_str}],
        tokenize=False,
        add_generation_prompt=True,
    )

def get_input_str_from_prompt(prompt: str) -> str:
    return prompt.lstrip("<|im_start|>user\n").rstrip("<|im_end|>\n<|im_start|>assistant\n<think>\n") # qwen3.5  4b


def main():
    tokenizer = load_tokenizer()
    model = load_model()
    streamer_model = StreamerModel(tokenizer=tokenizer, model=model)

    print(WELCOME)

    while True:
        input_text = input(LOOP_PROMPT)
        if input_text.lower() == "quit":
            break
        with torch.no_grad():
            for text in streamer_model.chat(get_prompt_from_input_str(input_text)):
                print_some(text)
        print()

if __name__ == "__main__":
    main()