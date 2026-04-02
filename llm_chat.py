from __future__ import annotations


import os
import sys
from threading import Thread
import time
from typing import Any, Callable, Iterator

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextIteratorStreamer

import mlx_lm
import mlx_lm.sample_utils


from pydantic import BaseModel

MAX_NEW_TOKENS = 262144

ROLE_USER = "user"
ROLE_SYSTEM = "system"
ROLE_ASSISTANT = "assistant"

class Message(BaseModel):
    role: str
    content: str

class Conversation:
    conversation_path: str
    message_list: list[Message]

    def __init__(self, conversation_path: str):
        self.conversation_path = conversation_path
        self.message_list = []
        if os.path.exists(conversation_path):
            with open(conversation_path) as f:
                for line in f:
                    message = Message.model_validate_json(line)
                    self.message_list.append(message)
    
    def append(self, message: Message) -> Conversation:
        self.message_list.append(message)
        with open(self.conversation_path, "a") as f:
            f.write(message.model_dump_json() + "\n")
        return self

    def dumps(self) -> list[dict[str, str]]:
        return [message.model_dump() for message in self.message_list]

def enable_thinking(prompt: str) -> str:
    prompt = prompt.rstrip()
    prompt = prompt.rstrip("</think>")
    prompt = prompt.rstrip()
    prompt = prompt + "\n\n"

    if "</think>" in prompt:
        raise RuntimeError(f"enable_thinking: {prompt}")

    return prompt

def apply_chat_template_with_thinking(tokenizer, message_list: list[Message]) -> str:
    input_text = tokenizer.apply_chat_template(
        conversation=[message.model_dump() for message in message_list],
        tokenize=False,
        add_generation_prompt=True,
    )
    return enable_thinking(input_text)

class Streamer:
    def chat(self, message_list: list[Message]) -> Iterator[str]:
        raise NotImplemented

class TransformerStreamer(Streamer):
    def __init__(self, model_path: str, generate_kwargs: dict[str, Any] | None = None):
        super().__init__()
        print(f"loading transformer {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
        self.generate_kwargs = {}
        if generate_kwargs is not None:
            self.generate_kwargs.update(generate_kwargs)
    
    def _generate(self, message_list: list[Message], text_streamer: TextIteratorStreamer):
        input_text = apply_chat_template_with_thinking(self.tokenizer, message_list)
        input_ids = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        self.model.generate(
            **input_ids,
            streamer=text_streamer,
            max_new_tokens=MAX_NEW_TOKENS,
            **self.generate_kwargs,
        )
    
    def chat(self, message_list: list[Message]) -> Iterator[str]:
        text_streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,  # skip the prompt, stream the output only
            skip_special_tokens=True,  # pass into tokenizer.decode, skip EOS for example
        )

        thread = Thread(target=TransformerStreamer._generate, args=(self, message_list, text_streamer))
        thread.start()

        def streamer() -> Iterator[str]:
            yield from text_streamer
            thread.join()

        return streamer()

class MlxStreamer(Streamer):
    def __init__(self, model_path: str, generate_kwargs: dict[str, Any] | None = None):
        super().__init__()
        print(f"loading mlx {model_path}")
        model, tokenizer, config = mlx_lm.load( # type: ignore
            path_or_hf_repo=model_path,
            return_config=True,
        )
        self.model = model
        self.tokenizer = tokenizer
        self.generate_kwargs = {}
        if generate_kwargs is not None:
            self.generate_kwargs.update(generate_kwargs)
    
    def chat(self, message_list: list[Message]) -> Iterator[str]:
        input_text = apply_chat_template_with_thinking(self.tokenizer, message_list)

        sampler = mlx_lm.sample_utils.make_sampler(**self.generate_kwargs)

        response_generator = mlx_lm.stream_generate(
            model=self.model, tokenizer=self.tokenizer, prompt=input_text,
            max_tokens=MAX_NEW_TOKENS, sampler=sampler,
        )

        def streamer() -> Iterator[str]:
            for response in response_generator:
                yield response.text

        return streamer()


class Kwargs:
    def __init__(self, **kwargs: Any):
        setattr(self, "kwargs", kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
    def to_dict(self) -> dict[str, Any]:
        return getattr(self, "kwargs")


model_factory: dict[str, Callable[[str], Streamer]] = {}

def generate_model_factory():
    global model_factory
    for model_name in ["Qwen3.5-0.8B"]:
        model_path = f"Qwen/{model_name}"
        model_factory[model_path] = lambda model_path: TransformerStreamer(
            model_path=model_path,
            generate_kwargs=Kwargs(
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                min_p=0.0,
                # presence_penalty=0.0,
                repetition_penalty=1.1,
            ).to_dict(),
        )

    for model_name in ["Qwen3.5-0.8B"]:
        model_path = f"mnt/output_mlx/{model_name}"
        model_factory[model_path] = lambda model_path: MlxStreamer(
            model_path=model_path,
            generate_kwargs=Kwargs(
                temp=0.6,
                top_p=0.95,
                top_k=20,
                min_p=0.0,
            ).to_dict(),
        )
    
    for model_name in ["mnt/output_mlx/qwen3.5-0.8b-lora-calculator_checkpoint-300"]:
        model_path = "mnt/output_mlx/qwen3.5-0.8b-lora-calculator_checkpoint-300"
        model_factory[model_path] = lambda model_path: MlxStreamer(
            model_path=model_path,
            generate_kwargs=Kwargs(
                temp=0.6,
                top_p=0.95,
                top_k=20,
                min_p=0.0,
            ).to_dict(),
        )

generate_model_factory()



WELCOME = "type your prompt (type '# <prompt>' to set system prompt)\n"

PROMPT_PREFIX = "> "
SYSTEM_PREFIX = "# "
CONVERSATION_PATH = ".chat.jsonl"

def main(streamer: Streamer):
    c = Conversation(conversation_path=CONVERSATION_PATH)

    print(WELCOME)

    for message in c.message_list:
        if message.role == ROLE_USER:
            print(f"{PROMPT_PREFIX}{message.content}")
        elif message.role == ROLE_SYSTEM:
            print(f"{SYSTEM_PREFIX}{message.content}")
        else:
            print(message.content)

    while True:
        if len(c.message_list) > 0 and c.message_list[-1].role == ROLE_USER:
            text_list = []
            t0 = time.perf_counter()
            try:
                for text in streamer.chat(c.message_list):
                    text_list.append(text)
                    print(text, end="", flush=True)
                print()
            except Exception as e:
                print("ERROR: ", e)
            finally:
                text = "".join(text_list)
                c.append(Message(
                    role=ROLE_ASSISTANT,
                    content=text,
                ))
                t1 = time.perf_counter()
                word_per_sec = len(text.split()) / (t1-t0)
                print(f"stats: word_per_sec {word_per_sec}")
        else:
            input_text = input(PROMPT_PREFIX)
            if input_text.startswith(SYSTEM_PREFIX):
                system_prompt = input_text.lstrip(SYSTEM_PREFIX)
                c.append(Message(
                    role=ROLE_SYSTEM,
                    content=system_prompt,
                ))
            else:
                user_prompt = input_text
                c.append(Message(
                    role=ROLE_USER,
                    content=user_prompt,
                ))


if __name__ == "__main__":
    model_path = ""
    if len(sys.argv) >= 2:
        model_path = sys.argv[1]
    streamer = model_factory.get(model_path, None)
    if streamer is None:
        print("model not found")
        print("models available")
        for key in model_factory:
            print(f"\t{key}")
    else:
        main(streamer(model_path))
