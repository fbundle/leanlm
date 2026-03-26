from __future__ import annotations


import os
import sys
from threading import Thread
import time
from typing import Any, Iterator

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextIteratorStreamer

import mlx_lm
import mlx_lm.sample_utils


from pydantic import BaseModel



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
    def chat(self, message_list: list[Message], addon_kwargs: dict[str, Any] | None = None) -> Iterator[str]:
        raise NotImplemented

class TransformerStreamer(Streamer):
    def __init__(self, model_path: str):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    
    def _generate(self, message_list: list[Message], generate_kwargs: dict[str, Any], text_streamer: TextIteratorStreamer):
        input_text = apply_chat_template_with_thinking(self.tokenizer, message_list)
        input_ids = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        self.model.generate(
            **input_ids,
            steamer=text_streamer,
            max_new_tokens=-1,
            **generate_kwargs,
        )
    
    def chat(self, message_list: list[Message], addon_kwargs: dict[str, Any] | None = None) -> Iterator[str]:
        text_streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,  # skip the prompt, stream the output only
            skip_special_tokens=True,  # pass into tokenizer.decode, skip EOS for example
        )
        generate_kwargs: dict[str, Any] = {}
        if addon_kwargs is not None:
            generate_kwargs.update(addon_kwargs)

        thread = Thread(target=TransformerStreamer._generate, args=(self, message_list, generate_kwargs, text_streamer))
        thread.start()

        def streamer() -> Iterator[str]:
            yield from text_streamer
            thread.join()

        return streamer()

class MlxStreamer(Streamer):
    def __init__(self, model_path: str):
        super().__init__()
        model, tokenizer, config = mlx_lm.load( # type: ignore
            path_or_hf_repo=model_path,
            return_config=True,
        )
        self.model = model
        self.tokenizer = tokenizer
    
    def chat(self, message_list: list[Message], addon_kwargs: dict[str, Any] | None = None) -> Iterator[str]:
        input_text = apply_chat_template_with_thinking(self.tokenizer, message_list)
        generate_kwargs: dict[str, Any] = {}
        if addon_kwargs is not None:
            generate_kwargs.update(addon_kwargs)

        sampler = mlx_lm.sample_utils.make_sampler(**generate_kwargs)

        response_generator = mlx_lm.stream_generate(
            model=self.model, tokenizer=self.tokenizer, prompt=input_text,
            max_tokens=-1, sampler=sampler,
        )

        def streamer() -> Iterator[str]:
            for response in response_generator:
                yield response.text

        return streamer()


class Kwargs(BaseModel):
    dict: dict[str, Any]
    def __init__(self, **kwargs: Any):
        super().__init__(dict=kwargs)

class Model(BaseModel):
    model_type: str
    model_path: str
    generate_kwargs: Kwargs


models: dict[str, Model] = {
    "mnt/output_mlx/Qwen3.5-0.8B": Model(
        model_type="mlx",
        model_path="mnt/output_mlx/Qwen3.5-0.8B",
        generate_kwargs=Kwargs(
            temp=0.6,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
        )
    ),
    "Qwen/Qwen3.5-0.8B": Model(
        model_type="transformer",
        model_path="Qwen/Qwen3.5-0.8B",
        generate_kwargs=Kwargs(
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            # presence_penalty=0.0,
            repetition_penalty=1.1,
        )
    )
}



WELCOME = "type your prompt (type ':q' to quit) (type ':s <prompt>' to set system prompt)"
LOOP_PROMPT = ">>>"
CONVERSATION_PATH = ".chat.jsonl"

def main(model_path: str):
    model, tokenizer, config = mlx_lm.load( # type: ignore
        path_or_hf_repo=model_path,
        return_config=True,
    )
    sampler = mlx_lm.sample_utils.make_sampler(
        temp=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
    )

    c = Conversation(conversation_path=CONVERSATION_PATH)

    print(WELCOME)
    while True:
        if len(c.message_list) > 0 and c.message_list[-1].role == ROLE_USER:
            prompt = tokenizer.apply_chat_template(
                c.dumps(),
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt = enable_thinking(prompt)
            text_list = []
            t0 = time.perf_counter()
            try:
                for response in mlx_lm.stream_generate(
                    model=model, tokenizer=tokenizer, prompt=prompt,
                    max_tokens=-1,
                    sampler=sampler,
                ):
                    text_list.append(response.text)
                    print(response.text, end="", flush=True)
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
            input_text = input(LOOP_PROMPT)
            if input_text.startswith(":q"):
                break
            elif input_text.startswith(":s"):
                system_prompt = input_text.lstrip(":s")
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
    main(sys.argv[1])
