from __future__ import annotations

import os
import sys
import time
from typing import Iterator

import requests
from transformers import GenerationConfig

from .api import ChatCompletionDelta, ChatCompletionRequest, ChatCompletionChunk, Role
from .api import Message, ROLE_USER, ROLE_SYSTEM, ROLE_ASSISTANT


def chat(
        url: str, model: str,
        message_list: list[Message], generation_config: GenerationConfig | None = None,
) -> Iterator[ChatCompletionDelta]:
    request = ChatCompletionRequest(
        model=model,
        messages=message_list,
        stream=True,
    )
    if generation_config is not None:
        request.temperature = generation_config.temperature
        request.top_p = generation_config.top_p
        request.top_k = generation_config.top_k
        request.min_p = generation_config.min_p
        request.max_completion_tokens = generation_config.max_new_tokens

    with requests.post(url=url, json=request.model_dump(), stream=True) as response:
        response.raise_for_status()
        for b in response.iter_lines():
            line = b.decode("utf-8")
            parts = line.split(":", maxsplit=1)
            if len(parts) != 2:
                continue
            key, val = parts[0].strip(), parts[1].strip()
            if key != "data":
                continue

            chunk = ChatCompletionChunk.model_validate_json(val)
            if len(chunk.choices) > 0 and not chunk.choices[0].delta.is_empty():
                yield chunk.choices[0].delta


ROLE_CONFIG_SET_MODEL: Role = "config_set_model"


class Conversation:
    path: str
    model: str
    messages: list[Message]

    def __init__(self, path: str, default_model: str):
        self.path = path
        self.model = ""
        self.messages = []

        self.load()

        if len(self.model) == 0:
            print("using default model")
            self.append(Message(
                role=ROLE_CONFIG_SET_MODEL,
                content=default_model,
            ))


    def load(self):
        if os.path.exists(self.path):
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    try:
                        m = Message.model_validate_json(line)
                        self.append(m, write=False)
                    except Exception as e:
                        print(f"ERROR: parse {line}")

    def append(self, m: Message, write: bool = True):
        if m.role in {ROLE_USER, ROLE_SYSTEM, ROLE_ASSISTANT}:
            self.messages.append(m)
        elif m.role == ROLE_CONFIG_SET_MODEL:
            self.model = m.content
        else:
            print(f"ERROR: append {m}")

        if write:
            with open(self.path, "a") as f:
                f.write(m.model_dump_json() + "\n")


WELCOME = "type your prompt (type '# <prompt>' to set system prompt)\n"

PROMPT_PREFIX = "> "
SYSTEM_PREFIX = "# "


def main(path: str):
    url = "http://127.0.0.1:3000/v1/chat/completions"
    model = "transformer:gemma:google/gemma-4-E2B-it"
    c = Conversation(
        path=path,
        default_model=f"{url}@{model}",
    )

    url, model = c.model.split("@", maxsplit=1)
    print(f"using model at {url} {model}")

    print(WELCOME)
    for message in c.messages:
        if message.role == ROLE_USER:
            print(f"{PROMPT_PREFIX}{message.content}")
        elif message.role == ROLE_SYSTEM:
            print(f"{SYSTEM_PREFIX}{message.content}")
        else:
            print(message.content)

    while True:
        if len(c.messages) > 0 and c.messages[-1].role == ROLE_USER:
            text_list = []
            t0 = time.perf_counter()
            try:
                for delta in chat(url, model, c.messages):
                    if len(delta.content) > 0:
                        text_list.append(delta.content)

                    print(delta.reasoning_content, end="", flush=True, file=sys.stderr)
                    print(delta.content, end="", flush=True)
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
                word_per_sec = len(text.split()) / (t1 - t0)
                print(f"stats: word_per_sec {word_per_sec}", file=sys.stderr)
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
    main(sys.argv[1])
