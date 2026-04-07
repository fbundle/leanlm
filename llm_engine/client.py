from __future__ import annotations

import os
import sys
import time
from typing import Iterator

import requests
from transformers import GenerationConfig

from .api import ChatCompletionDelta, ChatCompletionRequest, ChatCompletionChunk
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


WELCOME = "type your prompt (type '# <prompt>' to set system prompt)\n"

PROMPT_PREFIX = "> "
SYSTEM_PREFIX = "# "


def main(url: str, model_path: str, log_path: str):
    c = Conversation(conversation_path=log_path)

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
                for delta in chat(url, model_path, c.message_list):
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
    log_path = sys.argv[1]

    if len(sys.argv) >= 3:
        model_path = sys.argv[2]
    else:
        model_path = "transformer:gemma:google/gemma-4-E2B-it"

    main(
        url="http://127.0.0.1:3000/v1/chat/completions",
        model_path=model_path,
        log_path=log_path,
    )
