from __future__ import annotations

import os
import sys
import time

from llm_engine.api import Message, ROLE_USER, ROLE_SYSTEM, ROLE_ASSISTANT


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
    log_path = sys.argv[1]

    url = " http://127.0.0.1:3000/v1/chat/completions"
    if len(sys.argv) >= 3:
        model_path = sys.argv[2]
    else:
        model_path = "transformer:gemma:google/gemma-4-E2B-it"

    main(url, model_path, log_path)

