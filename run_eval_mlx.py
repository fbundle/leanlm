from __future__ import annotations


import os
import sys
import time


import mlx_lm
import mlx_lm.sample_utils
from pydantic import BaseModel

def enable_thinking(prompt: str) -> str:
    prompt = prompt.rstrip()
    prompt = prompt.rstrip("</think>")
    prompt = prompt.rstrip()
    prompt = prompt + "\n\n"

    if "</think>" in prompt:
        raise RuntimeError(f"enable_thinking: {prompt}")

    return prompt

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

WELCOME = "type your prompt (type ':q' to quit) (type ':s <prompt>' to set system prompt)"
LOOP_PROMPT = ">>>"
CONVERSATION_PATH = "conversation.jsonl"

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

    c = Conversation(conversation_path="conversation.json")

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
                    max_tokens=32768,
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
            if input_text.startswith(":s"):
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
