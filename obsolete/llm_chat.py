import os
import sys
import time
from threading import Thread
from typing import *

import pydantic
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, Mistral3ForConditionalGeneration

ROLE_USER: str = "user"
ROLE_SYSTEM: str = "system"
ROLE_ASSISTANT: str = "assistant"
ROLE_COMMAND_EXPORT: str = "export"


class Message(pydantic.BaseModel):
    role: str
    content: str


class Model:
    tokenizer: Any
    model: Any
    def chat(self, message_list: list[Message]) -> Iterator[str]:
        raise NotImplemented


class TransformersModel(Model):
    def __init__(
            self,
            tokenizer: Any, model: Any,
            generate_kwargs: dict | None = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model.eval()

        self.generate_kwargs = {}
        if generate_kwargs is not None:
            self.generate_kwargs.update(generate_kwargs)

    def _generate(self, message_list: list[Message], text_streamer: TextIteratorStreamer):
        input_text = self.tokenizer.apply_chat_template(
            conversation=[message.model_dump() for message in message_list],
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device).to(self.model.dtype)
        self.model.generate(
            **model_inputs,
            streamer=text_streamer,
            **self.generate_kwargs,
        )

    def chat(self, message_list: list[Message]) -> Iterator[str]:
        text_streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,  # skip the prompt, stream the output only
            skip_special_tokens=True,  # pass into tokenizer.decode, skip EOS for example
        )
        thread = Thread(target=TransformersModel._generate, args=(self, message_list, text_streamer))
        thread.start()

        def streamer() -> Iterator[str]:
            yield from text_streamer
            thread.join()

        return streamer()


def load_conversation(conversation_path: str) -> list[Message]:
    message_list = []
    for line in open(conversation_path):
        line = line.strip()
        if len(line) == 0:
            continue
        message = Message.model_validate_json(line)
        message_list.append(message)
    return message_list


def export_conversation(conversation_path: str, message_list: list[Message], overwrite: bool = False):
    parent_path = os.path.dirname(conversation_path)
    if len(parent_path) > 0:
        os.makedirs(parent_path, exist_ok=True)
    mode = "w" if overwrite else "a"

    with open(conversation_path, mode=mode) as f:
        for message in message_list:
            f.write(message.model_dump_json() + "\n")


class Conversation:
    def __init__(self, init_message_list: list[Message] | None = None, conversation_path: str = ""):
        if init_message_list is None:
            init_message_list = []

        self.conversation_path: str = conversation_path
        self.message_list: list[Message] = init_message_list

        if len(self.conversation_path) > 0 and os.path.exists(self.conversation_path):
            self.message_list.extend(load_conversation(self.conversation_path))

    def extend(self, *message_list: Message):
        self.message_list.extend(message_list)
        if len(self.conversation_path) > 0:  # export new messages
            export_conversation(self.conversation_path, list(message_list), overwrite=False)


def parse_message(text: str) -> Message | None:
    """
    prompt should be something like
    (system prompt) - /system you are an AI agent
    (user prompt)   - hello, please help me; what is an R module in math
    """
    text = text.strip()
    if len(text) == 0:
        return None

    prefix = text.split(maxsplit=1)[0].lower()

    if prefix == "/export":
        return Message(
            role=ROLE_COMMAND_EXPORT,
            content=text.lstrip("/export").strip(),
        )
    if prefix == "/system":
        return Message(
            role=ROLE_SYSTEM,
            content=text.lstrip("/system")
        )
    # otherwise
    return Message(
        role=ROLE_USER,
        content=text,
    )


ModelConstructor = Callable[[Optional[str], Optional[str]], Model]


def get_model_factory() -> dict[str, ModelConstructor]:
    def openai_gpt_oss_20b(device_name: Optional[str], cache_dir: Optional[str]):
        path = "openai/gpt-oss-20b"
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=path,
            device_map={"": device_name},
            cache_dir=cache_dir,
        )
        return TransformersModel(
            tokenizer=tokenizer,
            model=model,
            generate_kwargs={
                "max_new_tokens": 131072,
                "min_new_tokens": 200,      # encourage substance, not just 1-liner
                "temperature": 0.7,         # balanced creativity
                "top_p": 0.9,               # nucleus sampling
                "repetition_penalty": 1.1,  # avoid loops
                "do_sample": True,          # needed with temperature/top_p
                "eos_token_id": tokenizer.eos_token_id,
            },
        )

    def deepseekr1_distill_qwen1p5b(device_name: Optional[str], cache_dir: Optional[str]):
        path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=path,
            device_map={"": device_name},
            cache_dir=cache_dir,
        )
        return TransformersModel(
            tokenizer=tokenizer,
            model=model,
            generate_kwargs={
                "max_new_tokens": 131072,
                "temperature": 0.6,
                "top_p": 0.95,
            },
        )

    def deepseekr1_distill_qwen32b(device_name: Optional[str], cache_dir: Optional[str]):
        path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=path,
            device_map={"": device_name},
            cache_dir=cache_dir,
        )
        return TransformersModel(
            tokenizer=tokenizer,
            model=model,
            generate_kwargs={
                "max_new_tokens": 131072,
                "temperature": 0.6,
                "top_p": 0.95,
            },
        )


    def qwen3_30b_a3b(device_name: Optional[str], cache_dir: Optional[str]):
        path = "Qwen/Qwen3-30B-A3B"
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=path,
            device_map={"": device_name},
            cache_dir=cache_dir,
        )
        return TransformersModel(
            tokenizer=tokenizer,
            model=model,
            generate_kwargs={
                "max_new_tokens": 32768,
            },
        )


    def qwen3_30b_a3b_instruct_2507(device_name: Optional[str], cache_dir: Optional[str]):
        path = "Qwen/Qwen3-30B-A3B-Instruct-2507"
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=path,
            device_map={"": device_name},
            cache_dir=cache_dir,
        )
        return TransformersModel(
            tokenizer=tokenizer,
            model=model,
            generate_kwargs={
                "max_new_tokens": 262144,
            },
        )

    def qwen3_30b_a3b_thinking_2507(device_name: Optional[str], cache_dir: Optional[str]):
        path = "Qwen/Qwen3-30B-A3B-Thinking-2507"
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=path,
            device_map={"": device_name},
            cache_dir=cache_dir,
        )
        return TransformersModel(
            tokenizer=tokenizer,
            model=model,
            generate_kwargs={
                "max_new_tokens": 262144,
            },
        )
    def mistral_small_3_1_24b_instruct_2503(device_name: Optional[str], cache_dir: Optional[str]):
        path = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = Mistral3ForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=path,
            device_map={"": device_name},
            cache_dir=cache_dir,
        )
        return TransformersModel(
            tokenizer=tokenizer,
            model=model,
            generate_kwargs={
                "max_new_tokens": 131072,
            },
        )

    return locals()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        prog="llm_chat.py",
        epilog="""llm_chat.py - a simple chatbot that uses LLMs to generate responses to user prompts.
        
        /system <system_prompt> - set the system prompt
        /export <path> - export the conversation to a file
        <other_prompt> - send a message to the LLM
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--list_model", help="list all models", default=False, action='store_true')
    parser.add_argument("--model", type=str, help="model name", default="deepseekr1_distill_qwen1p5b")
    parser.add_argument("--device", type=str, help="device", default="cpu")
    parser.add_argument("--conversation", type=str, help="conversation.json", default="")
    parser.add_argument("--cache", type=str, help="cache dir", default="")
    args = parser.parse_args()

    def print_some(*args, **kwargs):
        print(*args, **kwargs, end="", flush=True)

    model_factory: dict[str, ModelConstructor] = get_model_factory()

    if args.list_model:
        print("list of models:")
        for model_name in model_factory.keys():
            print(f"\t{model_name}")
        exit(0)

    model_name = args.model
    conversation_path = args.conversation
    device_name = args.device
    cache_dir = args.cache if len(args.cache) > 0 else None

    model = model_factory[model_name](device_name, cache_dir)
    conversation = Conversation(conversation_path=conversation_path)

    while True:
        message = parse_message(input("prompt: "))
        if message is not None:
            if message.role == ROLE_COMMAND_EXPORT:
                path = message.content
                export_conversation(path, conversation.message_list, overwrite=True)
                print(f"saved conversation to {path}")
                continue

            conversation.extend(message)
        if conversation.message_list[-1].role == ROLE_SYSTEM:
            continue

        print_some(f"assistant: ")
        response = ""
        token_count = 0
        t1 = time.perf_counter()
        try:
            for text in model.chat(conversation.message_list):
                print_some(text)

                response += text
                token_count += 1

        except KeyboardInterrupt:
            ...
        finally:
            if token_count > 0:
                conversation.extend(Message(
                    role=ROLE_ASSISTANT,
                    content=response,
                ))

            dur = time.perf_counter() - t1
            print(
                f"\ntotal_time: {int(round(dur))}s num_tokens: {token_count} token_per_sec: {token_count / dur}",
                file=sys.stderr,
            )
        print_some()


if __name__ == "__main__":
    main()