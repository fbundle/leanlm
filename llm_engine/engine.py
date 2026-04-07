from typing import Any, Iterator

from threading import Thread

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextIteratorStreamer

from .oai_api import Message


def apply_chat_template_with_thinking(tokenizer, message_list: list[Message]) -> str:
    input_text = tokenizer.apply_chat_template(
        conversation=[message.model_dump() for message in message_list],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return input_text

class Engine:
    def chat(self, message_list: list[Message], **generate_kwargs: Any) -> Iterator[str]:
        raise NotImplemented

class TransformerEngine:
    def __init__(self, model_path: str):
        super().__init__()
        print(f"loading transformer {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu")

    def generate(self, message_list: list[Message], text_streamer: TextIteratorStreamer, generate_kwargs: dict[str, Any]):
        input_text = apply_chat_template_with_thinking(self.tokenizer, message_list)
        input_ids = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        self.model.generate( # type: ignore
            **input_ids,
            streamer=text_streamer,
            **generate_kwargs,
        )

    def chat(self, message_list: list[Message], **generate_kwargs: Any) -> Iterator[str]:
        text_streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,  # skip the prompt, stream the output only
            skip_special_tokens=False,  # pass into tokenizer.decode, skip EOS for example
        )

        thread = Thread(
            target=TransformerEngine.generate,
            args=(self, message_list, text_streamer, generate_kwargs),
        )
        thread.start()

        def streamer() -> Iterator[str]:
            yield from text_streamer
            thread.join()

        return streamer()
