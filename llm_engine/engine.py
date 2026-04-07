from typing import Any, Iterator

from threading import Thread

import mlx_lm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextIteratorStreamer

from .api import Message


def apply_chat_template_with_thinking(tokenizer, message_list: list[Message]) -> str:
    input_text = tokenizer.apply_chat_template(
        conversation=[message.model_dump() for message in message_list],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return input_text

class Engine:
    def chat(self, message_list: list[Message], max_completion_tokens: int, **generate_kwargs: Any) -> Iterator[str]:
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
        self.model.generate(
            **input_ids,
            streamer=text_streamer,
            **generate_kwargs,
        )

    def chat(self, message_list: list[Message], max_completion_tokens: int, **addon_generate_kwargs: Any) -> Iterator[str]:
        text_streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,  # skip the prompt, stream the output only
            skip_special_tokens=False,  # pass into tokenizer.decode, skip EOS for example
        )

        generate_kwargs = {
            "max_new_tokens": max_completion_tokens,
        }

        generate_kwargs.update(addon_generate_kwargs)

        thread = Thread(
            target=TransformerEngine.generate,
            args=(self, message_list, text_streamer, generate_kwargs),
        )
        thread.start()

        def streamer() -> Iterator[str]:
            yield from text_streamer
            thread.join()

        return streamer()


class MlxEngine(Engine):
    def __init__(self, model_path: str):
        super().__init__()
        print(f"loading mlx {model_path}")
        model, tokenizer, config = mlx_lm.load( # type: ignore
            path_or_hf_repo=model_path,
            return_config=True,
        )
        self.model = model
        self.tokenizer = tokenizer

    def chat(self, message_list: list[Message], max_completion_tokens: int, **generate_kwargs: Any) -> Iterator[str]:
        input_text = apply_chat_template_with_thinking(self.tokenizer, message_list)

        sampler = mlx_lm.sample_utils.make_sampler(**self.generate_kwargs)

        response_generator = mlx_lm.stream_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=input_text,
            max_tokens=max_completion_tokens,
            sampler=sampler,
        )

        def streamer() -> Iterator[str]:
            for response in response_generator:
                yield response.text

        return streamer()

