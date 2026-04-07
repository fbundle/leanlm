from threading import Thread
from typing import Iterator

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
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
    def chat(self, message_list: list[Message], generation_config: GenerationConfig) -> Iterator[str]:
        raise NotImplemented

class TransformerEngine:
    def __init__(self, model_path: str):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu")

    def generate(self, message_list: list[Message], text_streamer: TextIteratorStreamer, generation_config: GenerationConfig):
        input_text = apply_chat_template_with_thinking(self.tokenizer, message_list)

        # TODO - implement caching for tokenizer
        input_ids = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        self.model.generate(
            **input_ids,
            streamer=text_streamer,
            generation_config=generation_config,
        )

    def chat(self, message_list: list[Message], generation_config: GenerationConfig) -> Iterator[str]:
        text_streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,  # skip the prompt, stream the output only
            skip_special_tokens=False,  # pass into tokenizer.decode, skip EOS for example
        )

        thread = Thread(
            target=TransformerEngine.generate,
            args=(self, message_list, text_streamer, generation_config),
        )
        thread.start()

        def streamer() -> Iterator[str]:
            yield from text_streamer
            thread.join()

        return streamer()


class MlxEngine(Engine):
    def __init__(self, model_path: str):
        super().__init__()
        import mlx_lm

        model, tokenizer, config = mlx_lm.load( # type: ignore
            path_or_hf_repo=model_path,
            return_config=True,
        )
        self.model = model
        self.tokenizer = tokenizer

    def chat(self, message_list: list[Message], generation_config: GenerationConfig) -> Iterator[str]:
        input_text = apply_chat_template_with_thinking(self.tokenizer, message_list)
        import mlx_lm

        # translate huggingface generation config to mlx generation config
        sampler = mlx_lm.sample_utils.make_sampler(
            temp=generation_config.temperature,
            top_p=generation_config.top_p,
            top_k=generation_config.top_k,
            min_p=generation_config.min_p,
        )

        logits_processors = mlx_lm.sample_utils.make_logits_processors(
            repetition_penalty=generation_config.repetition_penalty,
        )

        max_tokens = 1024
        if generation_config is not None:
            max_tokens = generation_config.max_new_tokens

        response_generator = mlx_lm.stream_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=input_text,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
        )

        def streamer() -> Iterator[str]:
            for response in response_generator:
                yield response.text

        return streamer()

