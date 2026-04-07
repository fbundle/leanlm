from threading import Thread
from typing import Iterator

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers import TextIteratorStreamer

from .api import Message, ChatCompletionGenerateConfig


def apply_chat_template_with_thinking(tokenizer, messages: list[Message]) -> str:
    input_text = tokenizer.apply_chat_template(
        conversation=[message.model_dump() for message in messages],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return input_text


class Engine:
    def chat(
            self,
            messages: list[Message],
            config: ChatCompletionGenerateConfig,
    ) -> Iterator[str]:
        raise NotImplementedError


class TransformerEngine:
    def __init__(self, model_path: str):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map={
            "": "mps",
        })

    def generate(self, messages: list[Message], text_streamer: TextIteratorStreamer,
                 generation_config: GenerationConfig):
        input_text = apply_chat_template_with_thinking(self.tokenizer, messages)

        # TODO - implement caching for tokenizer
        input_ids = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        self.model.generate(
            **input_ids,
            streamer=text_streamer,
            generation_config=generation_config,
        )

    def chat(
            self,
            messages: list[Message],
            config: ChatCompletionGenerateConfig,
    ) -> Iterator[str]:
        text_streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,  # skip the prompt, stream the output only
            skip_special_tokens=False,  # pass into tokenizer.decode, skip EOS, for example
        )

        generation_config = GenerationConfig(
            max_new_tokens=config.max_completion_tokens,

            temperature=config.temperature,
            top_p=config.top_p,
            min_p=config.min_p,
            top_k=config.top_k,

            repetition_penalty=config.repetition_penalty,
        )

        thread = Thread(
            target=TransformerEngine.generate,
            args=(self, messages, text_streamer, generation_config),
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

        model, tokenizer, config = mlx_lm.load(  # type: ignore
            path_or_hf_repo=model_path,
            return_config=True,
        )
        self.model = model
        self.tokenizer = tokenizer

    def chat(
            self,
            messages: list[Message],
            config: ChatCompletionGenerateConfig,
    ) -> Iterator[str]:
        input_text = apply_chat_template_with_thinking(self.tokenizer, messages)
        import mlx_lm.sample_utils

        response_generator = mlx_lm.stream_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=input_text,
            max_tokens=config.max_completion_tokens,
            sampler=mlx_lm.sample_utils.make_sampler(
                temp=config.temperature,
                top_p=config.top_p,
                min_p=config.min_p,
                top_k=config.top_k,
            ),
            logits_processors=mlx_lm.sample_utils.make_logits_processors(
                presence_penalty=config.presence_penalty,
                frequency_penalty=config.frequency_penalty,
                repetition_penalty=config.repetition_penalty,
            ),
        )

        def streamer() -> Iterator[str]:
            for response in response_generator:
                yield response.text

        return streamer()


class GgufEngine(Engine):
    def __init__(self, model_path: str):
        super().__init__()
        from llama_cpp import Llama

        self.llm = Llama(model_path=model_path)

    def chat(
            self,
            messages: list[Message],
            config: ChatCompletionGenerateConfig,
    ) -> Iterator[str]:

        chunk_iter = self.llm.create_chat_completion(
            messages=[{"role": m.role, "content": m.content} for m in messages],

            stream=True,

            max_tokens=config.max_completion_tokens,

            temperature=config.temperature,
            top_p=config.top_p,
            min_p=config.min_p,
            top_k=config.top_k,

            presence_penalty=config.presence_penalty,
            frequency_penalty=config.frequency_penalty,
            repeat_penalty=config.repetition_penalty,
        )

        for chunk in chunk_iter:
            try:
                yield chunk["choices"][0]["delta"]["content"]
            except Exception as e:
                pass



if __name__ == "__main__":
    model = GgufEngine("mnt/output_gguf/Gemma-4-E2B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf")

    chat = model.chat(messages=[
        Message(
            role="user",
            content="hello",
        ),
    ], config=ChatCompletionGenerateConfig())

    for content in chat:
        print("$$$", content)
