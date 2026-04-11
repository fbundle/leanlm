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
            messages: list[Message] | str,
            config: ChatCompletionGenerateConfig,
    ) -> Iterator[str]:
        raise NotImplementedError


class TransformerEngine:
    def __init__(self, model_path: str):
        super().__init__()

        print(f"loading transformer {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)

    def generate(self, input_text: str, text_streamer: TextIteratorStreamer,
                 generation_config: GenerationConfig):
        # TODO - implement caching for tokenizer
        input_ids = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        self.model.generate( # type: ignore
            **input_ids,
            streamer=text_streamer,
            generation_config=generation_config,
        ) 

    def chat(
            self,
            messages: list[Message] | str,
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
        if isinstance(messages, str):
            input_text = messages
        else:
            input_text = apply_chat_template_with_thinking(self.tokenizer, messages)

        thread = Thread(
            target=TransformerEngine.generate,
            args=(self, input_text, text_streamer, generation_config),
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

        print(f"loading mlx {model_path}")
        model, tokenizer, config = mlx_lm.load(  # type: ignore
            path_or_hf_repo=model_path,
            return_config=True,
        )
        self.model = model
        self.tokenizer = tokenizer

    def chat(
            self,
            messages: list[Message] | str,
            config: ChatCompletionGenerateConfig,
    ) -> Iterator[str]:
        if isinstance(messages, str):
            input_text = messages
        else:
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

        print(f"loading gguf {model_path}")
        self.llm = Llama(model_path=model_path)

    def chat(
            self,
            messages: list[Message] | str,
            config: ChatCompletionGenerateConfig,
    ) -> Iterator[str]:
        if isinstance(messages, str):
            raise RuntimeError("does not support message type str")

        chunk_iter = self.llm.create_chat_completion(
            messages=[{"role": m.role, "content": m.content} for m in messages], # type: ignore

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
                yield chunk["choices"][0]["delta"]["content"] # type: ignore
            except Exception as e:
                pass



if __name__ == "__main__":
    from peft import PeftModel
    from huggingface_hub import snapshot_download
    from transformers.trainer_utils import get_last_checkpoint

    engine = TransformerEngine("Qwen/Qwen3.5-4B")
    checkpoint = get_last_checkpoint("mnt/output/qwen3.5-4b-lora-calculator")
    engine.model = PeftModel.from_pretrained(engine.model, checkpoint) # type: ignore

    engine.model  = engine.model.to("mps")
    to_instruction = lambda input_text: "<|im_start|>user\n" + input_text + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
    # to_instruction = lambda input_str: f"<｜begin▁of▁sentence｜><｜User｜>{input_str}<｜Assistant｜><think>\n"

    chat = engine.chat(messages=to_instruction("12345*67890="), config=ChatCompletionGenerateConfig())
    print("-------------------------------------------------")
    for content in chat:
        print(content, end="", flush=True)
