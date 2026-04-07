from typing import Any, Iterator

from threading import Thread

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextIteratorStreamer

from pydantic import BaseModel


# MODEL
class Message(BaseModel):
    role: str = "user"
    content: str

def apply_chat_template_with_thinking(tokenizer, message_list: list[Message]) -> str:
    input_text = tokenizer.apply_chat_template(
        conversation=[message.model_dump() for message in message_list],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return input_text

class Streamer:
    def __init__(self, model_path: str):
        super().__init__()
        print(f"loading transformer {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu")

    def _generate(self, message_list: list[Message], text_streamer: TextIteratorStreamer, generate_kwargs: dict[str, Any]):
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
            target=Streamer._generate,
            args=(self, message_list, text_streamer, generate_kwargs),
        )
        thread.start()

        def streamer() -> Iterator[str]:
            yield from text_streamer
            thread.join()

        return streamer()

# API

from fastapi import FastAPI, HTTPException
from fastapi.sse import EventSourceResponse

class ChatCompletionRequest(BaseModel):
    model: str = "google/gemma-4-E2B-it"
    messages: list[Message]
    stream: bool = True

    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    max_completion_tokens: int = 4096
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

class Delta(BaseModel):
    content: str
    reasoning_content: str

class Choice(BaseModel):
    delta: Delta
    finish_reason: str | None = None

class ChatCompletionChunk(BaseModel):
    choices: list[Choice]


type ChatCompletionMode = str
MODE_REASON: ChatCompletionMode = "reason"
MODE_BODY: ChatCompletionMode = "body"
MODE_STOP: ChatCompletionMode = "stop"

BEG_REASON: str = "<|channel>"
END_REASON: str = "<channel|>"
END_BODY: str = "<turn|>"

def split_iter(sep: str, iter: Iterator[str]) -> Iterator[str]:
    for chunk in iter:
        parts = chunk.split(sep)
        yield parts[0]
        for part in parts[1:]:
            yield sep
            yield part

class StreamerAPI:
    streamer_dict: dict[str, Streamer] = {}
    def chat_completion(self, request: ChatCompletionRequest) -> Iterator[ChatCompletionChunk]:
        if not request.stream:
            raise HTTPException(status_code=400, detail="only support stream=True")

        # TODO - add MLX model
        if request.model not in self.streamer_dict:
            self.streamer_dict[request.model] = Streamer(request.model)

        # TODO - add remove model automatically after like 10 minutes

        streamer = self.streamer_dict[request.model]

        # TODO - add generation kwargs
        chunk_iter = streamer.chat(
            message_list=request.messages,
            max_new_tokens=request.max_completion_tokens,
        )

        # BEG_REASON, END_REASON, END_BODY are single tokens
        # so, they won't be split into multiple chunks
        chunk_iter = split_iter(BEG_REASON, chunk_iter)
        chunk_iter = split_iter(END_REASON, chunk_iter)
        chunk_iter = split_iter(END_BODY, chunk_iter)

        mode: ChatCompletionMode = MODE_BODY
        for chunk in chunk_iter:
            print(chunk, end="", flush=True)
            # check exit condition
            if mode == MODE_STOP:
                return

            # change mode if necessary
            if chunk == BEG_REASON:
                mode = MODE_REASON
            elif chunk == END_REASON:
                mode = MODE_BODY
            elif chunk == END_BODY:
                mode = MODE_STOP
            else:
                # if not change mode, then just write
                if len(chunk) > 0:
                    if mode == MODE_BODY:
                        delta = Delta(content=chunk, reasoning_content="")
                    else:
                        delta = Delta(content="", reasoning_content=chunk)

                    yield ChatCompletionChunk(
                        choices=[Choice(
                            delta=delta,
                            finish_reason=None,
                        )],
                    )

if __name__ == "__main__":
    app = FastAPI()
    api = StreamerAPI()

    # app.post("/v1/chat/completions", response_class=EventSourceResponse)(api.chat_completion)
    app.router.api_route(
        path="/v1/chat/completions",
        methods=["POST"],
        response_class=EventSourceResponse,
    )(api.chat_completion)

    import uvicorn
    print("docs at http://127.0.0.1:3000/docs")
    uvicorn.run(app, host="127.0.0.1", port=3000)