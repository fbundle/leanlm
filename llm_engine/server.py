import sys
from typing import Iterator, Callable

from fastapi import FastAPI, HTTPException
from fastapi.sse import EventSourceResponse

from .chat_completion import ChatCompletionConsumer, GemmaChatCompletionConsumer
from .engine import Engine, TransformerEngine, MlxEngine
from .api import ChatCompletionRequest, ChatCompletionChunk, ChatCompletionChoice


def split_iter(sep: str, iter: Iterator[str]) -> Iterator[str]:
    for chunk in iter:
        parts = chunk.split(sep)
        yield parts[0]
        for part in parts[1:]:
            yield sep
            yield part

type ChatCompletionEngine = str
TRANSFORMER_ENGINE: ChatCompletionEngine = "transformer"
MLX_ENGINE: ChatCompletionEngine = "mlx"

DEFAULT_ENGINE: ChatCompletionEngine = "transformer"

type ChatCompletionConsumerType = str
GEMMA_CONSUMER: ChatCompletionConsumerType = "gemma"
QWEN_CONSUMER: ChatCompletionConsumerType = "qwen"

DEFAULT_TOKEN_TYPE: ChatCompletionConsumerType = "gemma"

def parse_model_path(model_path: str) -> tuple[str, str, str]:
    parts = model_path.split(":")
    if len(parts) == 1:
        return DEFAULT_ENGINE, DEFAULT_TOKEN_TYPE, parts[0]
    elif len(parts) == 2:
        return DEFAULT_ENGINE, parts[0], parts[1]
    else:
        return parts[0], parts[1], parts[2]



chat_completion_consumer_dict: dict[str, Callable[[], ChatCompletionConsumer]] = {
    GEMMA_CONSUMER: GemmaChatCompletionConsumer,
}

engine_dict: dict[str, Callable[[str], Engine]] = {
    TRANSFORMER_ENGINE: TransformerEngine,
    MLX_ENGINE: MlxEngine,
}


class StreamerApp:
    fastapi: FastAPI
    engine_dict: dict[str, Engine]

    def __init__(self):
        self.fastapi = FastAPI()
        self.engine_dict = {}
        self.fastapi.router.api_route(
            path="/v1/chat/completions",
            methods=["POST"],
            response_class=EventSourceResponse,
        )(self.chat_completion)
        # self.fastapi.post(
        #     path="/v1/chat/completions",
        #     response_class=EventSourceResponse,
        # )(self.chat_completion)

    def chat_completion(self, request: ChatCompletionRequest) -> Iterator[ChatCompletionChunk]:
        if not request.stream:
            raise HTTPException(status_code=400, detail="only support stream=True")

        engine_type, consumer_type, model_path = parse_model_path(request.model)

        if consumer_type not in chat_completion_consumer_dict:
            raise HTTPException(status_code=400, detail=f"consumer type {consumer_type} not supported")
        consumer = chat_completion_consumer_dict[consumer_type]()

        if request.model not in self.engine_dict:
            if engine_type not in engine_dict:
                raise HTTPException(status_code=400, detail=f"engine {engine_type} not supported")

            engine = engine_dict[engine_type](model_path)
            self.engine_dict[request.model] = engine

        # TODO - add remove model automatically after like 10 minutes

        streamer = self.engine_dict[request.model]

        # TODO - add generation kwargs
        chunk_iter = streamer.chat(
            message_list=request.messages,
            max_completion_tokens=request.max_completion_tokens,
        )

        for token in consumer.split_tokens():
            chunk_iter = split_iter(token, chunk_iter)


        for chunk in chunk_iter:
            print(chunk, end="", flush=True, file=sys.stderr)

            delta, ok = consumer.consume(chunk)
            if not ok:
                break
            if delta is not None:
                yield ChatCompletionChunk(
                    choices=[ChatCompletionChoice(
                        delta=delta,
                        finish_reason=None,
                    )],
                )

if __name__ == "__main__":

    app = StreamerApp()

    import uvicorn
    print("docs at http://127.0.0.1:3000/docs")
    uvicorn.run(app.fastapi, host="127.0.0.1", port=3000)