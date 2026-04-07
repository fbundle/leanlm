from typing import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.sse import EventSourceResponse

from llm_engine.oai_api import ChatCompletionRequest, ChatCompletionChunk, ChatCompletionDelta, ChatCompletionChoice
from llm_engine.streamer import Streamer, TransformerStreamer

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
            self.streamer_dict[request.model] = TransformerStreamer(request.model)

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
                        delta = ChatCompletionDelta(content=chunk, reasoning_content="")
                    else:
                        delta = ChatCompletionDelta(content="", reasoning_content=chunk)

                    yield ChatCompletionChunk(
                        choices=[ChatCompletionChoice(
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