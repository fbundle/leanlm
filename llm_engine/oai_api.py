from pydantic import BaseModel

# a subset of OpenAI chat completion API streaming mode

# request

class Message(BaseModel):
    role: str = "user"
    content: str

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

# response

class ChatCompletionDelta(BaseModel):
    content: str
    reasoning_content: str

class ChatCompletionChoice(BaseModel):
    delta: ChatCompletionDelta
    finish_reason: str | None = None

class ChatCompletionChunk(BaseModel):
    choices: list[ChatCompletionChoice]