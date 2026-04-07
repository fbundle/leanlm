from pydantic import BaseModel

# a subset of OpenAI chat completion API streaming mode

# request

type Role = str
ROLE_USER: Role = "user"
ROLE_SYSTEM: Role = "system"
ROLE_ASSISTANT: Role = "assistant"

class Message(BaseModel):
    role: Role = ROLE_USER
    content: str # TODO - make this include other data type like images, videos

class ChatCompletionRequest(BaseModel):
    model: str = "transformer:gemma:google/gemma-4-E2B-it"
    messages: list[Message]
    stream: bool = True

    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    min_p: float = 0.0
    max_completion_tokens: int = 4096

    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

# request model

type ChatCompletionEngine = str
TRANSFORMER_ENGINE: ChatCompletionEngine = "transformer"
MLX_ENGINE: ChatCompletionEngine = "mlx"

DEFAULT_ENGINE: ChatCompletionEngine = TRANSFORMER_ENGINE

type ChatCompletionConsumerType = str
GEMMA_CONSUMER: ChatCompletionConsumerType = "gemma"
QWEN_CONSUMER: ChatCompletionConsumerType = "qwen"

DEFAULT_TOKEN_TYPE: ChatCompletionConsumerType = GEMMA_CONSUMER


def parse_model_path(model_path: str) -> tuple[str, str, str]:
    parts = model_path.split(":")
    if len(parts) == 1:
        return DEFAULT_ENGINE, DEFAULT_TOKEN_TYPE, parts[0]
    elif len(parts) == 2:
        return DEFAULT_ENGINE, parts[0], parts[1]
    else:
        return parts[0], parts[1], parts[2]

# response

class ChatCompletionDelta(BaseModel):
    content: str
    reasoning_content: str

    def is_empty(self) -> bool:
        return self.content == "" and self.reasoning_content == ""

class ChatCompletionChoice(BaseModel):
    delta: ChatCompletionDelta
    finish_reason: str | None = None

class ChatCompletionChunk(BaseModel):
    choices: list[ChatCompletionChoice]