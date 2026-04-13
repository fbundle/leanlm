

import os
import sys

from leanlm.llm_engine.api import ChatCompletionGenerateConfig, ChatCompletionRequest
from leanlm.llm_engine.client import main as client_main


if __name__ == "__main__":
    path = None
    if len(sys.argv) >= 2:
        path = sys.argv[1]

    token: str | None = os.environ.get("OPENAPI_TOKEN", None)

    client_main(
        path=path,
        url="http://100.77.152.9:1234/v1/chat/completions",
        req=ChatCompletionRequest(
            # model="gguf:qwen:mnt/output_gguf/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf",
            model="gguf:gemma:gemma-4-e4b-it",
            messages=[],
            stream=True,
            generate_config=ChatCompletionGenerateConfig(
                max_completion_tokens=4096,
                temperature=1.0,
                top_p=0.95,
                min_p=0.0,
                top_k=64,
                repetition_penalty=1.1,
            ),
        ),
        token=token,
    )
