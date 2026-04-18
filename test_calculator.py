import os
import sys
from typing import Iterator

from leanlm.llm_engine.api import ChatCompletionGenerateConfig
from leanlm.llm_engine.engine import MlxEngine, TransformerEngine
from leanlm.arithmetic.arithmetic import get_expected_output

from leanlm.llm_trainer.processor import Qwen3Processor


def is_mlx_checkpoint(path: str) -> bool: # type: ignore
    if os.path.exists(os.path.join(path, "README.md")):
        mlx: bool = False
        for line in open(os.path.join(path, "README.md")):
            line = line.strip()
            parts = line.split(":")
            if len(parts) == 2:
                if parts[0].strip() == "library_name" and parts[1].strip() == "mlx":
                    mlx = True
                    break
        return mlx

def split_token(i: Iterator[str], sep: str) -> Iterator[str]:
    for text in i:
        parts = text.split(sep)
        yield parts[0]
        for part in parts[1:]:
            yield sep
            yield part

def main(checkpoint_path: str):
    processor = Qwen3Processor()

    if is_mlx_checkpoint(checkpoint_path):
        print("LOADING MLX CHECKPOINT ...")
        engine = MlxEngine(checkpoint_path)
    else:
        print("LOADING TRANSFORMER CHECKPOINT ...")
        # full finetuning
        engine = TransformerEngine(checkpoint_path)
        engine.model = engine.model.to("mps") # type: ignore


    # question = "1234567890 * 6789012345"
    # answer from deepseek
    # https://chat.deepseek.com/share/t7cawkll4myikz7sq5

    while True:
        question = input("question: ") # "12345 * 67890"

        chat = engine.chat(messages=processor.marshal_input(question), config=ChatCompletionGenerateConfig(
            max_completion_tokens=131072,
            temperature=0.6,
            top_p=0.95,
            min_p=0.0,
            top_k=20,
            repetition_penalty=1.1,
        ))
        print("-------------------------------------------------", file=sys.stderr)
        outputs: list[str] = []
        for content in chat:
            print(content, end="", flush=True, file=sys.stderr)
            outputs.append(content)
        print()
        print("-------------------------------------------------", file=sys.stderr)
        
        
        expect = get_expected_output(question)
        _, actual = processor.unmarshal_output("".join(outputs))
        actual = actual.strip()

        print("question: ", question)
        print("expect:   ", expect)
        print("actual:   ", actual)

if __name__ == "__main__":
    main(sys.argv[1])
