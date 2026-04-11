from __future__ import annotations

import json
from typing import Callable, Any
import sys
import os
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

import mlx_lm

type Map = Callable[[Any], Any]

def repeat(n: int) -> Callable[[Map], Map]:
    def helper1(m: Map) -> Map:
        def helper2(x: Any) -> Any:
            for _ in range(n):
                x = m(x)
            return x
        return helper2
    return helper1


model_type_patch = {
    "qwen3_5_text": "qwen3_5"
}

def patch_hf(hf_path: str):
    with open(f"{hf_path}/config.json") as f:
        config = json.loads(f.read())
    if config["model_type"] in model_type_patch:
        config["model_type"] = model_type_patch[config["model_type"]]
    with open(f"{hf_path}/config.json", "w") as f:
        f.write(json.dumps(config, indent=2))

def prepare_hf_model(model_path: str, peft_path: str | None) -> tuple[str, str]:
    if peft_path is None:
        model_name = os.path.basename(os.path.dirname(model_path)) + "-" + os.path.basename(model_path)
        if os.path.exists(model_path):
            patch_hf(model_path)
        return model_name, model_path
    else:
        peft_path = os.path.abspath(sys.argv[2]).rstrip("/")
        prefix = repeat(2)(os.path.dirname)(peft_path)
        model_name = peft_path.lstrip(prefix).replace("/", "_")
        hf_path = f"mnt/output_hf/{model_name}"

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        base_model = AutoModelForCausalLM.from_pretrained(model_path)
        model = PeftModel.from_pretrained(base_model, peft_path)
        model = model.merge_and_unload() # type: ignore

        model.save_pretrained(hf_path)
        tokenizer.save_pretrained(hf_path)
        patch_hf(hf_path)
    
        return model_name, hf_path


def main(model_path: str, peft_path: str | None):
    model_name, hf_path = prepare_hf_model(model_path, peft_path)
    mlx_path = f"mnt/output_mlx/{model_name}"

    if not os.path.exists(mlx_path):
        mlx_lm.convert(
            hf_path=hf_path,
            mlx_path=mlx_path,
            quantize=True,
        )

if __name__ == "__main__":
    model_path = sys.argv[1]
    peft_path = None
    if len(sys.argv) >= 3:
        peft_path = sys.argv[2]
    main(model_path, peft_path)
        