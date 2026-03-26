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

def load_model(model_path: str, peft_path: str | None):
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_path,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
    )
    if peft_path is None:
        model_name = os.path.basename(model_path)
        return base_model, tokenizer, model_name
    else:
        peft_path = os.path.abspath(peft_path)
        prefix = repeat(2)(os.path.dirname)(peft_path)
        model_name = peft_path.lstrip(prefix).replace("/", "_")

        model = PeftModel.from_pretrained(base_model, peft_path)
        model = model.merge_and_unload() # type: ignore
        return model, tokenizer, model_name

def ensure_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
    

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

def save_model(hf_path: str, mlx_path: str, model, tokenizer):
    ensure_dir(hf_path)

    model.save_pretrained(hf_path)
    tokenizer.save_pretrained(hf_path)
    patch_hf(hf_path)

    ensure_dir(mlx_path)
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

    model, tokenizer, model_name = load_model(model_path, peft_path)

    if model.config.model_type in model_type_patch:
        model.config.model_type = model_type_patch[model.config.model_type]

    save_model(
        hf_path=f"mnt/output_hf/{model_name}",
        mlx_path=f"mnt/output_mlx/{model_name}",
        model=model,
        tokenizer=tokenizer,
    )