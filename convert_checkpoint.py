from __future__ import annotations

import json
import os
import pathlib
import sys
from typing import Iterator

import mlx_lm
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


def list_files(dir: str) -> Iterator[str]:
    path = pathlib.Path(dir)
    for f in path.rglob("*"):
        if f.is_file():
            yield str(f.relative_to(path))

def copy_dir(src_dir: str, dst_dir: str, link: bool = True):
    for path in list_files(src_dir):
        src_path = os.path.join(src_dir, path)
        dst_path = os.path.join(dst_dir, path)
        os.makedirs(
            name=os.path.dirname(dst_path),
            exist_ok=True,
        )
        if link:
            os.link(src=src_path, dst=dst_path)
        else:
            raise NotImplementedError

def get_checkpoint_name(checkpoint_path: str) -> str:
    name1 = os.path.basename(checkpoint_path)
    name2 = os.path.basename(os.path.dirname(checkpoint_path))
    return f"{name2}-{name1}"

def is_lora_checkpoint(path: str) -> bool:
    return os.path.exists(os.path.join(path, "adapter_config.json"))

def patch_hf(hf_path: str):
    model_type_patch = {
        "qwen3_5_text": "qwen3_5"
    }

    with open(f"{hf_path}/config.json") as f:
        config = json.loads(f.read())
    
    with open(f"{hf_path}/config.backup.json", "w") as f:
        f.write(json.dumps(config, indent=2))

    if config["model_type"] in model_type_patch:
        config["model_type"] = model_type_patch[config["model_type"]]
    
        with open(f"{hf_path}/config.json", "w") as f:
            f.write(json.dumps(config, indent=2))

def prepare_hf_model(checkpoint_path: str, cache_dir: str = "mnt/output_hf") -> str:
    hf_path = os.path.join(cache_dir, get_checkpoint_name(checkpoint_path))

    if os.path.exists(hf_path):
        return hf_path

    if is_lora_checkpoint(checkpoint_path):
        adapter_config = json.loads(
            open(os.path.join(checkpoint_path, "adapter_config.json")).read()
        )
        base_model_path = adapter_config["base_model_name_or_path"]
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        model = model.merge_and_unload() # type: ignore
    else:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path)

    tokenizer.save_pretrained(hf_path)
    model.save_pretrained(hf_path)
    patch_hf(hf_path)
    return hf_path

def get_model_name(hf_path: str) -> str:
    return os.path.basename(hf_path)

def main(model_path: str):
    hf_path = prepare_hf_model(model_path, cache_dir="mnt/output_hf")
    model_name = get_model_name(hf_path)
    mlx_path = f"mnt/output_mlx/{model_name}"

    if not os.path.exists(mlx_path):
        mlx_lm.convert(
            hf_path=hf_path,
            mlx_path=mlx_path,
            quantize=True,
        )
    
    print("mlx model", mlx_path)

if __name__ == "__main__":
    model_path = sys.argv[1]
    main(model_path)
        