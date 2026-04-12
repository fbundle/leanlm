from __future__ import annotations

import json
import pathlib
from typing import Callable, Any, Iterator
import sys
import os
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

import mlx_lm

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
    
    with open(f"{hf_path}/config.backup.json", "w") as f:
        f.write(json.dumps(config, indent=2))

    if config["model_type"] in model_type_patch:
        config["model_type"] = model_type_patch[config["model_type"]]
    
        with open(f"{hf_path}/config.json", "w") as f:
            f.write(json.dumps(config, indent=2))

def restore_hf(hf_path: str):
    with open(f"{hf_path}/config.backup.json") as f:
        config = json.loads(f.read())
    
    with open(f"{hf_path}/config.json", "w") as f:
        f.write(json.dumps(config, indent=2))

def is_local(path: str) -> bool:
    return os.path.exists(path)

def is_lora_checkpoint(path: str) -> bool:
    return os.path.exists(os.path.join(path, "adapter_config.json"))

def prepare_hf_model(model_path: str) -> tuple[str, str, bool]:









    if peft_path is None:
        patched = False
        if os.path.exists(model_path):
            patch_hf(model_path)
            patched = True
        model_name = os.path.basename(os.path.dirname(model_path)) + "-" + os.path.basename(model_path)
        hf_path = model_path
        return model_name, hf_path, patched
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
    
        return model_name, hf_path, True


def main(model_path: str):
    model_name, hf_path, patched = prepare_hf_model(model_path)
    mlx_path = f"mnt/output_mlx/{model_name}"

    if not os.path.exists(mlx_path):
        mlx_lm.convert(
            hf_path=hf_path,
            mlx_path=mlx_path,
            quantize=False,
        )
    
    if patched:
        restore_hf(hf_path)

if __name__ == "__main__":
    model_path = sys.argv[1]
    main(model_path)
        