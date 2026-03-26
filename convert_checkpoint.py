import sys
import os
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_model(model_path: str, peft_path: str | None):
    peft_path = os.path.abspath(peft_path)

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
        prefix = os.path.dirname(os.path.dirname(peft_path))
        model_name = peft_path.lstrip(prefix).replace("/", "_")

        model = PeftModel.from_pretrained(base_model, peft_path)
        model = model.merge_and_unload() # type: ignore
        return model, tokenizer, model_name

def ensure_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
    
def save_model_hf(path: str, model, tokenizer):
    ensure_dir(path)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


if __name__ == "__main__":
    model_path = sys.argv[1]
    peft_path = None
    if len(sys.argv) >= 3:
        peft_path = sys.argv[2]

    model, tokenizer, model_name = load_model(model_path, peft_path)

    save_model_hf(f"mnt/output_hf/{model_name}", model, tokenizer)