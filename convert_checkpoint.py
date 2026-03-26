import sys
import os
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_model(model_path: str, peft_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_path,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
    )
    model = PeftModel.from_pretrained(base_model, peft_path)
    return model, tokenizer

def ensure_dir(path: str):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    if os.path.exists(path):
        shutil.rmtree(path)
    
def save_model_hf(path: str, model, tokenizer):
    ensure_dir(path)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


if __name__ == "__main__":
    model_path, peft_path = sys.argv[1], sys.argv[2]
    peft_path = os.path.abspath(peft_path)

    model, tokenizer = load_model(model_path, peft_path)

    prefix = os.path.dirname(os.path.dirname(peft_path))
    peft_name = peft_path.lstrip(prefix).replace("/", "_")

    print(peft_name)

    # save_model_hf(f"mnt/output_hf/{model_name}", model, tokenizer)