import sys

from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model_and_tokenizer(model_path: str, peft_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_path,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
    )
    model = PeftModel.from_pretrained(base_model, peft_path)
    return model, tokenizer


if __name__ == "__main__":
    model_path, peft_path = sys.argv[1], sys.argv[2]
    print(model_path, peft_path)