from typing import Any, Callable

from transformers import AutoTokenizer, AutoModelForCausalLM

from peft import PeftModel

type Tokenizer = Any
type ModelForCausalLM = Any

from pydantic import BaseModel

class TextToText(BaseModel):
    get_prompt_from_input: Callable[[str], str]
    get_input_from_prompt: Callable[[str], str]
    get_output_from_completion: Callable[[str], str]

    model_path: str
    lora_checkpoint_path: str | None = None

    # attributes
    _tokenizer: Tokenizer | None = None
    @property
    def tokenizer(self) -> Tokenizer:
        if self._tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if tokenizer.padding_side is None:
                tokenizer.padding_side = "left"
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            self._tokenizer = tokenizer
        
        return self._tokenizer

    _model: ModelForCausalLM | None = None
    @property
    def model(self) -> ModelForCausalLM:
        if self._model is None:
            model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=self.model_path)
            if self.lora_checkpoint_path is not None:
                model = PeftModel.from_pretrained(model, self.lora_checkpoint_path)
            self._model = model
        
        return self._model

if __name__ == "__main__":
    t2t = TextToText(
        get_prompt_from_input=lambda input_str: f"<|im_start|>user\n{input_str}<|im_end|>\n<|im_start|>assistant\n<think>\n",
        get_input_from_prompt=lambda prompt: prompt.lstrip("<|im_start|>user\n").rstrip("<|im_end|>\n<|im_start|>assistant\n<think>\n"),
        get_output_from_completion=lambda completion: completion.split("</think>")[-1],
        model_path="Qwen/Qwen3-0.6B",
    )

    print(t2t.tokenizer)
    print(t2t.model)

    