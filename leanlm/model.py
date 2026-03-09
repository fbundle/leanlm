from __future__ import annotations

from typing import Any, Callable

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

type Tokenizer = Any
type ModelForCausalLM = Any

from pydantic import BaseModel

class TextToText(BaseModel):
    get_input_from_question: Callable[[str], str]
    get_question_from_input: Callable[[str], str]
    get_answer_from_output: Callable[[str], str]

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

    def to_device(self, device: torch.device | str = torch.device("cpu")) -> TextToText:
        self._model = self.model.to(device)
        return self

    def decode(self, questions: list[str], **generate_kwargs) -> list[str]:
        inputs = [self.get_input_from_question(question) for question in questions]

        input_ids = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
        )
        input_ids = {k: v.to(self.model.device) for k, v in input_ids.items()}

        _generate_kwargs: dict[str, Any] = {}
        _generate_kwargs.update(generate_kwargs)

        output_ids = self.model.generate(
            **input_ids,
            **_generate_kwargs,
        )
        output_ids = output_ids.cpu()

        outputs = self.tokenizer.batch_decode(
            output_ids,
            skip_prompt=True,
            # skip_special_tokens=True,
        )
        answers = [self.get_answer_from_output(output) for output in outputs]

        return answers

    def merge_and_unload(self) -> ModelForCausalLM:
        return self.model.merge_and_unload()


if __name__ == "__main__":
    t2t = TextToText(
        get_input_from_question=lambda input_str: f"<|im_start|>user\n{input_str}<|im_end|>\n<|im_start|>assistant\n<think>\n",
        get_question_from_input=lambda prompt: prompt.lstrip("<|im_start|>user\n").rstrip("<|im_end|>\n<|im_start|>assistant\n<think>\n"),
        get_answer_from_output=lambda completion: completion.split("</think>")[-1],
        model_path="Qwen/Qwen3-0.6B",
        lora_checkpoint_path="mnt/output/calculator_qwen3_0p6b_lora_v1/checkpoint-550",
    )

    t2t = t2t.to_device("mps")

    questions = [
        "52342+1123160=",
        "52342*1123160=",
    ]
    expected_answers = [
        "1175502",
        "58788440720"
    ]

    answers = t2t.decode(
        questions,
        # hard code
        max_new_tokens=32768,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        repetition_penalty=1.0,
    )
    
    for e, a in zip(expected_answers, answers):
        print(f"expected_answer:\t{e}")
        print(f"actual_answer:\t\t{a.strip()}")
        print()


    