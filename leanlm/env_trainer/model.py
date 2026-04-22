

from typing import Any, Protocol
from jaxtyping import Int, Float

import torch
from torch import Tensor
from torch.functional import F  # type: ignore
from transformers import PreTrainedModel, PreTrainedTokenizerBase

def collapse_eos_token(text: str, eos_token: str) -> str:
    while text.endswith(eos_token + eos_token):
        text = text[:-len(eos_token)]
    return text

def collapse_eos_token_id(completion_ids: Int[Tensor, "n"], eos_token_id: int) ->  Int[Tensor, "n1"]:
    while len(completion_ids) >= 2 and completion_ids[-1] == eos_token_id and completion_ids[-2] == eos_token_id:
        completion_ids = completion_ids[:-1]
    return completion_ids


class Model:
    def __init__(
            self, tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel,
            generation_kwargs: dict[str, Any] | None = None,
        ):
        self.tokenizer = tokenizer
        self.model = model
        self.generation_kwargs = {
            "max_new_tokens": 256,
            "eos_token_id": [self.tokenizer.eos_token_id],
        }
        if generation_kwargs is not None:
            self.generation_kwargs.update(generation_kwargs)
        
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # pop hard coded keys
        for key in ["input_ids", "attention_mask", "output_logits", "return_dict_in_generate"]:
            if key in self.generation_kwargs:
                raise RuntimeError(f"generation_kwargs[{key}] must not be set")
        
    
    def tokenizer_encode(self, input_text: str) -> Int[Tensor, "m"]:
        i = self.tokenizer(
            text=input_text,
            return_tensors="pt",
        )
        input_ids = i.input_ids.sequeeze()
        return input_ids

    def tokenizer_decode(self, completions_ids: Int[Tensor, "n"]) -> str:
        output_text = self.tokenizer.decode(completions_ids)
        assert isinstance(output_text, str)
        return output_text


    def model_generate(
        self, input_ids: Int[Tensor, "b m"], attention_mask: Int[Tensor, "b m"],
    ) -> tuple[Int[Tensor, "b n"], Float[Tensor, "b n d"]]:
        b1, m1 = input_ids.shape
        o = self.model.generate(                                        # type: ignore
            input_ids=input_ids.to(self.model.device),
            attention_mask=attention_mask.to(self.model.device),
            output_logits=True,
            return_dict_in_generate=True,
            **self.generation_kwargs,
        )
        logits: Float[Tensor, "b n d"] = torch.stack(o.logits, dim=1)
        logprobs: Float[Tensor, "b n d"] = F.log_softmax(logits, dim=-1)
        b2, n2, d2 = logits.shape
        b3, mn3 = o.sequences.shape
        assert b1 == b2 and b2 == b3 and m1 + n2 == mn3

        completions_ids: Int[Tensor, "b n"] = o.sequences[:, -n2:]
        return completions_ids, logprobs

    
if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM
    path = "Qwen/Qwen3.5-0.8B"
    m = Model(
        tokenizer=AutoTokenizer.from_pretrained(path),
        model=AutoModelForCausalLM.from_pretrained(path).to("mps"),
        generation_kwargs=dict(
            max_new_tokens=512,
        ),
    )

    input_text = [
        "hello, this is an example",
        "water is blue"
    ]

    input_text: list[str] = [m.tokenizer.apply_chat_template( # type: ignore
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    ) for text in input_text]

    print(input_text)

    input_ids, attention_mask = m.tokenizer_encode(input_text)
    completions_ids, logprobs = m.model_generate(input_ids, attention_mask)
    output_text = m.tokenizer_decode(completions_ids)
    print(output_text)
    
