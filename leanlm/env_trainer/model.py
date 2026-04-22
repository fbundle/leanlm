

from typing import Any, Protocol
from jaxtyping import Int, Float

import torch
from torch import Tensor
from torch.functional import F  # type: ignore
from transformers import BatchEncoding, PreTrainedModel, PreTrainedTokenizerBase

def collapse_eos_token_id(completion_ids: Int[Tensor, "n"], eos_token_id: int) ->  Int[Tensor, "n1"]:
    indices: Int[Tensor, "k"] = (completion_ids == eos_token_id).nonzero(as_tuple=True)[0]
    if len(indices) == 0: # no eos_token found
        return completion_ids
    # eos_token found
    # [tok, tok, eos, eos, eos] -> [tok, tok, eos]
    index: int = int(indices[0]) + 1
    return completion_ids[:index]

class Model:
    def __init__(
            self, tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel,
            generation_kwargs: dict[str, Any] | None = None,
        ):
        self.tokenizer = tokenizer
        self.model = model
        self.eos_token_id = int(self.tokenizer.eos_token_id) # type: ignore
        self.generation_kwargs = {
            "max_new_tokens": 256,
            "eos_token_id": [self.eos_token_id],
        }
        if generation_kwargs is not None:
            self.generation_kwargs.update(generation_kwargs)
        
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token_id = self.eos_token_id

        # pop hard coded keys
        for key in ["input_ids", "attention_mask", "output_logits", "return_dict_in_generate"]:
            if key in self.generation_kwargs:
                raise RuntimeError(f"generation_kwargs[{key}] must not be set")
        
    
    def tokenizer_encode(self, input_text: str) -> list[int]:
        return self.tokenizer(input_text).input_ids

    def tokenizer_decode(self, completions_ids: Int[Tensor, "n"]) -> str:
        output_text = self.tokenizer.decode(completions_ids)
        assert isinstance(output_text, str)
        return output_text


    def model_batch_generate(self, input_ids_list: list[list[int]]) -> list[tuple[list[int], Float[Tensor, "n d"]]]:
        e: BatchEncoding = self.tokenizer.pad(
            {"input_ids": input_ids_list},
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        o = self.model.generate( # type: ignore
            input_ids=e.input_ids,
            attention_mask=e.attention_mask,
            output_logits=True,
            return_dict_in_generate=True,
            **self.generation_kwargs,
        )
        logits_batch: Float[Tensor, "b n d"] = torch.stack(o.logits, dim=1)
        logprobs_batch: Float[Tensor, "b n d"] = F.log_softmax(logits_batch, dim=-1)
        b, n, d = logits_batch.shape
        completions_ids_batch: Int[Tensor, "b n"] = o.sequences[:, -n:]

        completions_ids_batch = completions_ids_batch.detach().cpu()

        outputs: list[tuple[list[int], Float[Tensor, "n d"]]] = []
        for i in range(b):
            completions_ids: list[int] = collapse_eos_token_id(completions_ids_batch[i, :], self.eos_token_id).tolist()
            logprobs = logprobs_batch[i, :len(completions_ids), :]
            outputs.append((completions_ids, logprobs))
        
        return outputs
    
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
    
