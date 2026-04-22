from dataclasses import dataclass
from typing import Protocol


import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .processor import Processor
from .environment import Env, Seed

class RolloutModel(Protocol):
    def tokenizer_encode(self, input_text: str) -> torch.Tensor: ...
    def tokenizer_decode(self, completions_ids: torch.Tensor) -> str: ...
    def model_generate(self, prompt_ids: torch.Tensor, max_new_tokens: int, eos_token_id: list[int] | None = None) -> tuple[torch.Tensor, torch.Tensor]: ...


@dataclass
class RolloutResult:
    prompt_ids: torch.Tensor            # shape (m,)
    completion_ids: torch.Tensor        # shape (n,)
    env_mask: torch.Tensor              # shape (n,)
    logprobs: torch.Tensor              # shape (n,)
    env_reward: float                   # scalar


def rollout_once(
        model: RolloutModel, processor: Processor,
        env: Env, seed: Seed,
        system_prompt: str,
        max_turn_tokens: int, max_conversation_tokens: int,
) -> RolloutResult:
    completions_ids_list = []
    logprobs_list = []
    env_mask_list = []

    print("system>\t", system_prompt, flush=True)

    initial_state_delta = env.reset(seed=seed)
    print(f"user>\t", initial_state_delta, flush=True)

    # initial_prompt_ids is of shape (m,)
    initial_state = processor.init_system_input(system_prompt) + processor.append_user_input(initial_state_delta)
    initial_prompt_ids: torch.Tensor = model.tokenizer_encode(initial_state)

    prompt_ids: torch.Tensor = initial_prompt_ids

    last_reward = 0

    while True:
        # MODEL GENERATE
        completions_ids, logprobs = model.model_generate(prompt_ids=prompt_ids, max_new_tokens=max_turn_tokens)
        d = logprobs.shape[1] # number of tokens

        completions_ids_list.append(completions_ids)
        env_mask_list.append(torch.ones_like(completions_ids))
        logprobs_list.append(logprobs)

        # INTERACT WITH ENVIRONMENT
        completion_text = model.tokenizer_decode(completions_ids=completions_ids)

        reason, action = processor.parse_agent_output(completion_text)
        print("agent>\t", action, flush=True)
        state_delta = env.step(action)
        print("user>\t", state_delta, flush=True)

        last_reward = env.reward

        if env.terminate:
            break
    
        if len(prompt_ids) + len(completions_ids) >= max_conversation_tokens:
            break

        # assuming tokenizer is additive
        # tok(a ++ b) = tok(a) ++ tok(b)
        state_delta_ids = model.tokenizer_encode(processor.append_user_input(state_delta))
        prompt_ids = torch.cat([prompt_ids, completions_ids, state_delta_ids])

        completions_ids_list.append(state_delta_ids)
        env_mask_list.append(torch.zeros_like(state_delta_ids))
        logprobs_list.append(torch.zeros(size=[len(state_delta_ids), d]))
    
    return RolloutResult(
        prompt_ids = initial_prompt_ids,
        completion_ids = torch.cat(completions_ids_list),
        env_mask = torch.cat(env_mask_list), # Support custom env_mask from rollout_func (e.g., for environment feedback masking)
        logprobs = torch.cat(logprobs_list),
        env_reward=last_reward,
    )


# some examples below

class TransformerRolloutModel(RolloutModel):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel):
        self.tokenizer = tokenizer
        self.model = model
    
    def tokenizer_encode(self, input_text: str) -> torch.Tensor:
        i = self.tokenizer(text=input_text, return_tensors="pt").to(self.model.device)
        prompt_ids = i.input_ids.squeeze()
        return prompt_ids
    
    def tokenizer_decode(self, completions_ids: torch.Tensor) -> str:
        s = self.tokenizer.decode(completions_ids)
        if isinstance(s, list): s = s[0]
        return s
    
    def model_generate(self, prompt_ids: torch.Tensor, max_new_tokens: int, eos_token_id: list[int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        eos_token_ids = [self.tokenizer.eos_token_id]
        if eos_token_id is not None:
            eos_token_ids.extend(eos_token_id)


        input_ids = prompt_ids.unsqueeze(dim=0) # prompt_ids is of shape (m,)
        attention_mask = torch.ones_like(input_ids)
        o = self.model.generate(                    # type: ignore
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_ids,              # stop generation when receiving one of eos_token_ids
            output_logits=True,
            return_dict_in_generate=True,
        )
        # o.sequences is of shape (1, m + n)
        # o.logits is of shape (n, 1, d) where n is len(completion) and d is the number of tokens

        completions_ids = o.sequences[0, -len(o.logits) :]          # completions_ids is of shape (n,)
        logprobs = torch.cat(o.logits)                              # logprobs is of shape (n, d)

        return completions_ids, logprobs