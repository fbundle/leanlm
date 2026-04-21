from pydantic import BaseModel, ConfigDict
import torch

from .processor import Processor
from .environment import Env, Seed

def tokenizer_encode(tokenizer, model, input_text: str) -> torch.Tensor:
    i = tokenizer(text=input_text, return_tensors="pt").to(model.device)
    prompt_ids = i.input_ids.squeeze()
    return prompt_ids

def tokenizer_decode(tokenizer, model, completions_ids: torch.Tensor) -> str:
    return tokenizer.decode(completions_ids)

def model_generate(tokenizer, model, prompt_ids: torch.Tensor, max_new_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = prompt_ids.unsqueeze(dim=0) # prompt_ids is of shape (m,)
    attention_mask = torch.ones_like(input_ids)
    o = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        eos_token_id=[tokenizer.eos_token_id],              # stop generation when receiving eos_token_id <|im_end|>
        output_logits=True,
        return_dict_in_generate=True,
    )
    # o.sequences is of shape (1, m + n)
    # o.logits is of shape (n, 1, d) where n is len(completion) and d is the number of tokens

    completions_ids = o.sequences[0, -len(o.logits) :]          # completions_ids is of shape (n,)
    logprobs = torch.cat(o.logits)                              # logprobs is of shape (n, d)

    return completions_ids, logprobs



class RolloutResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    prompt_ids: torch.Tensor            # shape (m,)
    completion_ids: torch.Tensor        # shape (n,)
    env_mask: torch.Tensor              # shape (n,)
    logprobs: torch.Tensor              # shape (n,)
    env_reward: float                   # scalar

def rollout_once(
        tokenizer, model, processor: Processor,
        env: Env, seed: Seed,
        max_turn_tokens: int,
        max_tokens: int,
) -> RolloutResult:
    completions_ids_list = []
    logprobs_list = []
    env_mask_list = []

    initial_state_delta = env.reset(seed=seed)
    print(f"user>\t", initial_state_delta, flush=True)

    # initial_prompt_ids is of shape (m,)
    initial_prompt_ids: torch.Tensor = tokenizer_encode(
        tokenizer=tokenizer, model=model,
        input_text=processor.concat_input(prompt=initial_state_delta),
    )
    prompt_ids: torch.Tensor = initial_prompt_ids

    last_reward = 0

    while True:
        # MODEL GENERATE
        completions_ids, logprobs = model_generate(tokenizer, model, prompt_ids=prompt_ids, max_new_tokens=max_turn_tokens)
        d = logprobs.shape[1] # number of tokens

        completions_ids_list.append(completions_ids)
        env_mask_list.append(torch.ones_like(completions_ids))
        logprobs_list.append(logprobs)

        # INTERACT WITH ENVIRONMENT
        completion_text = tokenizer_decode(tokenizer, model, completions_ids=completions_ids)
        print("agent>\t", completion_text, flush=True)

        reason, action = processor.parse_output(completion_text)
        result = env.step(action)
        print("user>\t", result.state_delta, flush=True)

        last_reward = result.reward

        if result.terminate:
            break
    
        if len(prompt_ids) + len(completions_ids) >= max_tokens:
            break


        # assuming tokenizer is additive
        # tok(a ++ b) = tok(a) ++ tok(b)
        state_delta_ids = tokenizer_encode(
            tokenizer=tokenizer, model=model,
            input_text=processor.concat_input(result.state_delta),
        )
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
