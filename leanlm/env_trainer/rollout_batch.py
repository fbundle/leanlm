from dataclasses import dataclass
from typing import Any, Protocol


import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .processor import Processor
from .environment import Env, Seed
from .rollout import RolloutResult

class BatchRolloutModel(Protocol):
    def tokenizer_encode(self, input_texts: list[str]) -> list[torch.Tensor]: ...
    def tokenizer_decode(self, completions_ids: list[torch.Tensor]) -> list[str]: ...
    def model_generate(self, prompt_ids: list[torch.Tensor], max_new_tokens: int, eos_token_id: list[int] | None = None) -> tuple[list[torch.Tensor], list[torch.Tensor]]: ...


def rollout_batch(
        model: BatchRolloutModel, processor: Processor,
        envs: list[Env], seeds: list[Seed],
        system_prompt: str,
        max_turn_length: int, max_conversation_length: int,
        debug: bool = False,
) -> list[RolloutResult]:
    batch_size = len(envs)
    
    completions_ids_lists = [[] for _ in range(batch_size)]
    logprobs_lists = [[] for _ in range(batch_size)]
    env_mask_lists = [[] for _ in range(batch_size)]
    
    initial_states = []
    for i in range(batch_size):
        initial_state_delta = envs[i].reset(seed=seeds[i])
        initial_state = processor.init_system_input(system_prompt) + processor.append_user_input(initial_state_delta)
        initial_states.append(initial_state)
        if debug:
            print(f"env {i} system>\t", system_prompt, flush=True)
            print(f"env {i} user>\t", initial_state_delta, flush=True)

    initial_prompt_ids_list = model.tokenizer_encode(initial_states)
    current_prompt_ids_list = [p.detach().clone() for p in initial_prompt_ids_list]
    
    active_indices = list(range(batch_size))
    last_rewards = [0.0] * batch_size

    while active_indices:
        active_prompts = [current_prompt_ids_list[i] for i in active_indices]
        
        batch_completions_ids, batch_logprobs = model.model_generate(
            prompt_ids=active_prompts, 
            max_new_tokens=max_turn_length
        )
        
        new_active_indices = []
        
        # Collect next turn interactions
        next_turn_states = []
        next_turn_indices = []
        
        for idx_in_batch, i in enumerate(active_indices):
            completions_ids = batch_completions_ids[idx_in_batch]
            logprobs = batch_logprobs[idx_in_batch]
            d = logprobs.shape[1]

            completions_ids_lists[i].append(completions_ids)
            env_mask_lists[i].append(torch.ones_like(completions_ids))
            logprobs_lists[i].append(logprobs)

            completion_text = model.tokenizer_decode([completions_ids])[0]
            reason, action = processor.parse_agent_output(completion_text)
            
            if debug:
                print(f"env {i} agent>\t", action, flush=True)
            
            state_delta = envs[i].step(action)
            last_rewards[i] = envs[i].reward
            
            if debug:
                print(f"env {i} user>\t", state_delta, flush=True)

            if envs[i].terminate or len(current_prompt_ids_list[i]) + len(completions_ids) >= max_conversation_length:
                continue
            
            next_turn_states.append(processor.append_user_input(state_delta))
            next_turn_indices.append(i)

        if next_turn_states:
            next_turn_ids_list = model.tokenizer_encode(next_turn_states)
            for idx, i in enumerate(next_turn_indices):
                state_delta_ids = next_turn_ids_list[idx]
                
                # Update prompt for next turn
                # Find the completion_ids for this env from this turn
                idx_in_batch = active_indices.index(i)
                comp_ids = batch_completions_ids[idx_in_batch]
                
                current_prompt_ids_list[i] = torch.cat([current_prompt_ids_list[i], comp_ids, state_delta_ids])

                completions_ids_lists[i].append(state_delta_ids)
                env_mask_lists[i].append(torch.zeros_like(state_delta_ids))
                # logprobs for environment feedback are zero
                ref_logprobs = logprobs_lists[i][-1]
                d = ref_logprobs.shape[1]
                logprobs_lists[i].append(torch.zeros(size=[len(state_delta_ids), d], device=ref_logprobs.device))
                
                new_active_indices.append(i)
        
        active_indices = new_active_indices

    return [
        RolloutResult(
            prompt_ids = initial_prompt_ids_list[i],
            completion_ids = torch.cat(completions_ids_lists[i]),
            env_mask = torch.cat(env_mask_lists[i]),
            logprobs = torch.cat(logprobs_lists[i]),
            env_reward = last_rewards[i],
        )
        for i in range(batch_size)
    ]


class TransformerBatchRolloutModel(BatchRolloutModel):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel, generation_kwargs: dict[str, Any] | None = None):
        self.tokenizer = tokenizer
        self.model = model
        self.generation_kwargs = generation_kwargs or {}
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def tokenizer_encode(self, input_texts: list[str]) -> list[torch.Tensor]:
        return [self.tokenizer.encode(t, return_tensors="pt").squeeze(0).to(self.model.device) for t in input_texts]

    def tokenizer_decode(self, completions_ids: list[torch.Tensor]) -> list[str]:
        return [self.tokenizer.decode(c) for c in completions_ids]

    def model_generate(self, prompt_ids: list[torch.Tensor], max_new_tokens: int, eos_token_id: list[int] | None = None) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if not prompt_ids:
            return [], []
            
        eos_token_ids = [self.tokenizer.eos_token_id]
        if eos_token_id is not None:
            eos_token_ids.extend(eos_token_id)

        max_len = max(len(p) for p in prompt_ids)
        input_ids = torch.full((len(prompt_ids), max_len), self.tokenizer.pad_token_id, dtype=torch.long, device=self.model.device)
        attention_mask = torch.zeros((len(prompt_ids), max_len), dtype=torch.long, device=self.model.device)
        
        for i, p in enumerate(prompt_ids):
            l = len(p)
            input_ids[i, max_len - l:] = p
            attention_mask[i, max_len - l:] = 1
            
        o = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_ids,
            output_logits=True,
            return_dict_in_generate=True,
            **self.generation_kwargs,
        )
        
        logits_tensor = torch.stack(o.logits, dim=1) # (batch_size, n_gen, vocab_size)
        n_gen = logits_tensor.shape[1]
        
        batch_completions = []
        batch_logprobs = []
        
        for i in range(len(prompt_ids)):
            comp_ids = o.sequences[i, max_len : max_len + n_gen]
            lprobs = logits_tensor[i]
            
            # Trim to first EOS
            # We check for any of the eos_token_ids
            is_eos = torch.zeros_like(comp_ids, dtype=torch.bool)
            for eid in eos_token_ids:
                is_eos |= (comp_ids == eid)
            
            eos_pos = is_eos.nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                first_eos = eos_pos[0].item()
                comp_ids = comp_ids[:first_eos + 1]
                lprobs = lprobs[:first_eos + 1]
            
            batch_completions.append(comp_ids)
            batch_logprobs.append(lprobs)
            
        return batch_completions, batch_logprobs
