from __future__ import annotations
from dataclasses import dataclass
import io
from typing import Callable

from torch import Tensor
from jaxtyping import Float
import torch

from leanlm.env_trainer.environment import Action, Env, Seed
from leanlm.env_trainer.model import Model
from leanlm.env_trainer.processor import Processor

@dataclass
class RolloutState:
    initial_prompt_length: int
    current_prompt_ids: list[int]
    env_mask: list[int]
    logprobs: Float[Tensor, "n1 d"] | None
    total_step_reward: float

    def append_completion(
        self,
        completion_ids: list[int],
        logprobs: Float[Tensor, "n1 d"] | None,
        step_reward: float,
    ):
        if logprobs is None:
            env_mask = [0] * len(completion_ids)
            logprobs = torch.zeros(size=[len(completion_ids)])
        else:
            env_mask = [1] * len(completion_ids)

        self.current_prompt_ids.extend(completion_ids)
        self.env_mask.extend(env_mask)
        if self.logprobs is None:
            self.logprobs = logprobs
        else:
            # concat logprobs, pass to the newer logprobs device
            current_logprobs = self.logprobs.to(logprobs.device)
            self.logprobs = torch.cat([current_logprobs, logprobs], dim=0)
        self.total_step_reward += step_reward

def init_rollout_state(initial_prompt_ids: list[int]) -> RolloutState:
    return RolloutState(
        initial_prompt_length=len(initial_prompt_ids),
        current_prompt_ids=initial_prompt_ids,
        env_mask=[],
        logprobs=None,
        total_step_reward=0,
    )


def batch_rollout(
    model: Model, processor: Processor, env_factory: Callable[[], Env],
    system_prompt: str, max_turn_length: int, max_conversation_length: int,
    seed_list: list[Seed],
    log_file: io.TextIOBase | None = None, 
) -> list[RolloutState]:
    def LOG(*args):
        if log_file is not None:
            print(*args, end="\n", file=log_file, flush=True)

    LOG("system>\t" + system_prompt)
    system_text = processor.init_system_input(system_prompt)
    
    env_list: list[Env] = []
    state_list: list[RolloutState] = []
    for i, seed in enumerate(seed_list):
        env = env_factory()
        initial_delta = env.reset(seed)
        LOG(f"user_{i}>\t" + initial_delta)

        initial_delta_text = processor.append_user_input(initial_delta)
        initial_prompt_ids = model.tokenizer_encode(system_text + initial_delta_text)

        state = init_rollout_state(initial_prompt_ids=initial_prompt_ids)

        env_list.append(env)
        state_list.append(state)

    while True:
        # MODEL BATCH GENERATE
        completion_ids_list, logprobs_list = model.model_batch_generate([state.current_prompt_ids for state in state_list])

        # UPDATE STATE
        for state, completion_ids, logprobs in zip(state_list, completion_ids_list, logprobs_list):
            state.append_completion(
                completion_ids=completion_ids,
                logprobs=logprobs,
                step_reward=0.0,
            )
        
        # PARSE ACTION
        completion_text_list = [model.tokenizer_decode(completion_ids) for completion_ids in completion_ids_list]
        action_list: list[Action] = []
        for i, completion_text in enumerate(completion_text_list):
            reason, action = processor.parse_agent_output(completion_text)
            LOG(f"agent_{i}>\t" + action)
            action_list.append(action)
        
        # INTERACT WITH ENVIRONMENT
        for i, (action, env) in enumerate(zip(action_list, env_list)):










