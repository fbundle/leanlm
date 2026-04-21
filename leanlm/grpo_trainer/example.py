

from typing import Callable, Literal, Protocol

from peft import LoraConfig, get_peft_model
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

type Action = str
type StateDelta = str
type Seed = str

class StepResult(BaseModel):
    state_delta: StateDelta
    reward: float
    terminate: bool

class Env(Protocol):
    def reset(self, seed: Seed) -> StateDelta:
        raise NotImplementedError
    def step(self, action: Action) -> StepResult:
        raise NotImplementedError

import re

def get_last_integer(text):
    """
    Finds and returns the last sequence of digits in a string.
    Returns None if no digits are present.
    """
    # Pattern: \d+ (digits) that are NOT followed by any other digits (?!.*\d)
    pattern = r'(\d+)(?!.*\d)'
    
    match = re.search(pattern, text)
    
    return int(match.group(0)) if match else None

class GuessEnv(Env):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def reset(self, seed: Seed) -> StateDelta:
        self.target = int(seed)
        self.reward = 0
        return "I have a number between 0 and 100 in mind, guess that number, just output the number at every turn"
    
    def step(self, action: Action) -> StepResult:
        # use regex to get the last integer
        guess = get_last_integer(action)
        
        if guess is None:
            return StepResult(
                state_delta=f"can't find the number in your input",
                reward=self.reward,
                terminate=False,
            )
        

        f = lambda x: 1 / (1 + x) # map [0, inf) -> [1, 0)
        points = f(abs(self.target - guess))
        if guess < self.target:
            state_delta, terminate = f"{guess} is too low", False
        elif guess > self.target:
            state_delta, terminate = f"{guess} is too high", False
        else:
            state_delta, terminate = f"{guess} is correct", True
        
        self.reward = max(points, self.reward) # reward = maximum points over time
        return StepResult(
            state_delta=state_delta,
            reward=self.reward,
            terminate=terminate,
        )

from trl.trainer.grpo_trainer import GRPOTrainer

def qwen3_prompt_init(prompt: StateDelta) -> str:
    return "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

def qwen3_prompt_concat(prompt: StateDelta) -> str:
    return "\n" + "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

def qwen3_parse_completion_text(completion_text: str) -> Action:
    completion_text = completion_text.split("</think>")[-1]
    completion_text =  completion_text.split("<|im_end|>")[0]
    completion_text = " ".join(completion_text.split())
    return completion_text

def gemma4_prompt_init(prompt: StateDelta) -> str:
    return "<bos>" + "<|turn>user\n" + prompt + "<turn|>\n<|turn>model\n"

def gemma4_prompt_concat(prompt: StateDelta) -> str:
    return "<|turn>user\n" + prompt + "<turn|>\n<|turn>model\n"

def gemma4_parse_completion_text(completion_text: str) -> Action:
    completion_text = completion_text.split("<channel|>")[-1]
    completion_text =  completion_text.split("<turn|>")[0]
    completion_text = " ".join(completion_text.split())
    return completion_text

def deepseekr1_prompt_init(prompt: StateDelta) -> str:
    return "<｜begin▁of▁sentence｜><｜User｜>" + prompt + "<｜Assistant｜><think>\n"

def deepseekr1_prompt_concat(prompt: StateDelta) -> str:
    return "<｜begin▁of▁sentence｜><｜User｜>" + prompt + "<｜Assistant｜><think>\n"

def deepseekr1_parse_completion_text(completion_text: str) -> Action:
    completion_text = completion_text.split("</think>")[-1]
    completion_text =  completion_text.split("<｜end▁of▁sentence｜>")[0]
    completion_text = " ".join(completion_text.split())
    return completion_text


prompt_init = deepseekr1_prompt_init
prompt_concat = deepseekr1_prompt_concat
parse_completion_text = deepseekr1_parse_completion_text

def tokenizer_encode(tokenizer, model, input_text: str) -> torch.Tensor:
    i = tokenizer(text=input_text, return_tensors="pt").to(model.device)
    prompt_ids = i.input_ids.squeeze()
    return prompt_ids

def tokenizer_decode(tokenizer, model, completions_ids: torch.Tensor) -> str:
    return tokenizer.decode(completions_ids)

def model_generate(tokenizer, model, prompt_ids: torch.Tensor):
    input_ids = prompt_ids.unsqueeze(dim=0) # prompt_ids is of shape (m,)
    attention_mask = torch.ones_like(input_ids)
    o = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=256,
        eos_token_id=[tokenizer.eos_token_id],              # stop generation when receiving eos_token_id <|im_end|>
        output_logits=True,
        return_dict_in_generate=True,
    )
    # o.sequences is of shape (1, m + n)
    # o.logits is of shape (n, 1, d) where n is len(completion) and d is the number of tokens

    completions_ids = o.sequences[:, -len(o.logits) :][0, :]    # completions_ids is of shape (n,)
    logprobs = torch.cat(o.logits)                              # logprobs is of shape (n, d)

    return {
        "completions_ids": completions_ids,
        "logprobs": logprobs,
    }

def rollout_once(tokenizer, model, env: Env, seed: Seed):
    MAX_TURNS = 999

    completions_ids_list = []
    logprobs_list = []
    env_mask_list = []

    
    state_delta = env.reset(seed=seed)
    print("user>\t", state_delta)
    # original_prompt_ids is of shape (m,)
    original_prompt_ids: torch.Tensor = tokenizer_encode(
        tokenizer=tokenizer, model=model,
        input_text=prompt_init(prompt=state_delta),
    )  

    prompt_ids: torch.Tensor = original_prompt_ids
    

    assert MAX_TURNS >= 1
    for turn in range(MAX_TURNS):
        # MODEL GENERATE
        generate = model_generate(tokenizer, model, prompt_ids)
        completions_ids, logprobs = generate["completions_ids"], generate["logprobs"]
        d = logprobs.shape[1]

        completions_ids_list.append(completions_ids)
        env_mask_list.append(torch.ones_like(completions_ids))
        logprobs_list.append(logprobs)
        

        # INTERACT WITH ENVIRONMENT
        completion_text = tokenizer_decode(
            tokenizer=tokenizer, model=model,
            completions_ids=completions_ids,
        )
        print("agent>\t", completion_text)

        result = env.step(parse_completion_text(completion_text))

        print("user>\t", result.state_delta)
        if result.terminate:
            break

        # assuming tokenizer is additive
        # tok(a ++ b) = tok(a) ++ tok(b)
        state_delta_ids = tokenizer_encode(
            tokenizer=tokenizer, model=model,
            input_text=prompt_concat(result.state_delta),
        )
        prompt_ids = torch.cat([prompt_ids, completions_ids, state_delta_ids])

        completions_ids_list.append(state_delta_ids)
        env_mask_list.append(torch.zeros_like(state_delta_ids))
        logprobs_list.append(torch.zeros(size=[len(state_delta_ids), d]))


    return {
        "prompt_ids": original_prompt_ids,
        "completion_ids": torch.cat(completions_ids_list),
        "env_mask": torch.cat(env_mask_list), # Support custom env_mask from rollout_func (e.g., for environment feedback masking)
        "logprobs": torch.cat(logprobs_list),
        "env_reward": result.reward,
    }

def rollout_func(prompts: list[str], trainer: GRPOTrainer):
    output_list = {}
    for prompt in prompts:
        o = rollout_once(trainer.processing_class, trainer.model, GuessEnv(), prompt)
        for k, v in o.items():
            if k not in output_list:
                output_list[k] = []
            output_list[k].append(v)
    return output_list



def load_model_and_tokenizer(model_path: str, lora: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path)
    if tokenizer.padding_side is None:
        tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token # <|im_end|>

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
    )
    if not lora:
        return tokenizer, model

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        inference_mode=False,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    return tokenizer, model


if __name__ == "__main__":
    model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    tokenizer, model = load_model_and_tokenizer(model_path, lora=False)
    model = model.to("mps")

    o = rollout_once(tokenizer, model, GuessEnv(), "45")
    print(o)
