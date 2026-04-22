from typing import Protocol

type Action = str
type StateDelta = str
type Seed = str

class Env(Protocol):
    last_step_reward: float
    terminate: bool
    def reset(self, seed: Seed) -> StateDelta: ...
    def step(self, action: Action) -> StateDelta: ...

# some examples below

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
        self.best_reward = 0
        self.last_step_reward = 0
        self.terminate = False
        return """
I have an integer between 0 and 10 in mind
every turn, you have to take a guess, output
GUESS <number>
I will say if your guess is higher or lower than my number
"""
    
    def step(self, action: Action) -> StateDelta:
        # use regex to get the last integer
        guess = get_last_integer(action)
        
        if guess is None:
            self.terminate = True
            return f"can't find the number in your input"
        

        f = lambda x: 1 / (1 + x) # map [0, inf) -> [1, 0)
        points = f(abs(self.target - guess))
        if guess < self.target:
            state_delta = f"{guess} is too low"
        elif guess > self.target:
            state_delta = f"{guess} is too high"
        else:
            state_delta = f"{guess} is correct"
            self.terminate = True
        
        # best_reward = maximum points over time
        self.last_step_reward = max(0, points - self.best_reward)
        self.best_reward += self.last_step_reward
        
        return state_delta