from typing import Protocol

type Action = str
type StateDelta = str
type Seed = str

class Env(Protocol):
    reward: float
    terminate: bool
    def reset(self, seed: Seed) -> StateDelta: ...
    def step(self, action: Action) -> StateDelta: ...

# some examples below
import math
import jiwer

def get_int(s: str) -> int | None:
    try:
        return int(s)
    except Exception:
        return None

def get_int2(s: str) -> tuple[int, int, bool]:
    try:
        a, b = s.split()
        return int(a), int(b), True
    except Exception:
        return 0, 0, False


class GcdEnv(Env):
    def reset(self, seed: Seed) -> StateDelta:
        a_str, b_str = seed.split()
        a, b = int(a_str), int(b_str)
        self.gcd = math.gcd(a, b)
        self.reward = 0
        self.terminate = False
        return f"""
calculate the GCD of {a} and {b}
every turn, you are able to either output the answer by
ANSWER <answer>
or output 
SUBTRACT <number1> <number2>
I will help you to calculate the difference between two numbers with absolute precision
every turn, you can output a maximum number of 128 tokens
the whole conversation should not last longer than 4096 tokens
"""
    
    def step(self, action: Action) -> StateDelta:
        f = lambda x: 1 / (1 + x) # map [0, inf) -> [1, 0)
        parts = action.split("SUBTRACT ")
        if len(parts) >= 2: # detected subtract
            last = parts[-1]
            a, b, ok = get_int2(last)
            if not ok:
                format_reward, answer_reward = 0, 0
                self.reward = format_reward + answer_reward
                self.terminate = True
                return f"subtract_format_error: {last}"
            else:
                format_reward = f(jiwer.cer(f"{a} {b}", last))
                answer_reward = 0
                self.reward = format_reward + answer_reward
                return f"subtract: {a} - {b} = {a - b}"
        
        parts = action.split("ANSWER ")
        if len(parts) >= 2: # detected answer
            self.terminate = True
            last = parts[-1]
            answer = get_int(last)
            if answer is None:
                format_reward, answer_reward = 0, 0
                self.reward = format_reward + answer_reward
                
                return f"answer_format_error: {last}"
            else:
                format_reward = f(jiwer.cer(str(answer), last))
                answer_reward = 1 if answer == self.gcd else 0
                self.reward = format_reward + answer_reward
                if answer_reward == 1:
                    return f"answer_correct: {answer}"
                else:
                    return f"answer_wrong: {answer}"
        
        format_reward, answer_reward = 0, 0
        self.reward = format_reward + answer_reward
        self.terminate = True
        return "format_error"

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
        self.terminate = False
        return """
I have an integer between 0 and 50 in mind
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
        
        self.reward = max(points, self.reward) # reward = maximum points over time
        return state_delta