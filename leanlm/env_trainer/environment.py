from typing import Protocol

type Action = str
type Delta = str
type Seed = str

class Env(Protocol):
    last_step_reward: float
    alive: bool
    def reset(self, seed: Seed) -> Delta: ...
    def step(self, action: Action) -> Delta: ...

# some examples below
class GuessEnv(Env):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def reset(self, seed: Seed) -> Delta:
        self.target = int(seed)
        self.best_reward = 0
        self.last_step_reward = 0
        self.alive = True
        return """
I have an integer between 0 and 10 in mind
every turn, you have to take a guess, output
GUESS <number>
I will say if your guess is higher or lower than my number
"""
    
    def step(self, action: Action) -> Delta:
        words = action.split()
        if "GUESS" not in words:
            self.alive = False
            self.last_step_reward = 0
            return f"can't find your guess"
        
        guess_str = words[words.index("GUESS") + 1]

        try:
            guess = int(guess_str)
        except ValueError:
            guess = None

        if guess is None:
            self.alive = False
            self.last_step_reward = 0
            return f"can't find the number in your guess"

        f = lambda x: 1 / (1 + x) # map [0, inf) -> [1, 0)
        points = f(abs(self.target - guess))
        if guess < self.target:
            state_delta = f"{guess} is too low"
        elif guess > self.target:
            state_delta = f"{guess} is too high"
        else:
            state_delta = f"{guess} is correct"
            self.alive = False
        
        # best_reward = maximum points over time
        if points > self.best_reward:
            self.last_step_reward = points - self.best_reward
            self.best_reward = points
        else:
            self.last_step_reward = 0        
        return state_delta