# LEANLM

A toy project of finetuning LLM with math data: [stacks](https://stacks.math.columbia.edu/), [arxiv](https://arxiv.org/archive/math), [math.SE](https://math.stackexchange.com/), [mathoverflow](https://mathoverflow.net/), [mathlib4](https://github.com/leanprover-community/mathlib4), [trl-lib/DeepMath-103K](https://huggingface.co/datasets/trl-lib/DeepMath-103K) etc.

with the final goal of RL on [lean4](https://github.com/leanprover/lean4) theorem verifier


## SIMPLE CALCULATOR APP

input examples: `123+456=`, `123*456=`

- [khanh2023/qwen3-0.6b-calculator](https://huggingface.co/khanh2023/qwen3.5-4b-calculator)

## SOME IDEAS

- currently, our reward is the sum of CER reward and arithmetic reward, that is, projecting different scales into real number. maybe, we can try other RL algorithm that support non-total order, e.g. if two outputs are not integers, they're equally bad. if only one output is integer, they're better. if two outputs are integers, the closer value is better. if two outputs are correct, the one with shorter reasoning is better

```python
def reward_func(question: str, reason: str, answer: str) -> float:
    expected = get_expected_output(question)
    f = lambda x: 1 / (1 + x) # convert [0, +inf] -> [1, 0]

    # cer reward
    cer = jiwer.cer(expected, answer)
    cer_reward = f(cer)

    # arithmetic reward
    e: int = int(expected)
    a: int | None = None
    try:
        a = int(answer)
    except ValueError:
        pass

    if a is None:
        arith_reward = 0
    else:
        e = max(1, abs(e))
        diff = abs((a - e) / e)
        arith_reward = f(diff)
```

- we could try to prompt the model as follows: `<max_completion_length> : <number1> <operator> <number2>` where `max_completion_length` is like an input to the model saying how many tokens it can generate, from that information, model might be able to learn whether to estimate the answer if `max_completion_length` is too little and explore a many possibilities if `max_completion_length` is big enough. in real world, this is called exam duration


# REFERENCES

- [lean_dojo](https://github.com/lean-dojo/LeanDojo)

- [elan](https://github.com/leanprover/elan)

- [GPROTrainer](https://huggingface.co/docs/trl/en/grpo_trainer)

- [Yuval Ran-Milo, Yotam Alexander, Shahar Mendel, Nadav Cohen](https://arxiv.org/abs/2601.15158)

- [gazelle93](https://github.com/gazelle93/llm-fine-tuning-sft-lora-qlora)

# HOW TO UV

1. install uv

```shell
brew install uv # on mac
pip install uv  # on linux
```

2. create a new uv project with cpython 3.12

```shell
v init --bare --no-workspace .
uv python install 3.12
```

3. add and remove packages using `uv add` and `uv remove`

4. apply the environment to current shell by `source .venv/bin/activate` or run the project using `uv run python <script>`

5. install dependencies on new machine using `uv sync`

