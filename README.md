# LEANLM

A toy project of finetuning LLM with math data: [stacks](https://stacks.math.columbia.edu/), [arxiv](https://arxiv.org/archive/math), [math.SE](https://math.stackexchange.com/), [mathoverflow](https://mathoverflow.net/), [mathlib4](https://github.com/leanprover-community/mathlib4), [trl-lib/DeepMath-103K](https://huggingface.co/datasets/trl-lib/DeepMath-103K) etc.

with the final goal of RL on [lean4](https://github.com/leanprover/lean4) theorem verifier


## SIMPLE CALCULATOR APP

input examples: `123+456=`, `123*456=`

- [khanh2023/qwen3-0.6b-lora-calculator](https://huggingface.co/khanh2023/qwen3-0.6b-lora-calculator)

# REFERENCES

- [lean_dojo](https://github.com/lean-dojo/LeanDojo)

- [elan](https://github.com/leanprover/elan)

- [GPROTrainer](https://huggingface.co/docs/trl/en/grpo_trainer)

- [Yuval Ran-Milo, Yotam Alexander, Shahar Mendel, Nadav Cohen](https://arxiv.org/abs/2601.15158)

- [gazelle93](https://github.com/gazelle93/llm-fine-tuning-sft-lora-qlora)

# HOW TO UV

1. create a new mamba environment with go, python, uv and init uv project

```shell
mamba create --name leanlm 
mamba activate leanlm
mamba install "go==1.26.*" "python==3.12.*"
pip install "uv==0.11.*"
uv init .
```

2. export mamba environment

```shell
mamba env export --from-history | grep -v "^prefix: " > environment.yml
```

3. add and remove packages using `uv add` and `uv remove`

4. run the project using `uv run python`

