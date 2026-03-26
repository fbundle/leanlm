import sys

import mlx_lm
import mlx_lm.sample_utils

def enable_thinking(prompt: str) -> str:
    prompt = prompt.rstrip()
    prompt = prompt.rstrip("</think>")
    prompt = prompt.rstrip()
    prompt = prompt + "\n\n"

    if "</think>" in prompt:
        raise RuntimeError(f"enable_thinking: {prompt}")

    return prompt

def main(model_path: str):
    parts = mlx_lm.load(path_or_hf_repo=model_path)
    if len(parts) != 2:
        return
    model, tokenizer = parts

    messages = [{"role": "user", "content": "12345+67890="}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt = enable_thinking(prompt)

    sampler = mlx_lm.sample_utils.make_sampler(
        temp=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
    )

    for response in mlx_lm.stream_generate(
        model=model, tokenizer=tokenizer, prompt=prompt,
        max_tokens=32768,
        sampler=sampler,
    ):
        print(response.text, end="", flush=True)
    print()

if __name__ == "__main__":
    main(sys.argv[1])
