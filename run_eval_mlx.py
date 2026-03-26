import os

import mlx_lm
import mlx_lm.sample_utils



OUTPUT_DIR = "mnt/output_mlx"

MODEL_PATH = "Qwen/Qwen3-0.6B"
MODEL_PATH = "Qwen/Qwen3.5-0.8B"

def convert_model(output_dir: str, model_path: str) -> str:
    model_name = os.path.basename(model_path)
    output_path = f"{output_dir}/{model_name}"
    if os.path.exists(output_path):
        return output_path
    
    mlx_lm.convert(
        hf_path=model_path,
        mlx_path=output_path,
        quantize=True,
    )

    return output_path

def enable_thinking(prompt: str) -> str:
    prompt = prompt.rstrip()
    prompt = prompt.rstrip("</think>")
    prompt = prompt.rstrip()
    prompt = prompt + "\n\n"

    if "</think>" in prompt:
        raise RuntimeError(f"enable_thinking: {prompt}")
    return prompt

def main(output_dir: str, model_path: str):
    model_path_mlx = convert_model(output_dir, model_path)
    parts = mlx_lm.load(path_or_hf_repo=model_path_mlx)
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
    main(OUTPUT_DIR, MODEL_PATH)
