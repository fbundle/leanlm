import os

import mlx_lm

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

def main(output_dir: str, model_path: str):
    model_path_mlx = convert_model(output_dir, model_path)
    model, tokenizer = mlx_lm.load(model_path_mlx)

    prompt = "Write a story about Einstein"

    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True,
    )

    for response in mlx_lm.stream_generate(model, tokenizer, prompt, max_tokens=512):
        print(response.text, end="", flush=True)
    print()

if __name__ == "__main__":
    main(OUTPUT_DIR, MODEL_PATH)
