from huggingface_hub import login, upload_large_folder
login()
upload_large_folder(
    folder_path="mnt/output/calculator_qwen3p5_4b_lora",
    repo_id="khanh2023/qwen3.5-4b-lora-calculator",
    repo_type="model",
)