from huggingface_hub import login, upload_large_folder

OUTPUT_DIR = "mnt/output/calculator_qwen3p5_0p8b_lora"
REPO_ID = "khanh2023/qwen3.5-4b-lora-calculator"

login()
upload_large_folder(
    folder_path=OUTPUT_DIR,
    repo_id=REPO_ID,
    repo_type="model",
)