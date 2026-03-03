from huggingface_hub import snapshot_download


OUTPUT_DIR = "mnt/output/calculator_qwen3p5_0p8b_lora"
REPO_ID = "khanh2023/qwen3.5-4b-lora-calculator"

snapshot_download(
    local_dir=OUTPUT_DIR,
    repo_id=REPO_ID,
)