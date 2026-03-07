from huggingface_hub import snapshot_download


OUTPUT_DIR = "mnt/output/calculator_qwen3_0p6b_lora_v1"
REPO_ID = "khanh2023/qwen3-0.6b-v1-lora-calculator"

snapshot_download(
    local_dir=OUTPUT_DIR,
    repo_id=REPO_ID,
)