from huggingface_hub import snapshot_download


OUTPUT_DIR = "mnt/output/qwen3.5-0.8b-lora-calculator"
REPO_ID = "khanh2023/qwen3.5-0.8b-lora-calculator"

OUTPUT_DIR = "mnt/output/qwen3-0.6b-lora-calculator"
REPO_ID = "khanh2023/qwen3-0.6b-lora-calculator"

snapshot_download(
    local_dir=OUTPUT_DIR,
    repo_id=REPO_ID,
)