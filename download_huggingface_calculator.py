from huggingface_hub import snapshot_download


local_dir = "mnt/output/calculator"


snapshot_download(
    local_dir="mnt/output/calculator",
    repo_id="khanh2023/qwen3-0.6b-lora-calculator",
)