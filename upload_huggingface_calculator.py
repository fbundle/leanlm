from huggingface_hub import login, upload_large_folder
import datetime


OUTPUT_DIR = "mnt/output/qwen3.5-4b-lora-calculator"
REPO_ID = "khanh2023/qwen3.5-4b-lora-calculator"

now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S%z")
with open(f"{OUTPUT_DIR}/last_poll.txt", "w") as f:
    f.write(now)

login()
upload_large_folder(
    folder_path=OUTPUT_DIR,
    repo_id=REPO_ID,
    repo_type="model",
)

