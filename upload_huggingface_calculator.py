from huggingface_hub import login, upload_large_folder
import datetime


name = "qwen3.5-4b-length4096-lora-calculator"

folder_path = f"mnt/output/{name}"
repo_id=f"khanh2023/{name}"

now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S%z")
with open(f"{folder_path}/last_poll.txt", "w") as f:
    f.write(now)

login()
upload_large_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    repo_type="model",
)

