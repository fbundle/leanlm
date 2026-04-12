import os

from huggingface_hub import login, upload_large_folder
import datetime


name = "qwen3.5-4b-length4096-lora-calculator"
now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S%z")

for name in [
    "qwen3.5-4b-length4096-p0.3-lora-calculator",
    "qwen3.5-4b-length4096-p0.3-calculator",
    "qwen3.5-4b-length4096-p0.3-phoenix-calculator",
]:
    folder_path = f"mnt/output/{name}"
    repo_id=f"khanh2023/{name}"

    try:
        os.makedirs(folder_path, exist_ok=True)

        with open(f"{folder_path}/last_poll.txt", "w") as f:
            f.write(now)

        login()
        upload_large_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type="model",
        )
    except FileNotFoundError as e:
        print(e)

