import os
from threading import Thread

from huggingface_hub import login, upload_large_folder
import datetime


output_dir = "mnt/output"
user_id = "khanh2023"

now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S%z")

def upload(name: str):
    folder_path = f"{output_dir}/{name}"
    repo_id=f"{user_id}/{name}"

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


ts = []

for name in os.listdir(output_dir):
    t = Thread(target=upload, args=(name))
    ts.append(t)
    t.start()

for t in ts:
    t.join()