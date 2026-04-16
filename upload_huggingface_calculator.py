import os
import time

from huggingface_hub import login, upload_large_folder
import datetime

import multiprocess as mp


output_dir = "mnt/output"
user_id = "khanh2023"

now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S%z")

def upload(name: str):
    folder_path = f"{output_dir}/{name}"
    repo_id=f"{user_id}/{name}"

    try:
        os.makedirs(folder_path, exist_ok=True)
        login()

        while True:
            with open(f"{folder_path}/last_poll.txt", "w") as f:
                f.write(now)    
            upload_large_folder(
                folder_path=folder_path,
                repo_id=repo_id,
                repo_type="model",
            )
            time.sleep(10 * 60) # sleep 10 minutes
    except KeyboardInterrupt:
        return


with mp.Pool() as pool:
    pool.map(upload, os.listdir(output_dir))
