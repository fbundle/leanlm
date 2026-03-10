import os
import shutil
from huggingface_hub import login, upload_large_folder

OUTPUT_DIR = "mnt/output/qwen3-0.6b-lora-calculator"
REPO_ID = "khanh2023/qwen3-0.6b-lora-calculator"
CODE_SRC = "leanlm"
CODE_DST = f"{OUTPUT_DIR}/src/{CODE_SRC}"

if os.path.exists(CODE_DST):
    shutil.rmtree(CODE_DST)

if not os.path.exists(os.path.dirname(CODE_DST)):
    os.makedirs(os.path.dirname(CODE_DST))

shutil.copytree(CODE_SRC, CODE_DST)

login()
upload_large_folder(
    folder_path=OUTPUT_DIR,
    repo_id=REPO_ID,
    repo_type="model",
)
