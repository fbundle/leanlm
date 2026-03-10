import os
import shutil
from huggingface_hub import login, upload_large_folder

OUTPUT_DIR = "mnt/output/calculator_qwen3_0p6b_lora_v1"
REPO_ID = "khanh2023/qwen3-0.6b-lora-calculator"
CODE = "leanlm"

if os.path.exists(f"{OUTPUT_DIR}/{CODE}"):
    shutil.rmtree(f"{OUTPUT_DIR}/{CODE}")

shutil.copytree(CODE, f"{OUTPUT_DIR}/{CODE}")

login()
upload_large_folder(
    folder_path=OUTPUT_DIR,
    repo_id=REPO_ID,
    repo_type="model",
)
