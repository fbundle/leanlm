from huggingface_hub import login, upload_folder
login()
upload_folder(folder_path="mnt/output/calculator", repo_id="khanh2023/qwen3-0.6b-calculator", repo_type="model")