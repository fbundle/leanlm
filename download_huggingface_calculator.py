from huggingface_hub import HfApi, RepoFolder, snapshot_download



def download_latest_checkpoint(local_dir: str, repo_id: str, path_in_repo: str = ""):
    path_in_repo = path_in_repo.rstrip("/")

    api = HfApi()
    checkpoint_list: list[tuple[int, str]] = []
    for file in api.list_repo_tree(repo_id=repo_id, path_in_repo=path_in_repo):
        path = file.path
        name = path.lstrip(path_in_repo + "/")
        CHECKPOINT_PREFIX = "checkpoint-"
        if isinstance(file, RepoFolder) and name.startswith(CHECKPOINT_PREFIX):
            try:
                step = int(name.lstrip(CHECKPOINT_PREFIX))
                checkpoint_list.append((step, path))
            except:
                print("ERROR: checkpoint", file)
    
    checkpoint_list.sort(key=lambda tup: tup[0])
    step, path = checkpoint_list[-1]

    print(f"downloading {path} ...")
    api.snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        allow_patterns=[path + "/*"],
    )


for name in [
    "qwen3.5-4b-length4096-lora-calculator",
    "qwen3.5-4b-length4096-p0.3-calculator",
    "qwen3.5-4b-length4096-p0.3-phoenix-calculator",
]:
    print("downloading", name)
    OUTPUT_DIR = f"mnt/output/{name}"
    REPO_ID = f"khanh2023/{name}"
    download_latest_checkpoint(local_dir=OUTPUT_DIR, repo_id=REPO_ID)