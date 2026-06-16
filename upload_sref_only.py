import os
from huggingface_hub import HfApi, create_repo

TOKEN = "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"
REPO = "FreeStyle_Checkpoint"

FILES = [
    {"path": "/mnt/jfs/debug_sre_enrichment_new_0415_h100_from_12000-new/0415_qwen_image_sref_noise_query/converted/checkpoint-14000/model.safetensors", "repo_path": "freestyle-sref-14000-no-rope/model.safetensors"},
    {"path": "/mnt/jfs/model_zoo/checkpoint-12000_converted/model.safetensors", "repo_path": "freestyle-sref-12000-no-rope/model.safetensors"},
]

api = HfApi(token=TOKEN)
user = api.whoami()
username = user["name"]
repo_id = f"{username}/{REPO}"
print(f"User: {username}")
print(f"Repo: https://huggingface.co/{repo_id}")
create_repo(repo_id=repo_id, token=TOKEN, repo_type="model", exist_ok=True, private=False)
print("Repo ready")

for item in FILES:
    src, dst = item["path"], item["repo_path"]
    if not os.path.exists(src):
        print(f"SKIP: {src}"); continue
    sz = os.path.getsize(src) / (1024**3)
    print(f"Uploading {dst} ({sz:.1f}G)...")
    api.upload_file(path_or_fileobj=src, path_in_repo=dst, repo_id=repo_id, repo_type="model", token=TOKEN, commit_message=f"Add {dst}")
    print(f"  Done: {dst}")

print("All sref uploads complete!")
