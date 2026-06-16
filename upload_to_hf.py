import os, sys
from huggingface_hub import HfApi, create_repo

TOKEN = "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"
REPO_NAME = "FreeStyle_Checkpoint"

FILES = [
    {"path": "/mnt/jfs/debug_sre_enrichment_new_0415_h100_from_12000-new/0415_qwen_image_sref_noise_query/converted/checkpoint-14000/model.safetensors", "repo_path": "freestyle-sref-14000-no-rope/model.safetensors"},
    {"path": "/mnt/jfs/model_zoo/checkpoint-12000_converted/model.safetensors", "repo_path": "freestyle-sref-12000-no-rope/model.safetensors"},
    {"path": "/mnt/jfs/debug_sref_entropy_0429_cref_sref_full_diffusion_from36000_rope_fa_8gpu_from_no_illutrious_base/0505_qwen_cref_sref_full_diffusion_from40000_rope_fa/converted/checkpoint-50000/model.safetensors", "repo_path": "freestyle-cref-sref-50000-rope/model.safetensors"},
    {"path": "/mnt/jfs/debug_sref_entropy_0426_cref_sref_full_diffusion_no_illustrious/0426_qwen_cref_sref_full_diffusion/converted/checkpoint-40000/model.safetensors", "repo_path": "freestyle-cref-sref-40000-no-rope/model.safetensors"},
    {"path": "/mnt/jfs/model_zoo/checkpoint-36000_converted/checkpoint-36000.safetensors", "repo_path": "freestyle-cref-sref-36000-no-rope/model.safetensors"},
]

api = HfApi(token=TOKEN)
user = api.whoami()
username = user["name"]
repo_id = f"{username}/{REPO_NAME}"
print(f"Logged in as: {username}")
print(f"Target repo: https://huggingface.co/{repo_id}")

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

print("All uploads complete!")
