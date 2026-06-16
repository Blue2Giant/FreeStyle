import shutil, os, time

FILES = [
    ("/mnt/jfs/debug_sref_entropy_0429_cref_sref_full_diffusion_from36000_rope_fa_8gpu_from_no_illutrious_base/0505_qwen_cref_sref_full_diffusion_from40000_rope_fa/converted/checkpoint-50000/model.safetensors", "/data/FreeStyle/cref_checkpoints/cref-sref-50000-rope.safetensors"),
    ("/mnt/jfs/debug_sref_entropy_0426_cref_sref_full_diffusion_no_illustrious/0426_qwen_cref_sref_full_diffusion/converted/checkpoint-40000/model.safetensors", "/data/FreeStyle/cref_checkpoints/cref-sref-40000-no-rope.safetensors"),
    ("/mnt/jfs/model_zoo/checkpoint-36000_converted/checkpoint-36000.safetensors", "/data/FreeStyle/cref_checkpoints/cref-sref-36000-no-rope.safetensors"),
]

os.makedirs("/data/FreeStyle/cref_checkpoints", exist_ok=True)
t0 = time.time()
for src, dst in FILES:
    sz = os.path.getsize(src) / (1024**3)
    print(f"Copying {os.path.basename(dst)} ({sz:.1f}G)...")
    t1 = time.time()
    shutil.copy2(src, dst)
    elapsed = time.time() - t1
    print(f"  Done in {elapsed:.0f}s ({sz/elapsed:.1f} MB/s)")
total = time.time() - t0
print(f"All copies done in {total:.0f}s ({total/60:.1f} min)")
