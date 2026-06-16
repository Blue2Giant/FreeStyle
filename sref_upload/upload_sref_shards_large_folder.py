#!/usr/bin/env python3
from __future__ import annotations
import argparse, inspect, json, os, shutil, sys, time
from pathlib import Path
from datetime import datetime, timezone
from huggingface_hub import HfApi
try:
    from huggingface_hub import CommitOperationDelete
except Exception:  # pragma: no cover
    CommitOperationDelete = None

FINAL_REMOTE_ROOT = 'sref'
DEFAULT_SOURCE_ROOT = Path('/mnt/jfs/gemini_sref_export')
DEFAULT_STAGING_ROOT = Path('/mnt/jfs/gemini_sref_final_hf_upload_root')


def hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        try:
            if dst.stat().st_size == src.stat().st_size:
                return
        except Exception:
            pass
        dst.unlink()
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def write_final_readme(dst: Path, shard_folder: Path, source_root: Path) -> None:
    summary_path = shard_folder / 'shards_summary.json'
    summary = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding='utf-8'))
        except Exception:
            summary = {}
    num_shards = summary.get('num_shards', 'unknown')
    total_images = summary.get('total_image_files', 'unknown')
    total_bytes = summary.get('total_shard_bytes', 0)
    total_gib = total_bytes / 1024 / 1024 / 1024 if isinstance(total_bytes, (int, float)) else 0
    text = f'''---
pretty_name: SREF OneIG Dataset
---

# SREF OneIG Dataset

This dataset is distributed in **tar shards** for reliable high-throughput download/upload.
The previous per-image remote upload has been replaced by this shard layout.

## Files

- `pairs.csv`: pair-level records linking source/condition/target image paths.
- `metadata.jsonl`: sequence-level metadata.
- `export_summary.json`: export statistics.
- `shards_manifest.jsonl`: one JSON record per tar shard.
- `image_tar_shards/images_XX.tar`: image tar shards.

Shard summary:

- Number of shards: `{num_shards}`
- Image files covered: `{total_images}`
- Total shard size: `{total_gib:.2f} GiB`

## Image layout

Each tar contains paths like:

```text
images/00/<image>.jpg
images/0a/<image>.jpg
...
```

To reconstruct the original image tree next to `pairs.csv` and `metadata.jsonl`:

```bash
mkdir -p sref_data
cd sref_data
# Download pairs.csv, metadata.jsonl, export_summary.json, shards_manifest.jsonl,
# and image_tar_shards/images_*.tar from this folder first.
for f in image_tar_shards/images_*.tar; do
  tar -xf "$f"
done
```

After extraction, image paths referenced by `pairs.csv` / `metadata.jsonl` resolve under `images/`.

## Notes

- The tar files are uncompressed, because the payload is JPEG and already compressed.
- The shard layout avoids millions of tiny remote files and is intended to be robust for large dataset transfer.
- Local source root used to build this package: `{source_root}`.
'''
    dst.write_text(text, encoding='utf-8')


def prepare_staging(shard_folder: Path, staging: Path, source_root: Path) -> Path:
    """Create staging root containing final remote layout `sref/...` using hardlinks.

    We stage as `staging/sref/...` and upload `staging` as repo root because the
    installed huggingface_hub upload_large_folder may not support path_in_repo.
    """
    target = staging / FINAL_REMOTE_ROOT
    staging.mkdir(parents=True, exist_ok=True)
    target.mkdir(parents=True, exist_ok=True)

    # Core metadata at sref root. Do not rely on old remote files because remote
    # sref will be deleted before this new shard package is uploaded.
    for name in ['pairs.csv', 'metadata.jsonl', 'export_summary.json']:
        src = source_root / name
        if not src.is_file():
            raise FileNotFoundError(f'missing required source metadata: {src}')
        hardlink_or_copy(src, target / name)

    # Shard metadata and all tar shards.
    for name in ['shards_manifest.jsonl', 'shards_summary.json']:
        src = shard_folder / name
        if not src.is_file():
            raise FileNotFoundError(f'missing shard metadata: {src}')
        hardlink_or_copy(src, target / name)

    shard_dir = shard_folder / 'image_tar_shards'
    if not shard_dir.is_dir():
        raise FileNotFoundError(f'missing shard dir: {shard_dir}')
    n = 0
    total_bytes = 0
    for src in sorted(shard_dir.glob('images_*.tar')):
        hardlink_or_copy(src, target / 'image_tar_shards' / src.name)
        n += 1
        total_bytes += src.stat().st_size
        if n % 25 == 0:
            print(f'[staging] linked {n} tar files, {total_bytes/1024/1024/1024:.1f} GiB...', flush=True)

    if n == 0:
        raise RuntimeError(f'no tar shards found in {shard_dir}')
    write_final_readme(target / 'README.md', shard_folder, source_root)
    print(f'[staging] ready tar_files={n} tar_bytes={total_bytes/1024/1024/1024:.2f} GiB root={staging}', flush=True)
    return staging


def delete_remote_sref_once(api: HfApi, repo_id: str, repo_type: str, marker: Path) -> None:
    """Delete old remote sref tree once. Marker prevents deleting new partial upload on resume."""
    if marker.exists():
        print(f'[delete] marker exists, skip remote deletion: {marker}', flush=True)
        return
    marker.parent.mkdir(parents=True, exist_ok=True)
    print(f'[delete] deleting previous remote folder {repo_id}/{FINAL_REMOTE_ROOT}/ ...', flush=True)
    try:
        if hasattr(api, 'delete_folder'):
            api.delete_folder(
                repo_id=repo_id,
                repo_type=repo_type,
                path_in_repo=FINAL_REMOTE_ROOT,
                commit_message='Remove previous sref per-file upload before shard upload',
            )
        elif CommitOperationDelete is not None:
            api.create_commit(
                repo_id=repo_id,
                repo_type=repo_type,
                operations=[CommitOperationDelete(path_in_repo=FINAL_REMOTE_ROOT)],
                commit_message='Remove previous sref per-file upload before shard upload',
            )
        else:
            raise RuntimeError('No delete_folder or CommitOperationDelete available')
        marker.write_text(datetime.now(timezone.utc).isoformat() + '\n', encoding='utf-8')
        print('[delete] OK remote old sref removed', flush=True)
    except Exception as e:
        msg = str(e)
        if '404' in msg or 'Entry Not Found' in msg or 'not found' in msg.lower():
            marker.write_text(datetime.now(timezone.utc).isoformat() + '\nnot_found_ok\n', encoding='utf-8')
            print(f'[delete] remote sref not found; continue. ({type(e).__name__}: {e})', flush=True)
        else:
            print(f'[delete] ERROR {type(e).__name__}: {e}', flush=True)
            raise


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--folder', default='/mnt/jfs/gemini_sref_export_shards')
    ap.add_argument('--source-root', default=str(DEFAULT_SOURCE_ROOT))
    ap.add_argument('--repo-id', default='Blue2Giant/FreeStyle_Dataset')
    ap.add_argument('--repo-type', default='dataset')
    # Kept for backward compatibility with the already-running shell command. Ignored intentionally.
    ap.add_argument('--path-in-repo', default=FINAL_REMOTE_ROOT)
    ap.add_argument('--staging-root', default=str(DEFAULT_STAGING_ROOT))
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--skip-delete-old-sref', action='store_true')
    args = ap.parse_args()

    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    if not token:
        print('[ERR] HF_TOKEN missing', file=sys.stderr)
        return 2
    shard_folder = Path(args.folder).resolve()
    source_root = Path(args.source_root).resolve()
    staging = Path(args.staging_root).resolve()
    if not shard_folder.exists():
        print(f'[ERR] folder missing: {shard_folder}', file=sys.stderr)
        return 2
    os.environ.setdefault('HF_XET_HIGH_PERFORMANCE', '1')

    api = HfApi(token=token)
    print(f'[auth] {api.whoami(token=token).get("name")}', flush=True)
    print(f'[layout] final remote root is {FINAL_REMOTE_ROOT}/ ; CLI --path-in-repo={args.path_in_repo!r} is ignored for final layout', flush=True)

    if not args.skip_delete_old_sref:
        delete_remote_sref_once(api, args.repo_id, args.repo_type, staging / '.remote_sref_deleted_ok')

    upload_folder = prepare_staging(shard_folder, staging, source_root)
    sig = inspect.signature(api.upload_large_folder)
    kwargs = dict(repo_id=args.repo_id, repo_type=args.repo_type, folder_path=str(upload_folder))
    if 'num_workers' in sig.parameters:
        kwargs['num_workers'] = args.workers
    elif 'max_workers' in sig.parameters:
        kwargs['max_workers'] = args.workers
    if 'print_report' in sig.parameters:
        kwargs['print_report'] = True
    if 'print_report_every' in sig.parameters:
        kwargs['print_report_every'] = 60
    if 'ignore_patterns' in sig.parameters:
        kwargs['ignore_patterns'] = ['**/.cache/**', '**/*.tmp', '**/.remote_sref_deleted_ok']

    print(f'[start] upload_large_folder staging_root={upload_folder} -> {args.repo_id}/ with files under {FINAL_REMOTE_ROOT}/ workers={args.workers}', flush=True)
    t0 = time.time()
    api.upload_large_folder(**kwargs)
    print(f'[done] upload_large_folder elapsed={(time.time()-t0)/3600:.2f}h', flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
