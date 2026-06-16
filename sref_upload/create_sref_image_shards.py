#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path
from datetime import datetime, timezone


def count_files(path: Path) -> int:
    p = subprocess.run(['bash','-lc', f"find {str(path)!r} -type f | wc -l"], text=True, capture_output=True, check=True)
    return int(p.stdout.strip() or '0')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/mnt/jfs/gemini_sref_export')
    ap.add_argument('--out', default='/mnt/jfs/gemini_sref_export_shards')
    ap.add_argument('--prefixes', default='', help='comma-separated image prefix dirs; default all dirs under images/')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    root = Path(args.root).resolve()
    images = root / 'images'
    out = Path(args.out).resolve()
    shard_dir = out / 'image_tar_shards'
    shard_dir.mkdir(parents=True, exist_ok=True)
    if not images.is_dir():
        print(f'[ERR] images dir not found: {images}', file=sys.stderr)
        return 2
    if args.prefixes:
        prefixes = [x.strip() for x in args.prefixes.split(',') if x.strip()]
    else:
        prefixes = sorted([p.name for p in images.iterdir() if p.is_dir()])
    print(f'[start] root={root}', flush=True)
    print(f'[start] out={out}', flush=True)
    print(f'[start] prefixes={len(prefixes)}', flush=True)

    manifest_path = out / 'shards_manifest.jsonl'
    progress_path = out / '_shard_create_progress.jsonl'
    existing_done = set()
    if progress_path.exists() and not args.force:
        for line in progress_path.read_text(encoding='utf-8', errors='replace').splitlines():
            try:
                rec = json.loads(line)
                if rec.get('status') == 'ok':
                    existing_done.add(rec.get('prefix'))
            except Exception:
                pass

    records = []
    t_all = time.time()
    for idx, prefix in enumerate(prefixes, 1):
        src = images / prefix
        tar_path = shard_dir / f'images_{prefix}.tar'
        tmp_path = shard_dir / f'.images_{prefix}.tar.tmp'
        if not src.is_dir():
            print(f'[warn] missing prefix dir: {src}', flush=True)
            continue
        if tar_path.exists() and tar_path.stat().st_size > 0 and prefix in existing_done and not args.force:
            n = count_files(src)
            rec = {'prefix': prefix, 'shard': f'image_tar_shards/{tar_path.name}', 'bytes': tar_path.stat().st_size, 'file_count': n, 'status': 'ok_existing'}
            records.append(rec)
            print(f'[skip {idx}/{len(prefixes)}] {tar_path.name} exists size={rec["bytes"]} files={n}', flush=True)
            continue
        if args.force:
            tmp_path.unlink(missing_ok=True)
            tar_path.unlink(missing_ok=True)
        elif tar_path.exists() and tar_path.stat().st_size > 0:
            # Trust completed tar if it exists, even if progress file was lost.
            n = count_files(src)
            rec = {'prefix': prefix, 'shard': f'image_tar_shards/{tar_path.name}', 'bytes': tar_path.stat().st_size, 'file_count': n, 'status': 'ok_existing_no_progress'}
            records.append(rec)
            with progress_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps({**rec, 'created_at': datetime.now(timezone.utc).isoformat()}, ensure_ascii=False) + '\n')
            print(f'[skip {idx}/{len(prefixes)}] {tar_path.name} exists size={rec["bytes"]} files={n}', flush=True)
            continue
        tmp_path.unlink(missing_ok=True)
        n = count_files(src)
        print(f'[tar {idx}/{len(prefixes)}] prefix={prefix} files={n} -> {tar_path}', flush=True)
        t0 = time.time()
        # Store paths like images/0a/xxx.jpg inside tar so extraction at dataset root restores the expected layout.
        cmd = ['tar', '-cf', str(tmp_path), '-C', str(root), f'images/{prefix}']
        try:
            subprocess.run(cmd, check=True)
            tmp_path.rename(tar_path)
            rec = {
                'prefix': prefix,
                'shard': f'image_tar_shards/{tar_path.name}',
                'bytes': tar_path.stat().st_size,
                'file_count': n,
                'status': 'ok',
                'seconds': round(time.time() - t0, 3),
            }
            records.append(rec)
            with progress_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps({**rec, 'created_at': datetime.now(timezone.utc).isoformat()}, ensure_ascii=False) + '\n')
            print(f'[tar {idx}/{len(prefixes)}] OK {tar_path.name} size={rec["bytes"]/1024/1024/1024:.2f} GiB in {rec["seconds"]:.1f}s', flush=True)
        except Exception as e:
            tmp_path.unlink(missing_ok=True)
            rec = {'prefix': prefix, 'status': 'error', 'error': repr(e), 'seconds': round(time.time()-t0, 3)}
            with progress_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps({**rec, 'created_at': datetime.now(timezone.utc).isoformat()}, ensure_ascii=False) + '\n')
            print(f'[tar {idx}/{len(prefixes)}] ERROR {e!r}', flush=True)
            return 1

    # Rebuild manifest from all completed tar files so it is complete after resume.
    final_records = []
    for prefix in prefixes:
        src = images / prefix
        tar_path = shard_dir / f'images_{prefix}.tar'
        if src.is_dir() and tar_path.exists() and tar_path.stat().st_size > 0:
            final_records.append({
                'prefix': prefix,
                'shard': f'image_tar_shards/{tar_path.name}',
                'bytes': tar_path.stat().st_size,
                'file_count': count_files(src),
            })
    with manifest_path.open('w', encoding='utf-8') as f:
        for rec in final_records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    total_bytes = sum(r['bytes'] for r in final_records)
    total_files = sum(r['file_count'] for r in final_records)
    summary = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'source_root': str(root),
        'shard_dir': 'image_tar_shards',
        'num_shards': len(final_records),
        'total_image_files': total_files,
        'total_shard_bytes': total_bytes,
        'layout': 'Each tar contains paths like images/00/<image>.jpg. Extract at dataset root.',
    }
    (out / 'shards_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (out / 'README.md').write_text(f'''# SREF Image Tar Shards\n\nThis folder contains tar shards for the SREF OneIG image payload.\n\n- Number of shards: {len(final_records)}\n- Image files covered: {total_files:,}\n- Total tar bytes: {total_bytes:,}\n\nEach tar preserves paths such as `images/00/<image>.jpg`. To reconstruct the original layout next to `pairs.csv` and `metadata.jsonl`, download the shards and run:\n\n```bash\nfor f in image_tar_shards/images_*.tar; do tar -xf "$f"; done\n```\n\nThen image references in `pairs.csv` / `metadata.jsonl` resolve under `images/`.\n''', encoding='utf-8')
    print(f'[done] shards={len(final_records)} files={total_files:,} size={total_bytes/1024/1024/1024:.2f} GiB elapsed={(time.time()-t_all)/3600:.2f}h', flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
