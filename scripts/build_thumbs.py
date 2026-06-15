#!/usr/bin/env python3
"""Generate WebP thumbnails for every image referenced from data/gallery.json.

Originals stay untouched. Thumbnails mirror the source path under
``assets/thumbs/`` with the extension changed to ``.webp``::

    assets/cref_sref_result/00-cref.png
        -> assets/thumbs/cref_sref_result/00-cref.webp

Run from repo root::

    python3 scripts/build_thumbs.py
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
GALLERY_JSON = REPO_ROOT / "data" / "gallery.json"
THUMB_ROOT = REPO_ROOT / "assets" / "thumbs"
MAX_EDGE = 800
QUALITY = 80


def collect_sources() -> list[Path]:
    with GALLERY_JSON.open() as fh:
        data = json.load(fh)
    seen: set[Path] = set()
    out: list[Path] = []
    for sample in data["samples"]:
        for rel in sample["images"].values():
            src = (REPO_ROOT / rel).resolve()
            if src in seen:
                continue
            seen.add(src)
            out.append(src)
    return out


def thumb_path_for(src: Path) -> Path:
    rel = src.relative_to(REPO_ROOT / "assets")
    return THUMB_ROOT / rel.with_suffix(".webp")


def needs_rebuild(src: Path, dst: Path) -> bool:
    if not dst.exists():
        return True
    return dst.stat().st_mtime < src.stat().st_mtime


def convert_one(src_str: str) -> tuple[str, str]:
    src = Path(src_str)
    dst = thumb_path_for(src)
    if not needs_rebuild(src, dst):
        return ("skip", str(dst))
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB") if im.mode in ("RGBA", "P", "LA") else im
        im.thumbnail((MAX_EDGE, MAX_EDGE), Image.LANCZOS)
        im.save(dst, "WEBP", quality=QUALITY, method=6)
    return ("build", str(dst))


def main() -> int:
    sources = collect_sources()
    print(f"[build_thumbs] {len(sources)} unique source images")
    THUMB_ROOT.mkdir(parents=True, exist_ok=True)

    built = skipped = 0
    failures: list[tuple[str, str]] = []
    workers = max(1, (os.cpu_count() or 4) - 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(convert_one, str(src)): src for src in sources}
        for fut in as_completed(futures):
            src = futures[fut]
            try:
                action, _ = fut.result()
            except Exception as exc:  # noqa: BLE001
                failures.append((str(src), repr(exc)))
                continue
            if action == "build":
                built += 1
            else:
                skipped += 1

    print(f"[build_thumbs] built={built}  skipped={skipped}  failed={len(failures)}")
    for src, err in failures:
        print(f"  FAIL {src}: {err}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
