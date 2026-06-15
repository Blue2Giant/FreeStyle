#!/usr/bin/env python3
import argparse
import base64
import json
import os
import queue
import threading
from io import BytesIO
from pathlib import Path
from typing import Dict, Tuple, Optional
from urllib.parse import urlparse
 
import requests
from PIL import Image
 
 
ASPECT_RATIO = [
    "1024x1024",
    "864x1152",
    "1152x864",
    "1280x720",
    "720x1280",
    "832x1248",
    "1248x832",
    "1512x648",
    "2048x2048",
    "2304x1728",
    "1728x2304",
    "2848x1600",
    "1600x2848",
    "2496x1664",
    "1664x2496",
    "3136x1344",
    "4096x4096",
    "3520x4704",
    "4704x3520",
    "5504x3040",
    "3040x5504",
    "3328x4992",
    "4992x3328",
    "6240x2656",
]
 
ASPECT_RATIO_1K = [
    "1024x1024",
    "864x1152",
    "1152x864",
    "1280x720",
    "720x1280",
    "832x1248",
    "1248x832",
    "1512x648",
]
 
 
def _parse_size(s: str) -> Tuple[int, int]:
    w, h = s.lower().split("x", 1)
    return int(w), int(h)
 
 
def _select_size(width: int, height: int, candidates: list[str], min_area: int = 0) -> Optional[str]:
    target = width / float(height)
    best = None
    best_diff = None
    best_area = None
    for s in candidates:
        w, h = _parse_size(s)
        area = w * h
        if min_area and area < min_area:
            continue
        diff = abs((w / float(h)) - target)
        if best is None or diff < best_diff or (diff == best_diff and area < best_area):
            best = s
            best_diff = diff
            best_area = area
    return best
 
 
def _encode_image_data_url(path: Path, fmt: str) -> str:
    img = Image.open(path).convert("RGB")
    buf = BytesIO()
    if fmt == "jpeg":
        img.save(buf, format="JPEG", quality=90)
        mime = "image/jpeg"
    elif fmt == "png":
        img.save(buf, format="PNG")
        mime = "image/png"
    else:
        raise ValueError(f"unsupported fmt: {fmt}")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{b64}"
 
 
def _extract_min_area(msg: str) -> Optional[int]:
    needle = "image size must be at least "
    if needle not in msg:
        return None
    digits = ""
    started = False
    for ch in msg[msg.index(needle) + len(needle) :]:
        if ch.isdigit():
            digits += ch
            started = True
        elif started:
            break
    if not digits:
        return None
    try:
        return int(digits)
    except Exception:
        return None
 
 
def _list_images(dir_path: Path) -> Dict[str, Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    out: Dict[str, Path] = {}
    for p in dir_path.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            out[p.stem] = p
    return out


def _sleep_or_stop(stop_event: threading.Event, seconds: float) -> bool:
    if seconds <= 0:
        return stop_event.is_set()
    return stop_event.wait(seconds)


def _extract_error_body_and_msg(resp: requests.Response) -> Tuple[Optional[object], str]:
    try:
        body = resp.json()
    except Exception:
        body = resp.text[:2000]

    if resp.status_code != 200:
        if isinstance(body, dict):
            return body, str(body.get("msg", ""))
        return body, str(body)

    if isinstance(body, dict):
        code = body.get("code")
        msg = str(body.get("msg", ""))
        data = body.get("data")
        if (code not in (None, 0, 200)) or (msg and not data):
            return body, msg

    return None, ""


def _request_with_retries(
    session: requests.Session,
    method: str,
    url: str,
    *,
    attempts: int,
    backoff_s: float,
    timeout,
    stop_event: threading.Event,
    label: str = "",
    **kwargs,
) -> requests.Response:
    attempts = max(1, int(attempts))
    backoff_s = max(0.0, float(backoff_s))
    last_err = None
    for attempt in range(1, attempts + 1):
        if stop_event.is_set():
            raise RuntimeError("interrupted")
        try:
            return session.request(method=method, url=url, timeout=timeout, **kwargs)
        except requests.RequestException as e:
            last_err = e
            if attempt >= attempts:
                raise
            suffix = f" {label}" if label else ""
            print(
                f"[RETRY]{suffix} method={method} attempt={attempt}/{attempts} "
                f"err={type(e).__name__}: {e}",
                flush=True,
            )
            if _sleep_or_stop(stop_event, backoff_s * attempt):
                raise RuntimeError("interrupted")
    raise last_err


def _worker(
    task_queue: "queue.Queue[Optional[Tuple[str, str, Path, Path, Path]]]",
    result_queue: "queue.Queue[Tuple[str, bool, str]]",
    stop_event: threading.Event,
    worker_args: dict,
) -> None:
    while not stop_event.is_set():
        try:
            item = task_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        try:
            if item is None:
                return
            k, prompt, cref_path, sref_path, out_path = item
            result = _gen_one(
                key=k,
                prompt=prompt,
                cref_path=cref_path,
                sref_path=sref_path,
                out_path=out_path,
                stop_event=stop_event,
                **worker_args,
            )
        except Exception as e:
            result = (k, False, f"worker_exception:{type(e).__name__}:{e}")
        finally:
            task_queue.task_done()
        result_queue.put(result)
 
 
def _gen_one(
    key: str,
    prompt: str,
    cref_path: Path,
    sref_path: Path,
    out_path: Path,
    api_key: str,
    base_url: str,
    model: str,
    resolution: str,
    save_resolution: str,
    image_format: str,
    timeout_s: int,
    download_timeout_s: int,
    connect_timeout_s: int,
    request_retries: int,
    download_retries: int,
    retry_backoff_s: float,
    overwrite: bool,
    stop_event: threading.Event,
) -> Tuple[str, bool, str]:
    if (not overwrite) and out_path.exists():
        return key, True, "skipped"
    if stop_event.is_set():
        return key, False, "interrupted"
 
    content_img = Image.open(cref_path).convert("RGB")
    if resolution:
        size = resolution
    else:
        size = _select_size(content_img.width, content_img.height, ASPECT_RATIO_1K, min_area=0)
        if not size:
            size = _select_size(content_img.width, content_img.height, ASPECT_RATIO, min_area=0)
        if not size:
            return key, False, "no_valid_size"

    session = requests.Session()
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    request_timeout = (max(1, int(connect_timeout_s)), max(1, int(timeout_s)))
    download_timeout = (max(1, int(connect_timeout_s)), max(1, int(download_timeout_s)))
 
    def do_request(chosen_size: str):
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [
                _encode_image_data_url(cref_path, image_format),
                _encode_image_data_url(sref_path, image_format),
            ],
            "size": chosen_size,
            "response_format": "url",
            "sequential_image_generation": "auto",
            "sequential_image_generation_options": {"max_images": 1},
        }
        return _request_with_retries(
            session,
            "POST",
            base_url,
            attempts=request_retries,
            backoff_s=retry_backoff_s,
            timeout=request_timeout,
            stop_event=stop_event,
            label=f"id={key} stage=generate size={chosen_size}",
            headers=headers,
            json=payload,
        )

    try:
        resp = do_request(size)
        body, msg = _extract_error_body_and_msg(resp)
        if body is not None:
            min_area = _extract_min_area(msg)
            if (not resolution) and min_area:
                size2 = _select_size(content_img.width, content_img.height, ASPECT_RATIO, min_area=min_area)
                if size2 and size2 != size:
                    resp2 = do_request(size2)
                    body2, _ = _extract_error_body_and_msg(resp2)
                    if body2 is None:
                        resp = resp2
                        size = size2
                    else:
                        return key, False, f"http_{resp2.status_code}:{body2}"
                else:
                    return key, False, f"http_{resp.status_code}:{body}"
            else:
                return key, False, f"http_{resp.status_code}:{body}"

        data = resp.json()
        data_items = data.get("data") if isinstance(data, dict) else None
        if not data_items:
            return key, False, f"empty_data:{data}"
        url = data_items[0]["url"]
        host = urlparse(url).netloc or "unknown"
        try:
            img_resp = _request_with_retries(
                session,
                "GET",
                url,
                attempts=download_retries,
                backoff_s=retry_backoff_s,
                timeout=download_timeout,
                stop_event=stop_event,
                label=f"id={key} stage=download host={host}",
            )
        except requests.RequestException as e:
            return key, False, f"download_error:{type(e).__name__}:{e} host={host}"
        img_resp.raise_for_status()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_img = Image.open(BytesIO(img_resp.content)).convert("RGB")
        if save_resolution:
            save_w, save_h = _parse_size(save_resolution)
            out_img = out_img.resize((save_w, save_h), resample=getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS))
        out_img.save(out_path)
        return key, True, f"ok:{size}"
    except RuntimeError as e:
        return key, False, str(e)
    except requests.RequestException as e:
        return key, False, f"request_error:{type(e).__name__}:{e}"
    finally:
        session.close()
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cref_dir", required=True)
    ap.add_argument("--sref_dir", required=True)
    ap.add_argument("--prompts_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="doubao-seedream-4.5")
    ap.add_argument("--base_url", default="https://YOUR_OPENAI_COMPAT_ENDPOINT/v1/images/generations")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", type=str, default="")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--resolution",
        type=str,
        default="",
        help='Force size as "WIDTHxHEIGHT". If empty, auto-pick by aspect ratio.',
    )
    ap.add_argument(
        "--save_resolution",
        type=str,
        default="",
        help='Resize final saved image to "WIDTHxHEIGHT".',
    )
    ap.add_argument("--image_format", choices=["jpeg", "png"], default="jpeg")
    ap.add_argument("--timeout_s", type=int, default=600)
    ap.add_argument("--download_timeout_s", type=int, default=30)
    ap.add_argument("--connect_timeout_s", type=int, default=10)
    ap.add_argument("--request_retries", type=int, default=2)
    ap.add_argument("--download_retries", type=int, default=3)
    ap.add_argument("--retry_backoff_s", type=float, default=2.0)
    ap.add_argument("--progress_timeout_s", type=int, default=15)
    args = ap.parse_args()
 
    api_key = os.getenv("SEEDREAM_API_KEY", "")
    if not api_key:
        raise SystemExit("SEEDREAM_API_KEY is required")
 
    cref_dir = Path(args.cref_dir)
    sref_dir = Path(args.sref_dir)
    out_dir = Path(args.out_dir)
    with open(args.prompts_json, "r", encoding="utf-8") as f:
        prompts: Dict[str, str] = {str(k): str(v) for k, v in json.load(f).items()}
 
    cref_map = _list_images(cref_dir)
    sref_map = _list_images(sref_dir)
 
    keys = [k for k in prompts.keys() if k in cref_map and k in sref_map]
    keys.sort()
 
    if args.ids.strip():
        wanted = {x.strip() for x in args.ids.split(",") if x.strip()}
        keys = [k for k in keys if k in wanted]
 
    if args.limit and args.limit > 0:
        keys = keys[: args.limit]
 
    if not keys:
        raise SystemExit("no matched basenames between prompts_json, cref_dir and sref_dir")
 
    try:
        from tqdm import tqdm
 
        pbar = tqdm(total=len(keys), unit="img")
    except Exception:
        tqdm = None
        pbar = None
 
    ok = 0
    failed = 0
    skipped = 0
    total = len(keys)
    stop_event = threading.Event()
    task_queue: "queue.Queue[Optional[Tuple[str, str, Path, Path, Path]]]" = queue.Queue()
    result_queue: "queue.Queue[Tuple[str, bool, str]]" = queue.Queue()

    for k in keys:
        task_queue.put((k, prompts[k], cref_map[k], sref_map[k], out_dir / f"{k}.png"))
    num_workers = max(1, int(args.workers))
    for _ in range(num_workers):
        task_queue.put(None)

    worker_args = {
        "api_key": api_key,
        "base_url": args.base_url,
        "model": args.model,
        "resolution": args.resolution,
        "save_resolution": args.save_resolution,
        "image_format": args.image_format,
        "timeout_s": int(args.timeout_s),
        "download_timeout_s": int(args.download_timeout_s),
        "connect_timeout_s": int(args.connect_timeout_s),
        "request_retries": int(args.request_retries),
        "download_retries": int(args.download_retries),
        "retry_backoff_s": float(args.retry_backoff_s),
        "overwrite": bool(args.overwrite),
    }

    threads = []
    for idx in range(num_workers):
        t = threading.Thread(
            target=_worker,
            args=(task_queue, result_queue, stop_event, worker_args),
            name=f"seeddream-{idx}",
            daemon=True,
        )
        t.start()
        threads.append(t)

    completed = 0
    try:
        while completed < total:
            try:
                k, success, msg = result_queue.get(timeout=max(1, int(args.progress_timeout_s)))
            except queue.Empty:
                print(
                    f"[WAIT] done={completed}/{total} ok={ok} skipped={skipped} failed={failed} "
                    f"pending={total - completed}",
                    flush=True,
                )
                continue

            completed += 1
            if success:
                if msg == "skipped":
                    skipped += 1
                else:
                    ok += 1
            else:
                failed += 1
                print(f"[FAIL] id={k} {msg}", flush=True)
            if pbar is not None:
                pbar.update(1)
    except KeyboardInterrupt:
        stop_event.set()
        print(
            f"\n[INTERRUPT] stopping early: completed={completed}/{total} pending={total - completed}",
            flush=True,
        )
        raise SystemExit(130)
    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=0.2)
 
    if pbar is not None:
        pbar.close()
 
    print(f"[DONE] total={total} ok={ok} skipped={skipped} failed={failed} out_dir={out_dir}")
 
 
if __name__ == "__main__":
    main()
