import asyncio
import json
import os
import re
import sys
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import dual_lora_illustrious as dual_base
import illustrious_one_lora_diverse as base
from comfykit import ComfyKit


def _inject_qwen_dual_workflow_payload(
    wf: Dict[str, Any],
    *,
    seed: int,
    model_name: str,
    content_lora_name: str,
    style_lora_name: str,
    strength_model: float,
    strength_clip: float,
    positive_prompt: str,
    negative_prompt: str,
    filename_prefix: str,
):
    wf["3"]["inputs"]["seed"] = int(seed)
    wf["6"]["inputs"]["text"] = str(positive_prompt)
    wf["37"]["inputs"]["unet_name"] = str(model_name)
    wf["60"]["inputs"]["filename_prefix"] = str(filename_prefix)
    wf["79"]["inputs"]["lora_name"] = str(content_lora_name)
    wf["79"]["inputs"]["strength_model"] = float(strength_model)
    wf["79"]["inputs"]["strength_clip"] = float(strength_clip)
    wf["81"]["inputs"]["lora_name"] = str(style_lora_name)
    wf["81"]["inputs"]["strength_model"] = float(strength_model)
    wf["81"]["inputs"]["strength_clip"] = float(strength_clip)
    if negative_prompt:
        wf["7"]["inputs"]["text"] = str(negative_prompt)


async def run_comfy_batch_workflow(
    *,
    wf_json: str,
    base_model_name: str,
    lora_name: str,
    prompts_to_run: List[Tuple[int, str]],
    save_dir: str,
    strength_model: float,
    strength_clip: float,
    url: str,
    gen_seed: Optional[int] = None,
    save_prompt_json: bool = True,
    exec_timeout: Optional[float] = None,
    download_queue: Optional["asyncio.Queue[Optional[Dict[str, Any]]]"] = None,
):
    content_lora_name, style_lora_name = dual_base._split_dual_lora_name(lora_name)
    base.safe_makedirs(save_dir)
    with base.mopen(wf_json, "r", encoding="utf-8") as f:
        wf_template = json.load(f)
    negative_prompt = getattr(base, "NEGATIVE_PROMPT", "")

    async with ComfyKit(comfyui_url=url) as kit:
        for sel_idx, pos_prompt in prompts_to_run:
            wf = base._deepcopy(wf_template)
            if gen_seed is None:
                new_seed = int.from_bytes(os.urandom(4), byteorder="big", signed=False)
            else:
                new_seed = int(gen_seed) + int(sel_idx)
            raw_prefix = f"{content_lora_name}_{style_lora_name}_{sel_idx:05d}_{uuid.uuid4().hex}"
            filename_prefix = re.sub(r"[^0-9A-Za-z_\\-]+", "_", raw_prefix)
            _inject_qwen_dual_workflow_payload(
                wf,
                seed=int(new_seed),
                model_name=base_model_name,
                content_lora_name=content_lora_name,
                style_lora_name=style_lora_name,
                strength_model=float(strength_model),
                strength_clip=float(strength_clip),
                positive_prompt=pos_prompt,
                negative_prompt=negative_prompt,
                filename_prefix=filename_prefix,
            )

            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tf:
                json.dump(wf, tf, ensure_ascii=False, indent=2)
                temp_json = tf.name
            try:
                if exec_timeout and float(exec_timeout) > 0:
                    result = await asyncio.wait_for(kit.execute(temp_json), timeout=float(exec_timeout))
                else:
                    result = await kit.execute(temp_json)
            except asyncio.TimeoutError:
                continue
            finally:
                try:
                    os.remove(temp_json)
                except OSError:
                    pass

            images_all = getattr(result, "images", None) or []
            images: List[str] = []
            mismatch_log = os.path.join(save_dir, "mismatch_log.txt")
            for item in images_all:
                if isinstance(item, str):
                    url_i = item
                elif isinstance(item, dict):
                    url_i = item.get("url") or item.get("image_url") or item.get("filename") or ""
                    if not url_i:
                        continue
                else:
                    url_i = str(item)
                fname = base._get_filename_from_url(url_i)
                if fname.startswith(filename_prefix):
                    images.append(url_i)
                else:
                    base._append_text_line(mismatch_log, f"sel_idx={sel_idx} prefix={filename_prefix} filename={fname} url={url_i}")
            if not images:
                base._append_text_line(mismatch_log, f"sel_idx={sel_idx} prefix={filename_prefix} no_valid_image, images_all_len={len(images_all)}")
                continue
            images = list(dict.fromkeys([u for u in images if u]))[:1]
            for img_idx, img_url in enumerate(images):
                prefix = f"{sel_idx:05d}_{img_idx}"
                img_path = base.join_path(save_dir, prefix + ".png")
                meta_path = base.join_path(save_dir, prefix + ".json")
                img_url = img_url.strip() if isinstance(img_url, str) else str(img_url)
                meta = None
                if save_prompt_json:
                    meta = {
                        "base_model": base_model_name,
                        "content_lora_name": content_lora_name,
                        "style_lora_name": style_lora_name,
                        "strength_model": float(strength_model),
                        "strength_clip": float(strength_clip),
                        "positive_prompt": pos_prompt,
                        "negative_prompt": negative_prompt,
                        "seed": int(new_seed),
                        "workflow": wf_json,
                        "mode": "custom_workflow_dual_lora_qwen",
                        "sel_idx": int(sel_idx),
                        "endpoint": url,
                    }
                if download_queue is not None:
                    await download_queue.put({"img_url": img_url, "img_path": img_path, "meta_path": meta_path, "meta": meta, "sel_idx": int(sel_idx)})
                else:
                    success = False
                    for attempt in range(3):
                        try:
                            req = Request(img_url, headers={"User-Agent": "Mozilla/5.0", "Connection": "close"})
                            with urlopen(req, timeout=300) as resp:
                                if getattr(resp, "status", 200) != 200:
                                    if attempt < 2:
                                        time.sleep(2 * (2 ** attempt))
                                    continue
                                content = resp.read()
                            base._write_binary(img_path, content)
                            if isinstance(meta, dict):
                                base._write_json(meta_path, meta)
                            success = True
                            break
                        except (HTTPError, URLError):
                            if attempt < 2:
                                time.sleep(2 * (2 ** attempt))
                        except Exception:
                            break
                    if not success:
                        continue


def main():
    default_wf = "/data/benchmark_metrics/lora_pipeline/meta/workflows/qwen_dual_lora.json"
    if "--workflow-json" not in sys.argv:
        sys.argv.extend(["--workflow-json", default_wf])
    dual_base.run_comfy_batch_workflow = run_comfy_batch_workflow
    dual_base.main()


if __name__ == "__main__":
    main()
