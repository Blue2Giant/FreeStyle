import argparse
import asyncio
import concurrent.futures
import json
import os
import re
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import illustrious_one_lora_diverse as base
from comfykit import ComfyKit
from tqdm import tqdm


def _parse_pair_model_ids(path: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with base.mopen(path, "r", encoding="utf-8") as f:
        for line in f:
            s = (line or "").strip()
            if not s or s.startswith("#"):
                continue
            m = re.match(r"(\d+)\s*__\s*(\d+)", s)
            if not m:
                continue
            pairs.append((m.group(1), m.group(2)))
    seen = set()
    out: List[Tuple[str, str]] = []
    for c, st in pairs:
        key = f"{c}__{st}"
        if key in seen:
            continue
        seen.add(key)
        out.append((c, st))
    return out


def _build_latest_lora_map(lora_root: str) -> Dict[str, str]:
    by_id: Dict[str, List[str]] = {}
    for name in base.smart_listdir(lora_root):
        n = str(name)
        if not n.lower().endswith(".safetensors"):
            continue
        stem = os.path.splitext(os.path.basename(n))[0]
        m = re.match(r"(\d+)", stem)
        if not m:
            continue
        mid = m.group(1)
        by_id.setdefault(mid, []).append(n)
    out: Dict[str, str] = {}
    for mid, names in by_id.items():
        names.sort(reverse=True)
        out[mid] = base.join_path(lora_root, names[0])
    return out


def _compose_prompt_dual(
    prompt: str,
    content_trigger: str,
    style_trigger: str,
    prefix_phrase: str,
    allow_empty_prompt_body: bool = False,
) -> str:
    body = (prompt or "").strip()
    heads: List[str] = []
    ct = (content_trigger or "").strip()
    st = (style_trigger or "").strip()
    pp = (prefix_phrase or "").strip()
    if ct and (not body or not base._contains_phrase(body, ct)):
        heads.append(ct)
    if st and (not body or not base._contains_phrase(body, st)):
        heads.append(st)
    if pp and (not body or not base._contains_phrase(body, pp)):
        heads.append(pp)
    if not body:
        if allow_empty_prompt_body:
            return ", ".join(heads)
        return ""
    if not heads:
        return body
    return f"{', '.join(heads)}, {body}"


def _attach_dual_triggers(
    prompts: List[str],
    content_trigger: str,
    style_trigger: str,
    prefix_phrase: str,
    allow_empty_prompt_body: bool = False,
) -> List[str]:
    out: List[str] = []
    for p in prompts:
        q = _compose_prompt_dual(
            p,
            content_trigger,
            style_trigger,
            prefix_phrase,
            allow_empty_prompt_body=allow_empty_prompt_body,
        )
        if q:
            out.append(q)
    return out


def _inject_dual_workflow_payload(
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
    wf["1"]["inputs"]["ckpt_name"] = str(model_name)
    wf["5"]["inputs"]["seed"] = int(seed)
    wf["18"]["inputs"]["text"] = str(positive_prompt)
    if negative_prompt: 
        wf["19"]["inputs"]["text"] = str(negative_prompt) 
    wf["15"]["inputs"]["lora_name"] = str(content_lora_name)
    wf["15"]["inputs"]["strength_model"] = float(strength_model)
    wf["15"]["inputs"]["strength_clip"] = float(strength_clip)
    wf["17"]["inputs"]["lora_name"] = str(style_lora_name)
    wf["17"]["inputs"]["strength_model"] = float(strength_model)
    wf["17"]["inputs"]["strength_clip"] = float(strength_clip)
    if "25" in wf and isinstance(wf["25"], dict):
        inputs = wf["25"].get("inputs", {})
        if isinstance(inputs, dict) and "filename_prefix" in inputs:
            inputs["filename_prefix"] = str(filename_prefix)


def _split_dual_lora_name(name: str) -> Tuple[str, str]:
    if "||" in name:
        p = name.split("||", 1)
        return p[0], p[1]
    return name, name


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
    content_lora_name, style_lora_name = _split_dual_lora_name(lora_name)
    base.safe_makedirs(save_dir)
    with base.mopen(wf_json, "r", encoding="utf-8") as f:
        wf_template = json.load(f)
    kit = ComfyKit(comfyui_url=url)
    negative_prompt = getattr(base, "NEGATIVE_PROMPT", "")

    for sel_idx, pos_prompt in prompts_to_run:
        wf = base._deepcopy(wf_template)
        if gen_seed is None:
            new_seed = int.from_bytes(os.urandom(4), byteorder="big", signed=False)
        else:
            new_seed = int(gen_seed) + int(sel_idx)
        raw_prefix = f"{content_lora_name}_{style_lora_name}_{sel_idx:05d}_{uuid.uuid4().hex}"
        filename_prefix = re.sub(r"[^0-9A-Za-z_\\-]+", "_", raw_prefix)
        _inject_dual_workflow_payload(
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
                    "mode": "custom_workflow_dual_lora",
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


def process_one_pair(
    content_model_id: str,
    style_model_id: str,
    content_lora_path: str,
    style_lora_path: str,
    comfy_url: str,
    base_model: str,
    wf_json: str,
    meta_root: str,
    meta_index: Dict[str, List[str]],
    output_root: str,
    local_tmp_root: str,
    base_prompt_txt_local: str,
    num_prompts_to_gen: int,
    prompt_seed: int,
    gen_seed: Optional[int],
    overwrite: bool,
    strength_model: float,
    strength_clip: float,
    output_subdir: Optional[str],
    exec_timeout: Optional[float],
    download_retry_rounds: int,
    download_retry_wait: float,
    prefix_phrase: str,
    negative_prompt: str,
    download_workers: int,
    allow_empty_prompt_body: bool,
):
    pair_id = f"{content_model_id}__{style_model_id}"
    model_output_dir = base.join_path(output_root, pair_id)
    if output_subdir:
        model_output_dir = base.join_path(model_output_dir, output_subdir)
    eval_dir = base.join_path(model_output_dir, "eval_images_with_negative_new")

    content_trigger = (base._read_trigger_word_from_model_meta(content_model_id, meta_root) or "").strip()
    style_trigger = (base._read_trigger_word_from_model_meta(style_model_id, meta_root) or "").strip()
    if not content_trigger:
        content_trigger, _ = base.parse_trigger_from_meta(content_model_id, meta_root, meta_index)
        content_trigger = (content_trigger or "").strip()
    if not style_trigger:
        style_trigger, _ = base.parse_trigger_from_meta(style_model_id, meta_root, meta_index)
        style_trigger = (style_trigger or "").strip()

    base.safe_makedirs(eval_dir)
    all_prompts = []
    if base_prompt_txt_local and os.path.isfile(base_prompt_txt_local):
        all_prompts = base.load_prompts_from_txt(base_prompt_txt_local)
    if all_prompts:
        selected_base_prompts = base.select_diverse_prompts_for_model(
            model_id=pair_id,
            all_prompts=all_prompts,
            num_prompts=num_prompts_to_gen,
            prompt_seed=prompt_seed,
            trigger_word=f"{content_trigger} {style_trigger}".strip(),
            prefix_phrase=prefix_phrase,
            eval_dir=eval_dir,
            overwrite=overwrite,
        )
    elif allow_empty_prompt_body:
        selected_base_prompts = [""] * max(1, int(num_prompts_to_gen))
    else:
        raise RuntimeError("prompt 为空且未开启 --allow-empty-prompt-body")
    selected_prompts = _attach_dual_triggers(
        selected_base_prompts,
        content_trigger,
        style_trigger,
        prefix_phrase,
        allow_empty_prompt_body=allow_empty_prompt_body,
    )
    prompt_record_path = base.join_path(model_output_dir, "selected_prompts_final.json")
    base._write_json(
        prompt_record_path,
        {
            "pair_id": pair_id,
            "content_model_id": content_model_id,
            "style_model_id": style_model_id,
            "content_trigger": content_trigger,
            "style_trigger": style_trigger,
            "prefix_phrase": prefix_phrase,
            "selected_base_prompts": selected_base_prompts,
            "selected_prompts": selected_prompts,
        },
    )
    N = len(selected_prompts)
    done_indices = set(base.scan_done_prompt_indices(eval_dir))
    if overwrite:
        indices_to_run = list(range(N))
    else:
        if len(done_indices) >= N:
            print(f"[SKIP] {pair_id}: 已完成 {len(done_indices)}/{N}，跳过")
            return
        indices_to_run = [i for i in range(N) if i not in done_indices]
    prompts_to_run = [(i, selected_prompts[i]) for i in indices_to_run]
    if not prompts_to_run:
        return

    base.NEGATIVE_PROMPT = negative_prompt or ""
    dual_lora_name = f"{os.path.basename(content_lora_path)}||{os.path.basename(style_lora_path)}"
    asyncio.run(
        base._run_with_download_queue(
            wf_json=wf_json,
            base_model=base_model,
            lora_name=dual_lora_name,
            prompts_to_run=prompts_to_run,
            selected_prompts=selected_prompts,
            indices_to_run=indices_to_run,
            eval_dir=eval_dir,
            strength_model=strength_model,
            strength_clip=strength_clip,
            comfy_url=comfy_url,
            gen_seed=gen_seed,
            exec_timeout=exec_timeout,
            download_retry_rounds=download_retry_rounds,
            download_retry_wait=download_retry_wait,
            download_workers=download_workers,
        )
    )
    if not base.is_remote_path(eval_dir):
        base.ensure_only_images(eval_dir)


def main():
    base.run_comfy_batch_workflow = run_comfy_batch_workflow
    parser = argparse.ArgumentParser(description="双 LoRA 批量生成 eval_images")
    parser.add_argument("--lora-root", required=True)
    parser.add_argument("--meta-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--pair-model-id-txt", required=True)
    parser.add_argument("--prompt-txt", default="/data/LoraPipeline/assets/diverse_prompts_100.txt")
    parser.add_argument("--output-subdir", default=None)
    parser.add_argument("--prefix-phrase", default="")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--workflow-json", default="/data/benchmark_metrics/lora_pipeline/meta/workflows/sdxl_dual_lora_ljh.json")
    parser.add_argument("--strength-model", type=float, default=1.0)
    parser.add_argument("--strength-clip", type=float, default=1.0)
    parser.add_argument("--num-prompts", type=int, required=True)
    parser.add_argument("--prompt-seed", type=int, default=42)
    parser.add_argument("--gen-seed", type=int, default=None)
    parser.add_argument("--comfy-host", action="append", default=[])
    parser.add_argument("--base-port", type=int, default=8188)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--local-tmp-root", default="/tmp/lora_comfy_cache")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--exec-timeout", type=float, default=0)
    parser.add_argument("--download-retry-rounds", type=int, default=2)
    parser.add_argument("--download-retry-wait", type=float, default=2.0)
    parser.add_argument("--download-workers", type=int, default=1)
    parser.add_argument("--allow-empty-prompt-body", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.local_tmp_root, exist_ok=True)
    exec_timeout: Optional[float] = args.exec_timeout
    if exec_timeout is not None and exec_timeout <= 0:
        exec_timeout = None

    raw_hosts = args.comfy_host if isinstance(args.comfy_host, list) else [args.comfy_host]
    comfy_hosts: List[str] = []
    for h in raw_hosts:
        if not h:
            continue
        for part in h.split(","):
            part = part.strip()
            if part:
                comfy_hosts.append(part.rstrip("/"))
    comfy_hosts = list(dict.fromkeys(comfy_hosts)) or ["http://127.0.0.1"]
    endpoints: List[str] = []
    for host in comfy_hosts:
        for i in range(max(1, args.num_workers)):
            endpoints.append(f"{host}:{args.base_port + i}")

    pair_ids = _parse_pair_model_ids(args.pair_model_id_txt)
    if not pair_ids:
        print("[INFO] 没有读取到任何 pair model id，退出。")
        return

    base_prompt_txt_local = base.prepare_prompt_txt(args.prompt_txt, args.local_tmp_root)
    meta_index = base.build_meta_index(args.meta_root.rstrip("/"))
    lora_map = _build_latest_lora_map(args.lora_root.rstrip("/"))

    pair_tasks: List[Tuple[Tuple[str, str], str, str, str]] = []
    for idx, (cid, sid) in enumerate(pair_ids):
        cp = lora_map.get(cid)
        sp = lora_map.get(sid)
        if not cp or not sp:
            continue
        ep = endpoints[idx % len(endpoints)]
        pair_tasks.append(((cid, sid), cp, sp, ep))
    if not pair_tasks:
        print("[INFO] 没有可执行的 pair（LoRA 文件缺失），退出。")
        return

    workers = min(len(pair_tasks), len(endpoints))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_one_pair,
                pair[0][0],
                pair[0][1],
                pair[1],
                pair[2],
                pair[3],
                args.base_model,
                args.workflow_json,
                args.meta_root.rstrip("/"),
                meta_index,
                args.output_root.rstrip("/"),
                args.local_tmp_root,
                base_prompt_txt_local,
                args.num_prompts,
                args.prompt_seed,
                args.gen_seed,
                args.overwrite,
                args.strength_model,
                args.strength_clip,
                args.output_subdir,
                exec_timeout,
                args.download_retry_rounds,
                args.download_retry_wait,
                args.prefix_phrase,
                args.negative_prompt,
                args.download_workers,
                args.allow_empty_prompt_body,
            ): f"{pair[0][0]}__{pair[0][1]}"
            for pair in pair_tasks
        }
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            key = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"{key} generated an exception: {exc}")


if __name__ == "__main__":
    main()
