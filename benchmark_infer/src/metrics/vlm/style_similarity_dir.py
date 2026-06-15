#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
import base64
import mimetypes
import multiprocessing as mp
from typing import Optional, Dict, Any, Tuple, List

from tqdm import tqdm
from openai import OpenAI
from megfile.smart import (
    smart_open as mopen,
    smart_exists,
    smart_listdir,
    smart_makedirs,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

STYLE_5SCORE_PROMPT = r"""
你将看到两张图片：A 和 B。请只从“画风/风格”角度评估它们有多一致。
不要根据主体内容是否相同来判断（允许人物/物体/场景完全不同）。

你需要综合所有纬度的基础上总体的相似度给出 0 到 10 的评分。

维度定义（请严格按以下含义判断）：
1) BRUSHSTROKE（笔触/线条/边缘处理）
   - 线条粗细、线条是否抖动、勾边方式、笔刷痕迹是否明显
   - 边缘是硬边/软边、是否有描边、涂抹/泼洒/铅笔/钢笔等笔触特征
2) TEXTURE（纹理/材质表现/颗粒与噪声/画布纸张质感）
   - 表面微观颗粒感：例如“砂纸颗粒/胶片噪声/水彩纸纤维/油画布纹”
   - 材质的微细结构呈现方式：例如“涂料堆叠纹/喷点/网点/浮雕质感”
   - 注意：TEXTURE 不是配色，也不是形状轮廓
3) COLOR（配色体系与色彩分布/色温/饱和度/对比度）
   - 主色调与色温倾向（偏冷/偏暖）、饱和度高低、明暗对比强弱
   - 色彩分布相似：比如“主要颜色的组成比例/覆盖面积是否相近（大面积背景色、主色块比例）”
   - 注意：COLOR 不是纹理颗粒，也不是笔触线条
4) SHAPE（形状语言/造型习惯）
   - 几何化/写实化程度、比例是否夸张、轮廓更锐利还是更圆润
   - 结构简化方式（例如卡通扁平、Q版、极简几何、写实结构）
5) PATTERN（模式/母题/装饰性图案化规律）
   - 是否有重复纹样、装饰元素、固定符号化母题（如固定花纹、反复出现的装饰线、图案化背景）
   - 图案密度、重复规律、装饰性元素的组织方式是否相似

评分标准（0-10，评分要严格）：
* 0：风格完全不一致，基本无法认为在该维度相同。
* 1–3：大部分不一致，只有很弱/偶然的相似。
* 4–6：有部分相似，但存在明显缺失或关键差异。
* 7–9：大体一致，仅有少量小问题。
* 10：完全一致，关键特征高度匹配。

重要说明（必须遵守）：
* 评分要严格，除非该维度清晰且准确匹配，否则不要给高分。
* 只评估“风格维度”是否一致；不要考虑人物身份是否相同、物体是否同一个、背景/场景是否一致。
* 不要评估审美好坏或画面好看与否。

输出规则（非常重要）：
* score 必须是 0-10 的整数。
* reason 1-2 句，指出可观察到的具体依据。
* 严格输出格式为 score@reason（只输出这一行，不要输出其它任何字符/标点/换行）
""".strip()


def path_to_data_url(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/png"
    with mopen(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def strip_code_fences(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return s
    m = re.match(r"^\s*```(?:json)?\s*([\s\S]*?)\s*```\s*$", s, flags=re.IGNORECASE)
    if m:
        return (m.group(1) or "").strip()
    return s


def _clamp_score_0_10(score: int) -> int:
    if score < 0:
        return 0
    if score > 10:
        return 10
    return score


def parse_overall_score_reason(raw_text: str) -> Dict[str, Any]:
    clean = strip_code_fences(raw_text).strip()
    if not clean:
        raise ValueError("模型输出为空")
    first_line = clean.splitlines()[0].strip()
    m = re.match(r"^\s*(\d{1,2})\s*@\s*(.+?)\s*$", first_line)
    if m:
        score = _clamp_score_0_10(int(m.group(1)))
        reason = m.group(2).strip()
        if not reason:
            raise ValueError("reason 为空")
        return {"score": score, "reason": reason}
    try:
        obj = json.loads(clean)
    except Exception:
        raise ValueError(f"无法解析 score@reason，且不是 JSON：{first_line!r}")
    if not isinstance(obj, dict):
        raise ValueError("模型 JSON 输出不是 object")
    overall = obj.get("OVERALL", None)
    if not isinstance(overall, dict):
        raise ValueError("模型 JSON 缺少 OVERALL object")
    score_raw = overall.get("score", None)
    reason_raw = overall.get("reason", "")
    if isinstance(score_raw, bool) or score_raw is None:
        raise ValueError("OVERALL.score 无效")
    score = int(float(score_raw)) if isinstance(score_raw, str) else int(score_raw)
    score = _clamp_score_0_10(score)
    reason = reason_raw if isinstance(reason_raw, str) else str(reason_raw)
    reason = reason.strip()
    if not reason:
        raise ValueError("OVERALL.reason 为空")
    return {"score": score, "reason": reason}


def build_messages(img_a: str, img_b: str):
    content = []
    content.append({"type": "text", "text": "Image A:"})
    content.append({"type": "image_url", "image_url": {"url": path_to_data_url(img_a)}})
    content.append({"type": "text", "text": "Image B:"})
    content.append({"type": "image_url", "image_url": {"url": path_to_data_url(img_b)}})
    content.append({"type": "text", "text": STYLE_5SCORE_PROMPT})
    return [{"role": "user", "content": content}]


def run_style_score_onecall(
    client: OpenAI,
    model: str,
    img_a: str,
    img_b: str,
    max_tokens: int = 512,
) -> Tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    messages = build_messages(img_a, img_b)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    raw = (resp.choices[0].message.content or "").strip()
    try:
        overall = parse_overall_score_reason(raw)
    except Exception as e:
        return raw, None, {"error": str(e)}
    return raw, overall, None


def is_image_name(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in IMG_EXTS


def _worker_process(
    model: str,
    base_url: str,
    api_key: str,
    timeout: int,
    max_tokens: int,
    tasks: List[Tuple[str, str, str]],
    result_queue: mp.Queue,
):
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    for base, style_path, output_path in tasks:
        try:
            _, overall, err = run_style_score_onecall(
                client=client,
                model=model,
                img_a=style_path,
                img_b=output_path,
                max_tokens=max_tokens,
            )
            if overall is None:
                result_queue.put((base, None, None))
            else:
                result_queue.put((base, overall["score"], overall["reason"]))
        except Exception:
            result_queue.put((base, None, None))


def smart_write_json(path: str, obj: Any):
    data = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
    dir_path = os.path.dirname(path) or "."
    if path.startswith(("s3://", "oss://")):
        smart_makedirs(dir_path, exist_ok=True)
        with mopen(path, "wb") as f:
            f.write(data)
    else:
        os.makedirs(dir_path, exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)


def main():
    parser = argparse.ArgumentParser(description="画风相似度：双目录批量评分")
    parser.add_argument("--style_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--out_score_json", required=True)
    parser.add_argument("--out_reason_json", required=True)
    parser.add_argument("--base_url", required=True)
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--model", default="Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--num_procs", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    style_files = set(smart_listdir(args.style_dir))
    output_files = set(smart_listdir(args.output_dir))
    common_files = list(style_files & output_files)
    common_files = [f for f in common_files if is_image_name(f)]

    if args.num_samples > 0 and len(common_files) > args.num_samples:
        import random
        random.seed(args.seed)
        common_files = random.sample(common_files, args.num_samples)

    score_results = {}
    reason_results = {}
    if (not args.overwrite) and smart_exists(args.out_score_json) and smart_exists(args.out_reason_json):
        try:
            existing_score = None
            existing_reason = None
            if args.out_score_json.startswith(("s3://", "oss://")):
                with mopen(args.out_score_json, "r", encoding="utf-8") as f:
                    existing_score = json.load(f)
            else:
                with open(args.out_score_json, "r", encoding="utf-8") as f:
                    existing_score = json.load(f)
            if args.out_reason_json.startswith(("s3://", "oss://")):
                with mopen(args.out_reason_json, "r", encoding="utf-8") as f:
                    existing_reason = json.load(f)
            else:
                with open(args.out_reason_json, "r", encoding="utf-8") as f:
                    existing_reason = json.load(f)
            if isinstance(existing_score, dict) and isinstance(existing_reason, dict):
                score_results = existing_score
                reason_results = existing_reason
                processed_keys = set(existing_score.keys()) & set(existing_reason.keys())
                common_files = [f for f in common_files if os.path.splitext(f)[0] not in processed_keys]
        except Exception:
            pass

    if not common_files:
        smart_write_json(args.out_score_json, score_results)
        smart_write_json(args.out_reason_json, reason_results)
        return

    tasks = []
    for name in common_files:
        base = os.path.splitext(name)[0]
        style_path = args.style_dir.rstrip("/") + "/" + name
        output_path = args.output_dir.rstrip("/") + "/" + name
        tasks.append((base, style_path, output_path))

    num_procs = max(1, int(args.num_procs))
    chunk_size = (len(tasks) + num_procs - 1) // num_procs
    result_queue = mp.Queue()
    workers = []

    for i in range(num_procs):
        sub_tasks = tasks[i * chunk_size : (i + 1) * chunk_size]
        if not sub_tasks:
            continue
        p = mp.Process(
            target=_worker_process,
            args=(
                args.model,
                args.base_url,
                args.api_key,
                args.timeout,
                args.max_tokens,
                sub_tasks,
                result_queue,
            ),
        )
        p.start()
        workers.append(p)

    total_done = 0
    total_tasks = len(tasks)
    with tqdm(total=total_tasks, desc="Processing") as pbar:
        while total_done < total_tasks:
            try:
                base, score, reason = result_queue.get(timeout=5)
                score_results[base] = score
                reason_results[base] = reason
                total_done += 1
                pbar.update(1)
                if total_done % 50 == 0:
                    smart_write_json(args.out_score_json, score_results)
                    smart_write_json(args.out_reason_json, reason_results)
            except Exception:
                if not any(p.is_alive() for p in workers) and result_queue.empty():
                    break

    for p in workers:
        p.join()

    smart_write_json(args.out_score_json, score_results)
    smart_write_json(args.out_reason_json, reason_results)


if __name__ == "__main__":
    main()
