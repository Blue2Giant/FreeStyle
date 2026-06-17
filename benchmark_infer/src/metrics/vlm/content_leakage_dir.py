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

CONTENT_LEAKAGE_5SCORE_PROMPT = r"""
你将看到两张图片：A 和 B。

任务：评估图片 B 对图片 A 的"内容泄漏程度"。
这里的"内容泄漏"指：B 是否复用了 A 的具体主体内容、角色身份、物体结构、场景布局、姿态动作、镜头构图、局部关键元素等，而不仅仅是借用了 A 的画风。

请注意区分：
* "允许借用"的是风格：如笔触、材质、色彩体系、渲染方式、线条习惯、整体审美气质。
* "不应算作纯风格"的是内容：如同一个角色/发型/服装组合、同一种具体物体外形、同一种场景布局、相同姿势、相同构图、相同背景元素组合。

请重点观察以下内容泄漏信号：
1) SUBJECT IDENTITY（主体身份）
   - 是否像是同一个人物、同一个角色、同一个动物、同一个具体物体
   - 人物时关注：性别呈现、发型、脸部特征、服装款式、配饰、体态
2) POSE / ACTION（姿态与动作）
   - 是否保留了相同或非常接近的姿势、手势、朝向、动作关系
3) COMPOSITION / VIEW（构图与镜头）
   - 主体位置、取景远近、镜头角度、裁切方式是否明显复用
4) SCENE / OBJECT SETUP（场景与物体组织）
   - 背景中的关键物体、空间布局、前后景关系、装饰元素组合是否明显相似
5) DISTINCTIVE DETAILS（独特细节）
   - 是否复用了少见而具体的细节组合，例如特殊发饰、武器、纹样、家具结构、招牌、背景小物件等

严格规则：
* 不要因为两张图"风格很像"就给高分。
* 只有当你看到"具体内容被带过去了"，才提高分数。
* 如果只是题材大类相同（例如都是女性肖像、都是房子、都是街景），但具体内容不同，分数应保持低。
* 如果 B 看起来几乎只是学习了 A 的风格，而主体、构图、场景都换了，应给低分。

评分标准（0-10，必须输出整数，评分要严格）：
* 0：完全看不出内容泄漏，只是风格可能相近，具体主体/构图/场景都不同。
* 1-2：只有很弱的内容重合，更多只是题材相近或有少量泛化元素相似。
* 3-4：存在一些可见的内容借用，但主要主体、构图或场景仍明显不同。
* 5-6：中等内容泄漏，多个关键内容元素相似，能看出明显参考了 A 的具体内容。
* 7-8：较强内容泄漏，核心主体、姿态、构图或场景组织已有大面积复用，仅有部分改动。
* 9-10：极强内容泄漏，B 与 A 在具体内容上接近直接复现或仅做轻微改写。

输出规则（非常重要）：
* score 必须是 0-10 的整数。
* reason 1-2 句，明确指出是哪些"内容元素"造成该分数，不要只说风格像或不像。
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
    content.append({"type": "text", "text": CONTENT_LEAKAGE_5SCORE_PROMPT})
    return [{"role": "user", "content": content}]


def run_content_leakage_score_onecall(
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
    for base, ref_path, output_path in tasks:
        try:
            _, overall, err = run_content_leakage_score_onecall(
                client=client,
                model=model,
                img_a=ref_path,
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
    parser = argparse.ArgumentParser(description="内容泄漏检测：双目录批量评分")
    parser.add_argument("--ref_dir", required=True, help="参考图 A 所在目录")
    parser.add_argument("--output_dir", required=True, help="生成图 B 所在目录")
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

    ref_files = set(smart_listdir(args.ref_dir))
    output_files = set(smart_listdir(args.output_dir))
    common_files = list(ref_files & output_files)
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
        ref_path = args.ref_dir.rstrip("/") + "/" + name
        output_path = args.output_dir.rstrip("/") + "/" + name
        tasks.append((base, ref_path, output_path))

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
