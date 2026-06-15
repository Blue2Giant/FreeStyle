#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import base64
import argparse
import math
import multiprocessing as mp
from io import BytesIO
from typing import Dict, Optional, List, Tuple, Any

import requests
from PIL import Image
from tqdm import tqdm

from megfile.smart import (
    smart_listdir,
    smart_exists,
    smart_makedirs,
    smart_open as mopen,
)

API_KEY = "EMPTY"
MODEL = "Qwen3-VL-30B-A3B-Instruct"
BASE_URL = "http://10.201.19.61:22002/v1"
TIMEOUT = 720
RETRY_EXHAUSTED_REASON = "API 重试耗尽"

G_ARGS = None

RESIZE_MAX_SIDE = 1024
JPEG_QUALITY = 85
Image.MAX_IMAGE_PIXELS = None

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

CONTENT_SYSTEM_PROMPT = (
    "你是一个只关注“主体内容和主题”的严格评审。\n"
    "你必须完全忽略画风、渲染风格、分辨率、滤镜等视觉风格差异，\n"
    "只关心画面中出现的具体人物/物体/场景是否相同或高度一致。\n"
    "你的任务：判断两张图片在“主体内容/主题”上是否高度一致。\n"
    "你必须只输出一个字符：0 或 1。\n"
    "1 表示主体内容高度一致；0 表示主体内容不一致。\n"
    "不要输出任何多余文字、空格、换行、JSON。"
)

CONTENT_USER_INSTRUCTION = (
    "请只从“主体内容和主题”的角度比较图片 A 和图片 B，\n"
    "严格忽略画风、线条风格、色彩风格、渲染方式等一切视觉风格差异。\n\n"
    "你需要重点考虑：画面里“是什么”和“在做什么”，而不是“怎么画出来的”。\n\n"
    "具体判定规则如下：\n"
    "1. 如果是【人物】为主体：\n"
    "   - 关注人物是否为“同一个角色”或“极为相似的角色”，\n"
    "   - 包括性别、年龄段、身材、发型、头发颜色、肤色、服装类型、服装主色调、主要配饰等是否相近，\n"
    "   - 姿势、朝向、镜头视角可以有一定变化，但如果感觉明显是不同的人物或完全不同造型，则视为不一致。\n"
    "2. 如果是【单一物体】为主体（例如一辆车、一栋房子、一把椅子等）：\n"
    "   - 重点看物体类别和形状结构是否一致（例如都是跑车、都是 SUV、都是圆桌等），\n"
    "   - 允许颜色不同，例如黄色的车和红色的车，只要车型和外形高度相似，就可以视为一致，\n"
    "   - 如果只是都包含“车/房子/杯子”但明显是不同种类或完全不同结构，则视为不一致。\n"
    "3. 如果是【复杂场景】（例如街景、室内布景、多人物场景等）：\n"
    "   - 关注场景的类型、主要元素组合和布局是否相似，\n"
    "   - 例如：都为“一个人物站在城市夜景街道中央，背景有高楼和霓虹招牌”，可以视为一致，\n"
    "   - 如果只是都在室外/室内，但核心构图和主体物体完全不同，则视为不一致。\n"
    "4. 画风完全无关：\n"
    "   - 即使一张是写实照片，另一张是二次元插画或卡通，只要主体内容和主题高度一致，也要判为 1，\n"
    "   - 禁止因为画风差异而判为 0。\n\n"
    "综合以上规则：\n"
    "当你认为两张图展示的是“同一个人物/同一类具体物体/同一种具体场景和主题”，则输出 1；\n"
    "如果只是大概类别相似（例如都有人物/都有车），但主体明显不是同一个，就输出 0。\n\n"
    "最终只输出一个字符：0 或 1。"
)

STYLE_SYSTEM_PROMPT = (
    "你是一个只关注“画风/视觉风格”的资深评审。\n"
    "你只评估视觉表现形式（媒介感、材质感、线条/笔触、色彩与调色、光影与对比、渲染/后期、画面噪声与颗粒、细节表达方式）。\n"
    "你必须忽略：人物/物体身份、动作含义、故事语义、场景类别、构图内容是否相似。\n"
    "\n"
    "判定目标：两张图是否属于同一种稳定画风/同一风格族。\n"
    "允许以下差异仍判为风格一致：\n"
    "- 内容/主体/场景不同\n"
    "- 构图与视角不同\n"
    "- 色相轻微变化、亮度对比变化、局部调色差异\n"
    "- 细节密度不同、裁剪/分辨率不同、轻微压缩/噪声\n"
    "\n"
    "只有当出现“风格机制”层面的明显变化才判不一致，例如：\n"
    "- 真实摄影 vs 插画/渲染\n"
    "- 线稿/勾线体系变化（有线稿→无，粗线→细线，漫画勾线→水彩边缘）\n"
    "- 材质与纹理生成方式变化（油画厚涂→平涂赛璐璐→3D塑料感→像素/点描等）\n"
    "- 光影模型变化（硬边影视布光→柔和漫反射插画光→霓虹强对比等）\n"
    "- 调色与色彩策略变化（低饱和复古→高饱和糖果色→黑白素描等）\n"
    "\n"
    "输出规则：你只能输出一个字符：0 或 1。\n"
    "1 = 画风高度一致（同一风格族，核心机制一致）；0 = 画风不一致。\n"
    "不要输出任何多余文字、空格、换行或标点。"
)

STYLE_USER_INSTRUCTION = (
    "请仅从“画风 / 视觉风格”角度比较图片A与图片B，忽略人物/物体身份、动作含义、故事语义与场景类别。\n"
    "\n"
    "请综合以下维度做判断，并采用“宽松一致性”标准：只要核心风格机制一致，即使主体、构图、视角、细节密度不同，也可以判为一致。\n"
    "重点维度（更高权重）：\n"
    "1) 媒介与渲染方式：摄影/3D/插画/水彩/油画/厚涂/赛璐璐/像素/素描 等\n"
    "2) 笔触与线条体系：是否有线稿、线条粗细/抖动、边缘处理、笔触颗粒\n"
    "3) 材质与纹理生成方式：表面质感、噪声/颗粒、细节组织方式\n"
    "4) 光影模型与对比：硬/软阴影、体积光、漫反射/镜面、高反差与否\n"
    "5) 色彩策略：饱和度、色相偏好、综合色调、调色风格（复古/冷暖/霓虹等）\n"
    "次要维度（允许变化）：\n"
    "6) 构图与视角：机位、镜头感、取景范围不同不应直接判为不一致\n"
    "\n"
    "判定：\n"
    "- 若多数“重点维度”一致，输出 1。\n"
    "- 只要出现明显的风格机制改变（如摄影↔插画、线稿体系突变、材质/渲染范式突变、整体调色策略完全不同），输出 0。\n"
    "\n"
    "最终只输出一个字符：0 或 1。"
)


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def is_image_name(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in IMG_EXTS


def join_path(base: str, name: str) -> str:
    return base + name if base.endswith("/") else base + "/" + name


def _read_bytes(path: str) -> Optional[bytes]:
    try:
        if path.startswith("s3://") or path.startswith("oss://"):
            with mopen(path, "rb") as f:
                return f.read()
        with open(path, "rb") as f:
            return f.read()
    except Exception as e:
        log(f"[Warn] 读取失败 {path}: {e}")
        return None


def _load_image(path: str) -> Optional[Image.Image]:
    b = _read_bytes(path)
    if b is None:
        return None
    try:
        img = Image.open(BytesIO(b))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as e:
        log(f"[Warn] 解码图片失败 {path}: {e}")
        return None


def _resize_keep_long_side(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    side = max(w, h)
    if side <= max_side:
        return img
    scale = max_side / float(side)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)


def get_image_data_uri(path: str) -> Optional[str]:
    img = _load_image(path)
    if img is None:
        return None
    img = _resize_keep_long_side(img, RESIZE_MAX_SIDE)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1]).strip()
    return s


def _safe_exp(x: float) -> float:
    if x > 60:
        return math.exp(60)
    if x < -60:
        return math.exp(-60)
    return math.exp(x)


def _softmax2(logp0: float, logp1: float) -> Tuple[float, float]:
    m = max(logp0, logp1)
    a0 = _safe_exp(logp0 - m)
    a1 = _safe_exp(logp1 - m)
    denom = a0 + a1
    if denom <= 0:
        return 0.5, 0.5
    return a0 / denom, a1 / denom


def call_qwen_chat_raw(
    messages: list,
    temperature: float = 0.0,
    max_tokens: int = 1,
    need_logprobs: bool = True,
    top_logprobs: int = 8,
    max_retries: int = 0,
    retry_delay: float = 2.0,
) -> Optional[Dict[str, Any]]:
    payload: Dict[str, Any] = {
        "model": MODEL,
        "temperature": float(temperature),
        "messages": messages,
        "max_tokens": int(max_tokens),
    }
    if need_logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = int(top_logprobs)
        payload["top_k"] = int(top_logprobs)

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                BASE_URL.rstrip("/") + "/chat/completions",
                headers=headers,
                json=payload,
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            log(f"[Err] API 请求出错(第 {attempt + 1} 次): {e}")
            if attempt < max_retries:
                time.sleep(retry_delay)
            else:
                return None


def call_qwen_chat_with_retry(
    messages: list,
    temperature: float,
    max_tokens: int,
    need_logprobs: bool,
    top_logprobs: int,
    retry_times: int,
    retry_delay: float,
) -> Optional[Dict[str, Any]]:
    total_attempts = max(0, retry_times) + 1
    for attempt in range(1, total_attempts + 1):
        resp = call_qwen_chat_raw(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            need_logprobs=need_logprobs,
            top_logprobs=top_logprobs,
            max_retries=0,
            retry_delay=retry_delay,
        )
        if resp:
            return resp
        log(f"[Err] API 无响应/请求失败(第 {attempt}/{total_attempts} 次)")
        if attempt < total_attempts:
            time.sleep(retry_delay)
    return None


def _extract_text_from_choice(choice: Dict[str, Any]) -> str:
    msg = choice.get("message", {}) if isinstance(choice.get("message"), dict) else {}
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                parts.append(c.get("text", ""))
            elif isinstance(c, str):
                parts.append(c)
        return "\n".join(parts)
    return str(content)


def _collect_top_logprobs_mapping(resp_json: Dict[str, Any]) -> Dict[str, float]:
    mapping: Dict[str, float] = {}
    if not resp_json:
        return mapping
    choices = resp_json.get("choices", [])
    if not choices:
        return mapping
    choice0 = choices[0] if isinstance(choices[0], dict) else {}

    logprobs = choice0.get("logprobs", None)
    if logprobs is None:
        msg = choice0.get("message", {}) if isinstance(choice0.get("message"), dict) else {}
        logprobs = msg.get("logprobs", None)

    if not isinstance(logprobs, dict):
        return mapping

    content = logprobs.get("content")
    if isinstance(content, list) and content:
        first = content[0] if isinstance(content[0], dict) else None
        if isinstance(first, dict):
            top = first.get("top_logprobs")
            if isinstance(top, list):
                for item in top:
                    if isinstance(item, dict):
                        tok = item.get("token")
                        lp = item.get("logprob")
                        if isinstance(tok, str) and isinstance(lp, (int, float)):
                            mapping[tok] = float(lp)
                if mapping:
                    return mapping
            if isinstance(top, dict):
                for tok, lp in top.items():
                    if isinstance(tok, str) and isinstance(lp, (int, float)):
                        mapping[tok] = float(lp)
                if mapping:
                    return mapping

    top_lp = logprobs.get("top_logprobs")
    if isinstance(top_lp, list) and top_lp:
        first = top_lp[0]
        if isinstance(first, dict):
            for tok, lp in first.items():
                if isinstance(tok, str) and isinstance(lp, (int, float)):
                    mapping[tok] = float(lp)
            if mapping:
                return mapping
        if isinstance(first, list):
            for item in first:
                if isinstance(item, dict):
                    tok = item.get("token")
                    lp = item.get("logprob")
                    if isinstance(tok, str) and isinstance(lp, (int, float)):
                        mapping[tok] = float(lp)
            if mapping:
                return mapping

    return mapping

def _sigmoid(x: float) -> float:
    # stable sigmoid
    if x >= 0:
        z = _safe_exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = _safe_exp(x)
        return z / (1.0 + z)

def _p1_from_logprobs(logp0: float, logp1: float) -> float:
    # normalize only over {0,1}
    return _sigmoid(logp1 - logp0)

def _extract_01_logprobs(resp_json: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Dict[str, float]]:
    mapping_raw = _collect_top_logprobs_mapping(resp_json)
    # logp0 = None
    # logp1 = None
    # for tok, lp in mapping_raw.items():
    #     t = tok.strip()
    #     if t == "0":
    #         logp0 = lp
    #     elif t == "1":
    #         logp1 = lp
    # return logp0, logp1, mapping_raw
    cand0 = []
    cand1 = []
    for tok, lp in mapping_raw.items():
        t = tok.strip()
        if t == "0":
            cand0.append(lp)
        elif t == "1":
            cand1.append(lp)

    logp0 = max(cand0) if cand0 else None
    logp1 = max(cand1) if cand1 else None
    return logp0, logp1, mapping_raw


def direct_judge_images_generic(path_a: str, path_b: str, system_prompt: str, user_instruction: str) -> Tuple[Optional[bool], str, Optional[float]]:
    data_a = get_image_data_uri(path_a)
    data_b = get_image_data_uri(path_b)
    if not data_a or not data_b:
        return None, "图片编码失败", None

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_instruction},
                {"type": "text", "text": "下面是图片 A："},
                {"type": "image_url", "image_url": {"url": data_a}},
                {"type": "text", "text": "下面是图片 B："},
                {"type": "image_url", "image_url": {"url": data_b}},
                {"type": "text", "text": "只输出一个字符：0 或 1。"},
            ],
        },
    ]

    args = G_ARGS
    retry_times = int(args.conn_retry_times) if args is not None else 0
    retry_delay = float(args.conn_retry_delay) if args is not None else 2.0
    resp_json = call_qwen_chat_with_retry(
        messages,
        temperature=0.0,
        max_tokens=1,
        need_logprobs=True,
        top_logprobs=8,
        retry_times=retry_times,
        retry_delay=retry_delay,
    )
    if not resp_json:
        return None, RETRY_EXHAUSTED_REASON, None

    choices = resp_json.get("choices", [])
    if not choices or not isinstance(choices[0], dict):
        return None, "返回结构异常(无 choices)", None

    text = strip_code_fences(_extract_text_from_choice(choices[0])).strip()
    pred_char = None
    for ch in text:
        if not ch.isspace():
            pred_char = ch
            break
    if pred_char not in ("0", "1"):
        return None, f"输出不是 0/1 (got={text!r})", None

    pred_is_consistent = (pred_char == "1")

    logp0, logp1, mapping_raw = _extract_01_logprobs(resp_json)
    if logp0 is None or logp1 is None:
        return None, f"无法提取 0/1 top_logprobs (keys={list(mapping_raw.keys())[:8]})", None

    # p0, p1 = _softmax2(logp0, logp1)
    # conf = p1 if pred_is_consistent else p0
    # reason = f"pred={pred_char}, conf={conf:.3f} (p0={p0:.3f}, p1={p1:.3f})"
    score = _p1_from_logprobs(logp0, logp1)  # 统一的相似度分数
    margin = logp1 - logp0
    reason = f"pred={pred_char}, score(p1)={score:.3f}, margin(logp1-logp0)={margin:.3f}"
    return pred_is_consistent, reason, score


def mean_confidence(samples: List[Tuple[Optional[bool], Optional[float], str]]) -> Optional[float]:
    confs = [pred * conf for pred, conf, _ in samples if conf is not None]
    if len(confs) < 3:
        return None
    return sum(confs) / len(confs)


def _process_one_name_simple(name: str) -> Tuple[str, Any, Any]:
    args = G_ARGS
    content_dir = args.content_dir
    style_dir = args.style_dir
    result_dir = args.result_dir

    content_path = join_path(content_dir, name)
    style_path = join_path(style_dir, name)
    result_path = join_path(result_dir, name)

    if not smart_exists(content_path) or not smart_exists(style_path) or not smart_exists(result_path):
        return name, None, None

    content_samples = []
    for _ in range(3):
        pred, reason, conf = direct_judge_images_generic(
            result_path,
            content_path,
            CONTENT_SYSTEM_PROMPT,
            CONTENT_USER_INSTRUCTION,
        )
        content_samples.append((pred, conf, reason))

    style_samples = []
    for _ in range(3):
        pred, reason, conf = direct_judge_images_generic(
            result_path,
            style_path,
            STYLE_SYSTEM_PROMPT,
            STYLE_USER_INSTRUCTION,
        )
        style_samples.append((pred, conf, reason))

    content_score = mean_confidence(content_samples)
    style_score = mean_confidence(style_samples)

    return name, content_score, style_score


def _worker_process_main_simple(model: str, base_url: str, tasks: List[str], result_queue: mp.Queue):
    global MODEL, BASE_URL
    MODEL = model
    BASE_URL = base_url
    for name in tasks:
        name_out, content_score, style_score = _process_one_name_simple(name)
        result_queue.put((name_out, content_score, style_score))


def smart_write_json(path: str, obj: Any):
    data = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
    dir_path = os.path.dirname(path) or "."
    if path.startswith("s3://") or path.startswith("oss://"):
        smart_makedirs(dir_path, exist_ok=True)
        with mopen(path, "wb") as f:
            f.write(data)
    else:
        os.makedirs(dir_path, exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)


def smart_read_json(path: str):
    if not smart_exists(path):
        return None
    try:
        if path.startswith("s3://") or path.startswith("oss://"):
            with mopen(path, "r", encoding="utf-8") as f:
                return json.load(f)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log(f"[Warn] 读取 JSON 失败 {path}: {e}")
        return None


def main():
    global G_ARGS
    ap = argparse.ArgumentParser("双重判别：主体内容 + 画风")
    ap.add_argument("--content_dir", required=True, help="Content 图片目录")
    ap.add_argument("--style_dir", required=True, help="Style 图片目录")
    ap.add_argument("--result_dir", required=True, help="生成结果图片目录")
    ap.add_argument("--output_content_json", required=True, help="内容分数输出 JSON 路径")
    ap.add_argument("--output_style_json", required=True, help="风格分数输出 JSON 路径")
    ap.add_argument("--model", type=str, default=MODEL)
    ap.add_argument("--base_url", type=str, default=BASE_URL)
    ap.add_argument("--endpoint", action="append", default=[])
    ap.add_argument("--procs_per_endpoint", type=int, default=0)
    ap.add_argument("--conn_retry_times", type=int, default=5, help="连接失败时的重试次数")
    ap.add_argument("--conn_retry_delay", type=float, default=2.0, help="连接失败重试间隔(秒)")
    ap.add_argument("--num_samples", type=int, default=0, help="随机抽样数量（<=0 表示全量）")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--num_procs", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    G_ARGS = args

    endpoints: List[Tuple[str, str]] = []
    if args.endpoint:
        for e in args.endpoint:
            s = str(e).strip()
            if not s:
                continue
            if "@" in s:
                m_name, url = s.split("@", 1)
                m_name = m_name.strip()
                url = url.strip()
            else:
                m_name = args.model
                url = s
            if m_name and url:
                endpoints.append((m_name, url))
    if not endpoints:
        globals()["MODEL"] = args.model
        globals()["BASE_URL"] = args.base_url

    content_files = set(smart_listdir(args.content_dir))
    style_files = set(smart_listdir(args.style_dir))
    result_files = set(smart_listdir(args.result_dir))
    common_files = list(content_files & style_files & result_files)
    common_files = [f for f in common_files if is_image_name(f)]

    if args.num_samples > 0 and len(common_files) > args.num_samples:
        import random
        random.seed(args.seed)
        common_files = random.sample(common_files, args.num_samples)

    log(f"Found {len(common_files)} common images to process.")

    content_results = {}
    style_results = {}
    if (not args.overwrite) and smart_exists(args.output_content_json) and smart_exists(args.output_style_json):
        tmp_c = smart_read_json(args.output_content_json)
        tmp_s = smart_read_json(args.output_style_json)
        if isinstance(tmp_c, dict) and isinstance(tmp_s, dict):
            content_results = tmp_c
            style_results = tmp_s
            processed_keys = set(tmp_c.keys()) & set(tmp_s.keys())
            common_files = [f for f in common_files if os.path.splitext(f)[0] not in processed_keys]
            log(f"[Resume] {len(common_files)} remaining to process.")

    if not common_files:
        log("No tasks to run.")
        return

    num_procs = args.num_procs
    if num_procs <= 0:
        num_procs = len(endpoints) * args.procs_per_endpoint if endpoints and args.procs_per_endpoint > 0 else 4

    result_queue = mp.Queue()
    workers = []
    chunk_size = (len(common_files) + num_procs - 1) // num_procs

    for i in range(num_procs):
        sub_tasks = common_files[i * chunk_size : (i + 1) * chunk_size]
        if not sub_tasks:
            continue
        if endpoints:
            m_name, url = endpoints[i % len(endpoints)]
        else:
            m_name, url = MODEL, BASE_URL
        p = mp.Process(
            target=_worker_process_main_simple,
            args=(m_name, url, sub_tasks, result_queue),
        )
        p.start()
        workers.append(p)

    total_done = 0
    total_tasks = len(common_files)
    with tqdm(total=total_tasks, desc="Processing") as pbar:
        while total_done < total_tasks:
            try:
                name, content_score, style_score = result_queue.get(timeout=5)
                base = os.path.splitext(name)[0]
                content_results[base] = content_score
                style_results[base] = style_score
                total_done += 1
                pbar.update(1)
                if total_done % 50 == 0:
                    smart_write_json(args.output_content_json, content_results)
                    smart_write_json(args.output_style_json, style_results)
            except Exception:
                if not any(p.is_alive() for p in workers) and result_queue.empty():
                    break

    for p in workers:
        p.join()

    smart_write_json(args.output_content_json, content_results)
    smart_write_json(args.output_style_json, style_results)
    log(f"Done. Results saved to {args.output_content_json} and {args.output_style_json}")


if __name__ == "__main__":
    main()
