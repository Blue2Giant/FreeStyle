#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
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
# MODEL = "Qwen3VL30BA3B-Image-Edit"
# MODEL = "v1p3"
# BASE_URL = "http://stepcast-router.shai-core:9200/v1"
MODEL = "Qwen3-VL-30B-A3B-Instruct"
BASE_URL = "http://10.201.19.61:22002/v1"

# MODEL="Qwen3-VL-30B-A3B-Instruct"
# BASE_URL = "http://10.201.17.30:2202/v1"
TIMEOUT = 720
RETRY_EXHAUSTED_REASON = "API 重试耗尽"

# Globals for multiprocessing
G_TRIPLET_DIR = ""
G_CONTENT_DIRS = []
G_STYLE_DIRS = []
G_ARGS = None

RESIZE_MAX_SIDE = 1024
JPEG_QUALITY = 85
Image.MAX_IMAGE_PIXELS = None

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# ==================== Prompts ====================

# Content Prompts
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

# Style Prompts
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



# ==================== Helper Functions ====================

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def is_image_name(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in IMG_EXTS


def norm_dir(p: str) -> str:
    return p if p.endswith("/") else (p + "/")


def join_path(base: str, name: str) -> str:
    return base + name if base.endswith("/") else base + "/" + name


def _read_bytes(path: str) -> Optional[bytes]:
    try:
        if path.startswith("s3://"):
            with mopen(path, "rb") as f:
                return f.read()
        else:
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
    # 外层重试：失败就阻塞，直到成功或重试次数用尽
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


def _extract_01_logprobs(resp_json: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Dict[str, float]]:
    mapping_raw = _collect_top_logprobs_mapping(resp_json)
    logp0 = None
    logp1 = None
    for tok, lp in mapping_raw.items():
        t = tok.strip()
        if t == "0":
            logp0 = lp
        elif t == "1":
            logp1 = lp
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

    p0, p1 = _softmax2(logp0, logp1)
    conf = p1 if pred_is_consistent else p0
    reason = f"pred={pred_char}, conf={conf:.3f} (p0={p0:.3f}, p1={p1:.3f})"
    return pred_is_consistent, reason, conf


# ==================== Generic Voting Logic ====================

def judge_pair_voting(
    path_a: str,
    path_b: str,
    system_prompt: str,
    user_instruction: str,
    conf_thr: float,
    judge_times: int,
    min_true: int,
) -> Tuple[Optional[bool], Dict[str, Any], bool]:
    trials: List[Dict[str, Any]] = []
    good_true = 0
    retry_exhausted = False

    for i in range(1, judge_times + 1):
        pred, reason, conf = direct_judge_images_generic(
            path_a, 
            path_b,
            system_prompt,
            user_instruction
        )
        # 任意一次判别重试耗尽则视为本样本不可判
        if pred is None and reason == RETRY_EXHAUSTED_REASON:
            retry_exhausted = True
            break
        is_valid = isinstance(pred, bool) and isinstance(conf, (int, float)) and (conf > conf_thr)
        if is_valid and pred is True:
            good_true += 1
        trials.append(
            {
                "call": i,
                "pred": pred,
                "conf": conf,
                "valid": is_valid,
                "reason": reason,
            }
        )

    if retry_exhausted:
        detail = {
            "path_a": path_a,
            "path_b": path_b,
            "conf_thr": conf_thr,
            "judge_times": judge_times,
            "min_true": min_true,
            "good_true": good_true,
            "trials": trials,
            "status": "retry_exhausted",
        }
        return None, detail, True

    passed = good_true >= min_true
    detail = {
        "path_a": path_a,
        "path_b": path_b,
        "conf_thr": conf_thr,
        "judge_times": judge_times,
        "min_true": min_true,
        "good_true": good_true,
        "trials": trials,
        "status": "ok",
        "passed": passed,
    }
    return passed, detail, False




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
        else:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        log(f"[Warn] 读取 JSON 失败 {path}: {e}")
        return None


def outputs_exist(args) -> bool:
    paths = [args.out_all, args.out_pos, args.out_neg]
    if args.out_detail:
        paths.append(args.out_detail)
    for p in paths:
        if not smart_exists(p):
            return False
    return True


def list_dirs(root: str, prefix: str) -> Tuple[str, List[str]]:
    root = norm_dir(root)
    main_dir = join_path(root, "style_and_content/")
    if not smart_exists(main_dir):
        raise RuntimeError(f"缺少目录: {main_dir}")

    entries = smart_listdir(root)
    target_dirs = []
    for e in entries:
        name = str(e).rstrip("/")
        if re.fullmatch(rf"{prefix}_\d+", name):
            target_dirs.append(name + "/")

    def key_fn(x: str) -> int:
        m = re.search(rf"{prefix}_(\d+)/", x)
        return int(m.group(1)) if m else 10**9

    target_dirs.sort(key=key_fn)
    return main_dir, [join_path(root, d) for d in target_dirs]


def list_candidate_names(target_dir: str) -> List[str]:
    names = []
    for e in smart_listdir(target_dir):
        if e.endswith("/"):
            continue
        if is_image_name(e):
            names.append(e)
    names.sort()
    return names


def read_id_txt(path: str) -> List[str]:
    if not path:
        return []
    if not smart_exists(path):
        log(f"[Warn] id txt not found: {path}")
        return []
    try:
        with mopen(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        ids = []
        for line in lines:
            s = str(line).strip()
            if not s or s.startswith("#"):
                continue
            nums = re.findall(r"\d+", s)
            if nums:
                ids.append(nums[0])
            else:
                ids.append(s)
        return ids
    except Exception as e:
        log(f"[Warn] read id txt failed: {path} ({e})")
        return []


def extract_content_style_ids(name: str) -> Tuple[str, str]:
    base = os.path.splitext(os.path.basename(name))[0]
    m = re.search(r"(\d+)[^0-9]+(\d+)", base)
    if not m:
        return "", ""
    style_id, content_id = m.group(1), m.group(2)
    return content_id, style_id


def _process_one_name(name: str) -> Tuple[Optional[Dict[str, Any]], int, bool]:
    # Use globals
    triplet_dir = G_TRIPLET_DIR
    content_dirs = G_CONTENT_DIRS
    style_dirs = G_STYLE_DIRS
    args = G_ARGS
    
    main_img_path = join_path(triplet_dir, name)
    
    # --- 1. Content Judge ---
    passed_content = 0
    total_content = len(content_dirs)
    per_content_details = []
    
    for cdir in content_dirs:
        content_path = join_path(cdir, name)
        content_tag = os.path.basename(cdir.rstrip("/"))
        
        if not smart_exists(content_path):
            per_content_details.append({"dir": content_tag, "exists": False})
            continue

        decision, detail, retry_exhausted = judge_pair_voting(
            path_a=main_img_path,
            path_b=content_path,
            system_prompt=CONTENT_SYSTEM_PROMPT,
            user_instruction=CONTENT_USER_INSTRUCTION,
            conf_thr=args.content_conf_thr,
            judge_times=args.content_judge_times,
            min_true=args.content_min_true,
        )
        if retry_exhausted:
            return None, 0, True
        decision = bool(decision is True)
        if decision:
            passed_content += 1
        per_content_details.append({"dir": content_tag, "exists": True, "decision": decision, "detail": detail})
        
    content_r = passed_content / float(total_content) if total_content > 0 else 0.0
    content_pass = content_r >= args.content_ratio
    
    # --- 2. Style Judge ---
    passed_style = 0
    total_style = len(style_dirs)
    per_style_details = []
    
    for sdir in style_dirs:
        style_path = join_path(sdir, name)
        style_tag = os.path.basename(sdir.rstrip("/"))
        
        if not smart_exists(style_path):
            per_style_details.append({"dir": style_tag, "exists": False})
            continue
        
        if args.style_repeat_only_style1 and style_tag != "style_1":
            pred, reason, conf = direct_judge_images_generic(
                main_img_path, 
                style_path, 
                STYLE_SYSTEM_PROMPT, 
                STYLE_USER_INSTRUCTION
            )
            if pred is None and reason == RETRY_EXHAUSTED_REASON:
                return None, 0, True
            decision = bool(pred is True and conf is not None and conf > args.style_conf_thr)
            detail = {"status": "single", "pred": pred, "conf": conf, "reason": reason}
        else:
            decision, detail, retry_exhausted = judge_pair_voting(
                path_a=main_img_path, 
                path_b=style_path,
                system_prompt=STYLE_SYSTEM_PROMPT,
                user_instruction=STYLE_USER_INSTRUCTION,
                conf_thr=args.style_conf_thr,
                judge_times=args.style_judge_times,
                min_true=args.style_min_true,
            )
            if retry_exhausted:
                return None, 0, True
            decision = bool(decision is True)
        
        if decision:
            passed_style += 1
        per_style_details.append({"dir": style_tag, "exists": True, "decision": decision, "detail": detail})
        
    style_r = passed_style / float(total_style) if total_style > 0 else 0.0
    style_pass = style_r >= args.style_ratio
    
    # --- Dual Result ---
    dual_pass = content_pass and style_pass
    
    rec = {
        "name": name,
        "main_img": main_img_path,
        # Content stats
        "content_pass": content_pass,
        "content_ratio": content_r,
        "content_passed_cnt": passed_content,
        "content_total": total_content,
        "content_details": per_content_details,
        # Style stats
        "style_pass": style_pass,
        "style_ratio": style_r,
        "style_passed_cnt": passed_style,
        "style_total": total_style,
        "style_details": per_style_details,
        # Dual
        "dual_pass": dual_pass
    }
    return rec, 1 if dual_pass else 0, False


def _extract_pair_ids_from_path(path: str) -> Tuple[str, str]:
    parent = os.path.basename(os.path.dirname(path or ""))
    if "__" in parent:
        content_id, style_id = parent.split("__", 1)
        content_id, style_id = content_id.strip(), style_id.strip()
        if content_id and style_id:
            return content_id, style_id
    content_id, style_id = extract_content_style_ids(os.path.basename(path or ""))
    return content_id, style_id


def _collect_prefixed_items(rec: Dict[str, Any], prefix: str) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    for k, v in rec.items():
        if not isinstance(k, str) or not k.startswith(prefix):
            continue
        if not isinstance(v, str) or not v:
            continue
        items.append((k, v))

    def key_fn(x: Tuple[str, str]) -> int:
        m = re.search(r"(\d+)$", x[0])
        return int(m.group(1)) if m else 10**9

    items.sort(key=key_fn)
    return items


def _process_one_record(task: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], int, bool]:
    args = G_ARGS
    main_img_path = task.get("main_img", "")
    name = task.get("name", os.path.basename(main_img_path))
    content_items: List[Tuple[str, str]] = task.get("content_items", [])
    style_items: List[Tuple[str, str]] = task.get("style_items", [])

    passed_content = 0
    total_content = len(content_items)
    per_content_details = []

    for content_tag, content_path in content_items:
        if not smart_exists(content_path):
            per_content_details.append({"dir": content_tag, "exists": False})
            continue

        decision, detail, retry_exhausted = judge_pair_voting(
            path_a=main_img_path,
            path_b=content_path,
            system_prompt=CONTENT_SYSTEM_PROMPT,
            user_instruction=CONTENT_USER_INSTRUCTION,
            conf_thr=args.content_conf_thr,
            judge_times=args.content_judge_times,
            min_true=args.content_min_true,
        )
        if retry_exhausted:
            return None, 0, True
        decision = bool(decision is True)
        if decision:
            passed_content += 1
        per_content_details.append({"dir": content_tag, "exists": True, "decision": decision, "detail": detail})

    content_r = passed_content / float(total_content) if total_content > 0 else 0.0
    content_pass = content_r >= args.content_ratio

    passed_style = 0
    total_style = len(style_items)
    per_style_details = []

    for style_tag, style_path in style_items:
        if not smart_exists(style_path):
            per_style_details.append({"dir": style_tag, "exists": False})
            continue

        if args.style_repeat_only_style1 and style_tag != "style_1":
            pred, reason, conf = direct_judge_images_generic(
                main_img_path,
                style_path,
                STYLE_SYSTEM_PROMPT,
                STYLE_USER_INSTRUCTION,
            )
            if pred is None and reason == RETRY_EXHAUSTED_REASON:
                return None, 0, True
            decision = bool(pred is True and conf is not None and conf > args.style_conf_thr)
            detail = {"status": "single", "pred": pred, "conf": conf, "reason": reason}
        else:
            decision, detail, retry_exhausted = judge_pair_voting(
                path_a=main_img_path,
                path_b=style_path,
                system_prompt=STYLE_SYSTEM_PROMPT,
                user_instruction=STYLE_USER_INSTRUCTION,
                conf_thr=args.style_conf_thr,
                judge_times=args.style_judge_times,
                min_true=args.style_min_true,
            )
            if retry_exhausted:
                return None, 0, True
            decision = bool(decision is True)

        if decision:
            passed_style += 1
        per_style_details.append({"dir": style_tag, "exists": True, "decision": decision, "detail": detail})

    style_r = passed_style / float(total_style) if total_style > 0 else 0.0
    style_pass = style_r >= args.style_ratio

    dual_pass = content_pass and style_pass

    rec = {
        "name": name,
        "main_img": main_img_path,
        "content_pass": content_pass,
        "content_ratio": content_r,
        "content_passed_cnt": passed_content,
        "content_total": total_content,
        "content_details": per_content_details,
        "style_pass": style_pass,
        "style_ratio": style_r,
        "style_passed_cnt": passed_style,
        "style_total": total_style,
        "style_details": per_style_details,
        "dual_pass": dual_pass,
    }
    return rec, 1 if dual_pass else 0, False

def _worker_process_main(model: str, base_url: str, tasks: List[str], result_queue: mp.Queue):
    global MODEL, BASE_URL
    MODEL = model
    BASE_URL = base_url
    for name in tasks:
        rec, ok_inc, skipped = _process_one_name(name)
        result_queue.put((rec, ok_inc, skipped))


def _worker_process_main_records(model: str, base_url: str, tasks: List[Dict[str, Any]], result_queue: mp.Queue):
    global MODEL, BASE_URL
    MODEL = model
    BASE_URL = base_url
    for task in tasks:
        rec, ok_inc, skipped = _process_one_record(task)
        result_queue.put((rec, ok_inc, skipped))


def main():
    global MODEL, BASE_URL, G_TRIPLET_DIR, G_CONTENT_DIRS, G_STYLE_DIRS, G_ARGS

    ap = argparse.ArgumentParser("双重判别：同时评估主体内容一致性和画风相似度")
    ap.add_argument("--root", default="", help="输出目录根：包含 style_and_content/ content_*/ style_*/")
    ap.add_argument("--input_jsonl", default="", help="jsonl 输入，每行包含 style_and_content/content_*/style_* 的图片路径")
    ap.add_argument("--num_samples", type=int, default=2000, help="随机抽样数量（<=0 表示全量）")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--content_conf_thr", type=float, default=0.5, help="content 单次有效阈值")
    ap.add_argument("--style_conf_thr", type=float, default=0.5, help="style 单次有效阈值")

    ap.add_argument("--content_judge_times", type=int, default=3)
    ap.add_argument("--content_min_true", type=int, default=2)
    ap.add_argument("--content_ratio", type=float, default=0.66)

    ap.add_argument("--style_judge_times", type=int, default=3)
    ap.add_argument("--style_min_true", type=int, default=2)
    ap.add_argument("--style_ratio", type=float, default=0.66)
    ap.add_argument("--style_repeat_only_style1", action="store_true")

    ap.add_argument("--content_id_txt", default="", help="仅处理这些 content_id（txt 每行一个）")
    ap.add_argument("--style_id_txt", default="", help="仅处理这些 style_id（txt 每行一个）")

    ap.add_argument("--model", type=str, default=MODEL)
    ap.add_argument("--base_url", type=str, default=BASE_URL)
    ap.add_argument("--endpoint", action="append", default=[])
    ap.add_argument("--procs_per_endpoint", type=int, default=0)
    ap.add_argument("--conn_retry_times", type=int, default=5, help="连接失败时的重试次数")
    ap.add_argument("--conn_retry_delay", type=float, default=2.0, help="连接失败重试间隔(秒)")

    ap.add_argument("--out_all", required=True, help="全量 map json (Dual Pass)")
    ap.add_argument("--out_pos", required=True, help="正样本 map json (Dual Pass)")
    ap.add_argument("--out_neg", required=True, help="负样本 map json (Dual Pass)")
    ap.add_argument("--out_detail", default="", help="详细结果 JSON")
    ap.add_argument("--overwrite", action="store_true")

    ap.add_argument("--num_procs", type=int, default=0)

    args = ap.parse_args()

    existing_all_map = {}
    processed_keys = set()
    if (not args.overwrite) and smart_exists(args.out_all):
        tmp = smart_read_json(args.out_all)
        if isinstance(tmp, dict):
            existing_all_map = tmp
            processed_keys = set(tmp.keys())
            log(f"[Resume] 从已有 out_all 读取到 {len(processed_keys)} 条结果，将按 key 跳过已完成样本")

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
    log(f"[Info] endpoints={endpoints}")
    if not endpoints:
        MODEL = args.model
        BASE_URL = args.base_url

    use_jsonl = bool(args.input_jsonl)
    picked = []
    tasks: List[Dict[str, Any]] = []
    cand_names: List[str] = []

    if use_jsonl:
        if not smart_exists(args.input_jsonl):
            raise RuntimeError(f"input_jsonl 不存在: {args.input_jsonl}")
        with mopen(args.input_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                s = (line or "").strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                main_img = obj.get("style_and_content")
                if not isinstance(main_img, str) or not main_img:
                    continue
                content_id, style_id = _extract_pair_ids_from_path(main_img)
                base_name = os.path.basename(main_img)
                if content_id and style_id:
                    name = f"{style_id}__{content_id}_{base_name}"
                else:
                    name = base_name
                task = {
                    "name": name,
                    "main_img": main_img,
                    "content_items": _collect_prefixed_items(obj, "content_"),
                    "style_items": _collect_prefixed_items(obj, "style_"),
                }
                tasks.append(task)

        if not tasks:
            raise RuntimeError(f"{args.input_jsonl} 中未解析到有效样本")

        import random
        rng = random.Random(args.seed)
        if args.num_samples <= 0 or args.num_samples >= len(tasks):
            picked = list(tasks)
        else:
            picked = rng.sample(tasks, args.num_samples)

        if processed_keys:
            before = len(picked)
            picked = [
                task
                for task in picked
                if os.path.splitext(task["name"])[0] not in processed_keys
            ]
            log(f"[Resume] 采样 {before} 条后，过滤已存在 key 剩余 {len(picked)} 条待处理")
            if not picked:
                log("[Resume] 没有新的样本需要判别，直接退出。")
                return

        log(f"[Info] picked={len(picked)} from jsonl={len(tasks)}")
    else:
        if not args.root:
            raise RuntimeError("--root 不能为空")
        triplet_dir, content_dirs = list_dirs(args.root, "content")
        _, style_dirs = list_dirs(args.root, "style")

        if not content_dirs:
            raise RuntimeError(f"在 {args.root} 下没找到 content_*/ 目录")
        if not style_dirs:
            raise RuntimeError(f"在 {args.root} 下没找到 style_*/ 目录")

        cand_names = list_candidate_names(triplet_dir)
        if not cand_names:
            raise RuntimeError(f"{triplet_dir} 下没找到图片")

        content_ids = read_id_txt(args.content_id_txt)
        style_ids = read_id_txt(args.style_id_txt)
        content_id_set = set(content_ids)
        style_id_set = set(style_ids)
        if content_id_set or style_id_set:
            before = len(cand_names)
            print(f"[Filter] 过滤前 {before} 条样本")
            filtered = []
            for name in cand_names:
                content_id, style_id = extract_content_style_ids(name)
                if content_id_set and content_id not in content_id_set:
                    continue
                if style_id_set and style_id not in style_id_set:
                    continue
                filtered.append(name)
            cand_names = filtered
            log(f"[Filter] ids before={before} after={len(cand_names)}")
            if not cand_names:
                log("[Filter] 过滤后没有候选图片，直接退出。")
                return

        import random
        rng = random.Random(args.seed)
        if args.num_samples <= 0 or args.num_samples >= len(cand_names):
            picked = list(cand_names)
        else:
            picked = rng.sample(cand_names, args.num_samples)

        if processed_keys:
            before = len(picked)
            picked = [
                name
                for name in picked
                if os.path.splitext(name)[0] not in processed_keys
            ]
            log(f"[Resume] 采样 {before} 条后，过滤已存在 key 剩余 {len(picked)} 条待处理")
            if not picked:
                log("[Resume] 没有新的样本需要判别，直接退出。")
                return

        log(f"[Info] picked={len(picked)} from candidates={len(cand_names)}")

        G_TRIPLET_DIR = triplet_dir
        G_CONTENT_DIRS = content_dirs
        G_STYLE_DIRS = style_dirs
    G_ARGS = args

    results: List[Dict[str, Any]] = []
    dual_ok_cnt = 0
    skipped_cnt = 0

    all_map: Dict[str, int] = dict(existing_all_map)
    pos_map: Dict[str, int] = {}
    neg_map: Dict[str, int] = {}

    if (not args.overwrite) and smart_exists(args.out_pos):
        tmp_pos = smart_read_json(args.out_pos)
        if isinstance(tmp_pos, dict):
            pos_map = dict(tmp_pos)
    if (not args.overwrite) and smart_exists(args.out_neg):
        tmp_neg = smart_read_json(args.out_neg)
        if isinstance(tmp_neg, dict):
            neg_map = dict(tmp_neg)

    content_ok_cnt = 0
    style_ok_cnt = 0

    def _update_maps_and_flush(rec: Dict[str, Any]):
        nonlocal content_ok_cnt, style_ok_cnt
        fname = rec["name"]
        base = os.path.splitext(fname)[0]
        v = 1 if rec["dual_pass"] else 0
        all_map[base] = v
        if v == 1:
            pos_map[base] = 1
        else:
            neg_map[base] = 0

        if rec["content_pass"]:
            content_ok_cnt += 1
        if rec["style_pass"]:
            style_ok_cnt += 1

        smart_write_json(args.out_all, all_map)
        smart_write_json(args.out_pos, pos_map)
        smart_write_json(args.out_neg, neg_map)

    if endpoints:
        per = args.procs_per_endpoint if args.procs_per_endpoint and args.procs_per_endpoint > 0 else 1
        workers: List[mp.Process] = []
        result_queue: mp.Queue = mp.Queue()
        worker_specs: List[Tuple[str, str]] = []
        for _ in range(per):
            for model_name, url in endpoints:
                worker_specs.append((model_name, url))
        worker_count = len(worker_specs)
        sliced: List[List[Any]] = [[] for _ in range(worker_count)]
        for idx, item in enumerate(picked):
            sliced[idx % worker_count].append(item)
        for i, (model_name, url) in enumerate(worker_specs):
            sub_tasks = sliced[i]
            if not sub_tasks:
                continue
            if use_jsonl:
                p = mp.Process(target=_worker_process_main_records, args=(model_name, url, sub_tasks, result_queue))
            else:
                p = mp.Process(target=_worker_process_main, args=(model_name, url, sub_tasks, result_queue))
            p.daemon = False
            p.start()
            workers.append(p)
        total = len(picked)
        for _ in tqdm(range(total), desc="DualJudge-MP", unit="img"):
            rec, ok_inc, skipped = result_queue.get()
            if skipped or rec is None:
                skipped_cnt += 1
                continue
            results.append(rec)
            dual_ok_cnt += ok_inc
            _update_maps_and_flush(rec)
        for p in workers:
            p.join()
    elif args.num_procs and args.num_procs > 1:
        procs = args.num_procs
        if procs <= 0:
            procs = mp.cpu_count() or 1
        procs = max(1, procs)
        log(f"[Info] 使用多进程 num_procs={procs}，待处理样本={len(picked)}")
        with mp.Pool(processes=procs) as pool:
            if use_jsonl:
                it = pool.imap_unordered(_process_one_record, picked)
            else:
                it = pool.imap_unordered(_process_one_name, picked)
            for rec, ok_inc, skipped in tqdm(
                it,
                total=len(picked),
                desc="DualJudge-MP",
                unit="img",
            ):
                if skipped or rec is None:
                    skipped_cnt += 1
                    continue
                results.append(rec)
                dual_ok_cnt += ok_inc
                _update_maps_and_flush(rec)
    else:
        for i, item in enumerate(picked, 1):
            if use_jsonl:
                rec, ok_inc, skipped = _process_one_record(item)
            else:
                rec, ok_inc, skipped = _process_one_name(item)
            if skipped or rec is None:
                skipped_cnt += 1
                label = item.get("name") if isinstance(item, dict) else str(item)
                log(f"[{i}/{len(picked)}] {label} -> 跳过(重试耗尽)")
                continue
            results.append(rec)
            dual_ok_cnt += ok_inc
            log(f"[{i}/{len(picked)}] {rec['name']} -> Content:{rec['content_pass']} ({rec['content_ratio']:.2f}), Style:{rec['style_pass']} ({rec['style_ratio']:.2f}) -> Dual:{rec['dual_pass']}")
            _update_maps_and_flush(rec)

    processed_cnt = max(0, len(picked) - skipped_cnt)
    processed_den = processed_cnt if processed_cnt > 0 else 1
    log(f"[DONE] Processed {len(picked)} samples.")
    log(f"Content Pass: {content_ok_cnt} ({content_ok_cnt/processed_den:.2%})")
    log(f"Style Pass:   {style_ok_cnt} ({style_ok_cnt/processed_den:.2%})")
    log(f"Dual Pass:    {dual_ok_cnt} ({dual_ok_cnt/processed_den:.2%})")
    log(f"Skipped:      {skipped_cnt}")
    
    log(f"  -> {args.out_all}")
    log(f"  -> {args.out_pos}")
    log(f"  -> {args.out_neg}")

    if args.out_detail:
        summary = {
            "root": args.root,
            "picked": len(picked),
            "processed": processed_cnt,
            "content_ok": content_ok_cnt,
            "style_ok": style_ok_cnt,
            "dual_ok": dual_ok_cnt,
            "args": vars(args),
            "skipped": skipped_cnt,
        }
        smart_write_json(args.out_detail, {"summary": summary, "results": results})
        log(f"[Detail] -> {args.out_detail}")

if __name__ == "__main__":
    main()
