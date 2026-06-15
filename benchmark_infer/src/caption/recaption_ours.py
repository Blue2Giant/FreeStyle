
import os
"""
python /data/benchmark_metrics/caption_pipe/recaption_ours.py --base /mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content \
    --prompts_json /mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content/prompts.json \
    --output_json s3://lanjinghong-data/sample_800_sref_200_content/sref_prompts_recap.json
"""
# os.environ["no_proxy"] = "stepcast-router.shai-core"

import base64
import json
import random
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import megfile
import PIL.Image
import pyarrow as pa
import pyarrow.parquet as pq
from jinja2 import Template
from loguru import logger
from openai import OpenAI
from tqdm import tqdm


import re
import signal
OPENAI_BASE_URL = "http://YOUR_QWEN_VLM_ENDPOINT/Qwen3-VL-235B-A22B-W8A8/v1"
OPENAI_MODEL_NAME = "qwen3vlw8a8"

PROMPT_WITH_INSTUCTION_CREF_SREF_STYLE_TRANSFER = """

[角色定位] 你是一位顶级的 AI 图像分析与风格迁移标注专家。你的核心任务是解析2张图片：

scene_1 (Content Image): 提供物体、构图和结构的“底图”。
scene_2 (Style Image): 提供艺术风格、笔触、色调、材质的“风格参考图”。

[工作流程：四步思维链]

独立描述 (Independent Captioning): 分别客观描述三张图。
scene_3 的描述必须独立：严禁使用“保持不变”、“变为”等对比性词汇，假设读者没见过前两张图。
风格解构与对比 (Style Decomposition):
识别 scene_2 中的核心艺术特征（如：印象派笔触、赛博朋克霓虹、水墨渲染、低多边形等）。
指令提炼 (Instruction Synthesis): 编写能够指导模型进行这种转化的精确指令。
口语化简写 (Natural Prompting): 提供用户侧的自然语言指令。

[JSON 输出结构] 你必须输出纯粹的 JSON 格式，严禁任何额外解释。
{
  "independent_captions": {
    "scene_1": "（详细描述 Content 图：画面主体、几何结构、构图、背景。不少于 50 字。）",
    "scene_1_en": "（Detailed objective description of scene_1 content.）",
    "scene_2": "（详细描述 Style 图：艺术风格、色彩倾向、笔触质感、光影逻辑。不少于 50 字。）",
    "scene_2_en": "（Detailed description of the artistic style, texture, and color palette in scene_2.）",
    "scene_3": "（利用你的想象能力，在prompt的描述下，描述 scene_3 这张新生成的图像，作为一张独立作品进行描述。重点描述主体形象与整体风格效果，严禁提及任何参考来源。）",
    "scene_3_en": "（Use your imagination to describe the synthesized image scene_3 as a standalone artwork, without referencing other scenes.）"
  },
  "comparative_analysis": {
    "style_inheritance": "（分析说明 场景3从场景2中继承了哪些核心视觉特征，如：色彩映射、笔触走向、光影氛围或艺术流派特征。）",
    "visual_changes": [
      {
        "observation": "（描述具体转化，例如：'场景1中的写实人物在场景3中被赋予了场景2的油画笔触与厚涂纹理'。）",
        "tag": "Style Transfer"
      },
      {
        "observation": "（描述色彩或材质的演变，例如：'场景3采用了场景2的高饱和度霓虹色调'。）",
        "tag": "Color Grading"
      }
    ],
    "fusion_logic_cot": "（逻辑推理：解释场景3是如何由场景1和场景2融合生成的。例如：'保留了场景1的构图骨架与物体位置，但完全替换了场景的渲染逻辑，使其符合场景2的抽象主义审美。'）"
  },
  "predicted_edit_type": "（必须填选：'Local Editing', 'Global Editing', 'Subject Reference Style Transfer'）",
  "training_output": {
    "primary_instruction_cn_123": "（核心指令：描述如何参考场景2的风格对场景1进行风格化重绘。指令必须包含动词和具体的风格应用逻辑。例如：'提取场景2的水彩渲染风格，对场景1的内容进行重绘，重点应用场景2的晕染边缘和低饱和色彩，同时保持场景1的建筑布局。'）",
    "primary_instruction_en_123": "（Formal instruction: 'Restyle the content of scene_1 by applying the artistic style of scene_2, specifically incorporating its [style features] while maintaining the structural composition of scene_1.'）",
    "sample_instruction_cn_123": "（口语化指令：'参考场景2的风格，把场景1重新画一遍。'）",
    "sample_instruction_en_123": "（Natural prompt: 'Transfer the style of scene_2 onto scene_1.'）"
  }
}
"""


PROMPT_WITH_INSTUCTION_CREF_SREF = """
[角色定位]

你是一位顶级的 AI 图像分析与多参考合成标注专家，专注于人物 / 主体 ID 保持与风格一致性的生成任务分析。

你需要解析两张张图片，以及合成的场景3的指令：

输入的场景提示词是 : {{prompt}}

scene_1 (Content Identity Reference Image):
- 提供主体的身份信息（如人物 ID、面部特征、服饰符号、生物形态等）
- 不要求保持其构图、姿态或场景布局
- 场景提示词主要参考场景1的内容合成到合成图中

scene_2 (Style Reference Image):
- 提供整体艺术风格、视觉语言、色彩体系、材质与渲染逻辑

[工作流程：四步思维链]

第一步：独立描述 (Independent Captioning)
- 分别对两张图进行客观、独立的视觉描述
- 想象在prompt的描述下应该呈现的画面scene_3，并进行描述
- scene_3 的描述必须是“孤立的最终作品描述”
- 严禁使用“保持”“来自”“迁移”“参考”等对比性词汇
- 假设读者从未见过 scene_1 和 scene_2

第二步：身份与风格分析 (Identity & Style Analysis)
- 从 scene_1 中识别并总结可用于 ID 判断的关键视觉特征（如：面部比例、标志性外观、生物结构、服饰符号）
- 从 scene_2 中拆解核心风格要素（如：艺术流派、用色逻辑、材质质感、光影风格）
- 分析 scene_3 如何在主体身份层面与 scene_1 保持一致，同时在整体视觉风格上与 scene_2 对齐
- 若提供 style_trigger_words，content_trigger_words，仅在其对理解合成逻辑有帮助时进行参考，不可机械复述

第三步：指令提炼 (Instruction Synthesis)
- 提炼一条可指导模型完成该类“ID 保持 + 风格参考 + 新构图生成”任务的核心指令
- 指令需清晰区分“身份约束”与“风格约束”
- 不描述具体构图复刻行为

第四步：口语化简写 (Natural Prompting)
- 输出一条面向普通用户的、简洁自然的生成指令

[JSON 输出结构]

你必须输出纯粹的 JSON 格式，严禁任何额外解释性文字。

{
  "independent_captions": {
    "scene_1": "（详细描述 scene_1 中主体的身份相关视觉特征，如外观、结构、辨识度要素。不少于 50 字。）",
    "scene_1_en": "（Detailed objective description of identity-related visual traits in scene_1.）",
    "scene_2": "（详细描述 scene_2 的整体艺术风格，包括色彩、材质、笔触、渲染方式与氛围。不少于 50 字。）",
    "scene_2_en": "（Detailed description of the artistic style, texture, and visual language in scene_2.）",
    "scene_3": "（利用你的想象能力，在prompt的描述下，描述 scene_3 这张新生成的图像，作为一张独立作品进行描述。重点描述主体形象与整体风格效果，严禁提及任何参考来源。）",
    "scene_3_en": "（Use your imagination to describe the synthesized image scene_3 as a standalone artwork, without referencing other scenes.）"
  },
  "comparative_analysis": {
    "identity_consistency": "（分析 scene_3 在哪些关键视觉层面上与 scene_1 保持了主体身份一致性。）",
    "style_alignment": "（分析 scene_3 在整体视觉风格上如何与 scene_2 对齐，例如色彩体系、材质选择或艺术流派特征。）",
    "generation_logic_cot": "（逻辑推理：解释 scene_3 是如何在不复刻构图的前提下，同时满足身份约束与风格约束生成的。）"
  },
  "predicted_edit_type": "（必须填选：'Identity Consistent Generation with Style Reference'）",
  "training_output": {
    "primary_instruction_cn_123": "（核心指令：描述在生成新图像时，如何保持场景1中主体的身份特征，同时采用场景2的整体艺术风格进行重新创作。可在有意义时隐含 content_trigger_words， style_trigger_words 所暗示的生成倾向，但不可直接罗列。）",
    "primary_instruction_en_123": "（Formal instruction: 'Generate a new image that preserves the identity characteristics of scene_1 while rendering it entirely in the artistic style of scene_2, allowing for a newly composed pose and scene.'）",
    "sample_instruction_cn_123": "（口语化指令：'用场景2的风格，生成一张保持场景1这个角色感觉的新图。'）",
    "sample_instruction_en_123": "（Natural prompt: 'Create a new image of the same character from scene_1, but in the style of scene_2.'）"
  }
}
"""


def as_image_message(
    image: bytes | PIL.Image.Image | str,
    image_format: str = "WEBP",
    min_pixels: int | None = None,
    max_pixels: int | None = None,
):
    mime_type = f"image/{image_format.lower()}"
    m = {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime_type};base64,{image_to_base64(image, format=image_format)}"
        },
    }
    if min_pixels is not None:
        m["min_pixels"] = min_pixels
    if max_pixels is not None:
        m["max_pixels"] = max_pixels
    return m

def image_to_base64(image: bytes | PIL.Image.Image | str, format="PNG", quality=95):
    pil_image = None
    image_bytes = None
    if isinstance(image, str):
        with megfile.smart_open(image, "rb") as f:
            pil_image = PIL.Image.open(f).copy()

    if isinstance(image, PIL.Image.Image):
        pil_image = image

    if pil_image is not None:
        pil_image = pil_image.convert("RGB")
        buffered = BytesIO()
        if format.upper() == "JPEG":
            pil_image.save(buffered, format=format, quality=quality)
        else:
            pil_image.save(buffered, format=format)

        image_bytes = buffered.getvalue()

    if isinstance(image, bytes):
        image_bytes = image

    assert isinstance(image_bytes, bytes), f"got {type(image_bytes)}"
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    return image_base64


def describe_difference(scenes, text):
    client = OpenAI(api_key="EMPTY", base_url=OPENAI_BASE_URL, timeout=3600)

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        {
            "role": "user",
            "content": [
                *[as_image_message(scene , max_pixels=512 * 32 * 32) for scene in scenes],
                # as_image_message(source_image, max_pixels=512 * 32 * 32),
                # as_image_message(target_image, max_pixels=512 * 32 * 32),
                {
                    "type": "text",
                    "text": text,
                },
            ],
        },
    ]

    response = client.chat.completions.create(
        model=OPENAI_MODEL_NAME,
        messages=messages,  # pyright: ignore[reportArgumentType]
        max_tokens=2048,
        extra_body=dict(chat_template_kwargs=dict(add_vision_id=True)),
        timeout=60 * 15,
    )
    response_text = response.choices[0].message.content

    return response_text


def _extract_and_validate_json(text: str) -> str:
    """
    从响应文本中提取并验证JSON内容

    Args:
        text: 原始响应文本

    Returns:
        str: 有效的JSON字符串

    Raises:
        ValueError: 如果无法提取有效的JSON内容
    """
    if not text:
        raise ValueError("响应文本为空")

    # 首先尝试直接解析整个文本作为JSON
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    # 如果直接解析失败，尝试提取被```json```包裹的内容
    json_pattern = r"```json\s*(.*?)\s*```"
    match = re.search(json_pattern, text, re.DOTALL | re.IGNORECASE)

    if match:
        json_content = match.group(1).strip()
        try:
            # 验证提取的内容是否为有效JSON
            json.loads(json_content)
            return json_content
        except json.JSONDecodeError as e:
            raise ValueError(f"提取的JSON内容无效: {e}")

    # 如果都没有找到有效JSON，尝试查找其他可能的JSON模式
    # 查找以{开头，以}结尾的内容
    json_pattern = r"\{.*\}"
    match = re.search(json_pattern, text, re.DOTALL)

    if match:
        json_content = match.group(0).strip()
        try:
            # 验证提取的内容是否为有效JSON
            json.loads(json_content)
            return json_content
        except json.JSONDecodeError as e:
            raise ValueError(f"提取的JSON内容无效: {e}")

    # 如果所有方法都失败，抛出异常
    raise ValueError(f"无法从响应文本中提取有效的JSON内容。原始文本: {text[:200]}...")


def _smart_exists(path: str) -> bool:
    return megfile.smart_exists(path)


def _smart_read_json(path: str) -> Dict[str, Any]:
    if not _smart_exists(path):
        return {}
    with megfile.smart_open(path, "r") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _smart_write_json(path: str, data: Dict[str, Any]) -> None:
    dir_path = os.path.dirname(path) or "."
    if path.startswith(("s3://", "oss://")):
        megfile.smart_makedirs(dir_path, exist_ok=True)
    else:
        os.makedirs(dir_path, exist_ok=True)
    with megfile.smart_open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def _process_one(args: Tuple[str, str, str, int]) -> Tuple[str, Optional[str]]:
    base, key, prompt_text, max_retries = args
    cref = PIL.Image.open(os.path.join(base, "cref", f"{key}.png"))
    sref = PIL.Image.open(os.path.join(base, "sref", f"{key}.png"))
    retries = 0
    last_exception = None
    while retries < max_retries:
        try:
            response = describe_difference(
                [cref, sref],
                Template(PROMPT_WITH_INSTUCTION_CREF_SREF).render(prompt=prompt_text),
            )
            json_data = _extract_and_validate_json(response)
            json_data = json.loads(json_data)
            value = (
                json_data["training_output"]["primary_instruction_cn_123"]
                + ", "
                + json_data["independent_captions"]["scene_3"]
            )
            return key, value
        except Exception as e:
            retries += 1
            last_exception = e
    print(f"Failed for key {key} after {max_retries} retries, last error: {last_exception}")
    return key, None


import argparse
import concurrent.futures
import multiprocessing as mp

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/mnt/chengwei/sref_lora_cref_bmk")
    ap.add_argument("--prompts_json", default="")
    ap.add_argument("--output_json", default="")
    ap.add_argument("--num_procs", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max_retries", type=int, default=3)
    args = ap.parse_args()

    base = args.base
    prompts_json = args.prompts_json or os.path.join(base, "prompts.json")
    output_json = args.output_json or os.path.join(base, "prompt_output.json")

    meta = _smart_read_json(prompts_json)
    keys = list(meta.keys())
    result_meta = {} if args.overwrite else _smart_read_json(output_json)

    pending_keys = [key for key in keys if key not in result_meta]
    if pending_keys:
        worker_args = [(base, key, meta[key], args.max_retries) for key in pending_keys]
        num_procs = max(1, min(args.num_procs, len(worker_args)))
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=num_procs) as pool:
            for k, v in tqdm(pool.imap_unordered(_process_one, worker_args), total=len(worker_args)):
                if v is not None:
                    result_meta[k] = v

    # 保持未能获得结果的仍保留原始内容
    for key in keys:
        if key not in result_meta and key in meta:
            result_meta[key] = meta[key]

    _smart_write_json(output_json, result_meta)
