import os

# os.environ["no_proxy"] = "stepcast-router.shai-core"

import argparse
import base64
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Any, Dict, Optional, Tuple

import megfile
import PIL.Image
from openai import OpenAI
from tqdm import tqdm


import re

DEFAULT_BASE = "/mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content"
DEFAULT_SERVICE_SPEC = "qwen35-27b@http://stepcast-router.shai-core:9200/v1"
DEFAULT_OUTPUT_NAME = "prompt_output.json"
DEFAULT_STRUCTURED_OUTPUT_NAME = "recaption_structured.json"
TASK_TYPE_STYLE_TRANSFER = "style_transfer"
TASK_TYPE_IDENTITY_STYLE = "identity_style"

PROMPT_WITH_INSTUCTION_CREF_SREF_STYLE_TRANSFER = """

[角色定位] 你是一位顶级的 AI 图像分析与风格迁移标注专家。你的核心任务是解析2张图片：

输入的场景提示词是 : {{prompt}}

scene_1 (Content Image): 提供物体、构图和结构的“底图”。
scene_2 (Style Image): 提供艺术风格、笔触、色调、材质的“风格参考图”。

补充约束：
- 这是一个只有两张参考图的 benchmark，不提供真实的 scene_3。
- 你必须假定 scene_3 与 scene_1 保持相同的核心内容、主体关系和构图结构，同时整体采用 scene_2 的风格进行重绘。
- 如果输入 prompt 很泛化或信息量不足，只把它当作辅助信号；优先依据 scene_1 的内容和 scene_2 的风格来推断 scene_3。

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


PROMPT_WITH_INSTUCTION_CREF_SREF_STYLE_TRANSFER_MINIMAL = """
[角色定位] 你是一位顶级的 AI 图像分析与风格迁移标注专家。

输入的场景提示词是 : {{prompt}}

scene_1 是内容图。
scene_2 是风格图。

补充约束：
- 这是一个只有两张参考图的 benchmark，不提供真实的 scene_3。
- 你必须假定 scene_3 与 scene_1 保持相同的核心内容、主体关系和构图结构，同时整体采用 scene_2 的风格进行重绘。
- 如果输入 prompt 很泛化或信息量不足，只把它当作辅助信号；优先依据 scene_1 的内容和 scene_2 的风格来推断 scene_3。

你只需要返回下面这个 JSON，对象必须合法，不能有任何额外解释：
{
  "training_output": {
    "sample_instruction_cn_123": "（简洁自然的中文用户指令，描述如何把 scene_1 画成 scene_2 的风格。）"
  },
  "independent_captions": {
    "scene_3_en": "（A standalone English description of the final stylized image scene_3.）"
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


def describe_difference(
    scenes,
    text: str,
    service_spec: str = DEFAULT_SERVICE_SPEC,
):
    model_name, base_url = parse_service_spec(service_spec)
    client = OpenAI(api_key="EMPTY", base_url=base_url, timeout=3600)

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
        model=model_name,
        messages=messages,  # pyright: ignore[reportArgumentType]
        max_tokens=2048,
        extra_body=dict(chat_template_kwargs=dict(add_vision_id=True)),
        timeout=60 * 15,
    )
    response_text = extract_message_content(response.choices[0].message.content)

    return response_text


def extract_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(text)
        return "\n".join(parts)
    raise TypeError(f"不支持的 message.content 类型: {type(content)}")


def _try_extract_valid_json(text: str) -> Optional[str]:
    if not text:
        return None

    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    json_pattern = r"```json\s*(.*?)\s*```"
    match = re.search(json_pattern, text, re.DOTALL | re.IGNORECASE)

    if match:
        json_content = match.group(1).strip()
        try:
            json.loads(json_content)
            return json_content
        except json.JSONDecodeError:
            pass

    json_pattern = r"\{.*\}"
    match = re.search(json_pattern, text, re.DOTALL)

    if match:
        json_content = match.group(0).strip()
        try:
            json.loads(json_content)
            return json_content
        except json.JSONDecodeError:
            pass

    return None


def repair_json_response(text: str, service_spec: str) -> str:
    model_name, base_url = parse_service_spec(service_spec)
    client = OpenAI(api_key="EMPTY", base_url=base_url, timeout=3600)
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You repair malformed JSON. Return one valid JSON object only. "
                        "Remove text outside the JSON object and preserve the existing fields."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Convert the following response into valid JSON. "
                        "Do not add markdown fences or explanations.\n\n"
                        f"{text}"
                    ),
                }
            ],
        },
    ]
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,  # pyright: ignore[reportArgumentType]
        max_tokens=4096,
        timeout=60 * 5,
    )
    return extract_message_content(response.choices[0].message.content)


def _extract_and_validate_json(
    text: str,
    service_spec: Optional[str] = None,
) -> str:
    json_content = _try_extract_valid_json(text)
    if json_content is not None:
        return json_content

    if service_spec:
        repaired_text = repair_json_response(text, service_spec)
        repaired_json = _try_extract_valid_json(repaired_text)
        if repaired_json is not None:
            return repaired_json

    raise ValueError(f"无法从响应文本中提取有效的JSON内容。原始文本: {text[:200]}...")


def parse_service_spec(service_spec: str) -> Tuple[str, str]:
    if "@" not in service_spec:
        raise ValueError(
            f"service_spec 必须是 '<model>@<base_url>' 格式，当前值为: {service_spec}"
        )
    model_name, base_url = service_spec.split("@", 1)
    model_name = model_name.strip()
    base_url = base_url.strip().rstrip("/")
    if not model_name or not base_url:
        raise ValueError(f"无效的 service_spec: {service_spec}")
    return model_name, base_url


def build_training_text(json_data: Dict[str, Any]) -> str:
    training_output = json_data.get("training_output") or {}
    independent_captions = json_data.get("independent_captions") or {}
    parts = [
        (
            training_output.get("primary_instruction_cn_123")
            or training_output.get("sample_instruction_cn_123")
            or ""
        ).strip(),
        (independent_captions.get("scene_3") or independent_captions.get("scene_3_en") or "").strip(),
    ]
    parts = [part for part in parts if part]
    if not parts:
        raise ValueError("模型返回 JSON 中缺少可用的 instruction/caption 字段")
    return ", ".join(parts)


def has_required_fields(json_data: Dict[str, Any], minimal_only: bool = False) -> bool:
    training_output = json_data.get("training_output") or {}
    independent_captions = json_data.get("independent_captions") or {}
    if minimal_only:
        return bool(
            training_output.get("sample_instruction_cn_123", "").strip()
            and independent_captions.get("scene_3_en", "").strip()
        )
    try:
        build_training_text(json_data)
        return True
    except Exception:
        return False


def get_template(task_type: str) -> str:
    template_map = {
        TASK_TYPE_STYLE_TRANSFER: PROMPT_WITH_INSTUCTION_CREF_SREF_STYLE_TRANSFER,
        TASK_TYPE_IDENTITY_STYLE: PROMPT_WITH_INSTUCTION_CREF_SREF,
    }
    if task_type not in template_map:
        raise ValueError(
            f"不支持的 task_type: {task_type}，可选值: {sorted(template_map.keys())}"
        )
    return template_map[task_type]


def render_prompt(template: str, prompt: str) -> str:
    return template.replace("{{prompt}}", prompt)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(data: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def process_one(
    base: str,
    key: str,
    prompt: str,
    task_type: str,
    service_spec: str,
    max_retries: int = 3,
    minimal_only: bool = False,
) -> Tuple[str, Optional[str], Optional[Dict[str, Any]]]:
    cref_path = os.path.join(base, "cref", f"{key}.png")
    sref_path = os.path.join(base, "sref", f"{key}.png")
    if not os.path.exists(cref_path):
        raise FileNotFoundError(f"缺少内容参考图: {cref_path}")
    if not os.path.exists(sref_path):
        raise FileNotFoundError(f"缺少风格参考图: {sref_path}")

    prompt_template = get_template(task_type)
    request_text = render_prompt(prompt_template, prompt)
    minimal_request_text = render_prompt(
        PROMPT_WITH_INSTUCTION_CREF_SREF_STYLE_TRANSFER_MINIMAL,
        prompt,
    )

    retries = 0
    last_exception = None
    if not minimal_only:
        while retries < max_retries:
            try:
                response = describe_difference(
                    [cref_path, sref_path],
                    request_text,
                    service_spec=service_spec,
                )
                json_data = json.loads(_extract_and_validate_json(response, service_spec=service_spec))
                value = build_training_text(json_data)
                return key, value, json_data
            except Exception as e:
                retries += 1
                last_exception = e

    minimal_retries = 0
    while minimal_retries < max_retries:
        try:
            response = describe_difference(
                [cref_path, sref_path],
                minimal_request_text,
                service_spec=service_spec,
            )
            json_data = json.loads(_extract_and_validate_json(response, service_spec=service_spec))
            value = build_training_text(json_data)
            return key, value, json_data
        except Exception as e:
            minimal_retries += 1
            last_exception = e

    print(
        f"Failed for key {key} after full+minimal retries ({max_retries}+{max_retries}), "
        f"last error: {last_exception}",
        flush=True,
    )
    return key, None, None


def main(
    base: str = DEFAULT_BASE,
    output_name: str = DEFAULT_OUTPUT_NAME,
    structured_output_name: str = DEFAULT_STRUCTURED_OUTPUT_NAME,
    service_spec: str = DEFAULT_SERVICE_SPEC,
    task_type: str = TASK_TYPE_STYLE_TRANSFER,
    max_workers: int = 32,
    max_retries: int = 3,
    limit: Optional[int] = None,
    keep_original_on_failure: bool = True,
    save_every: int = 50,
    resume: bool = True,
    minimal_only: bool = False,
):
    meta_path = os.path.join(base, "prompts.json")
    meta = load_json(meta_path)
    if not isinstance(meta, dict):
        raise ValueError(f"{meta_path} 必须是 dict[str, str] 格式")

    parse_service_spec(service_spec)
    all_keys = list(meta.keys())
    selected_keys = all_keys if limit is None else all_keys[:limit]
    output_path = os.path.join(base, output_name)
    partial_output_path = f"{output_path}.partial"
    structured_output_path = os.path.join(base, structured_output_name)
    structured_partial_output_path = f"{structured_output_path}.partial"
    structured_meta: Dict[str, Any] = {}
    if resume and os.path.exists(structured_partial_output_path):
        partial_structured_meta = load_json(structured_partial_output_path)
        if not isinstance(partial_structured_meta, dict):
            raise ValueError(f"{structured_partial_output_path} 必须是 dict[str, Any] 格式")
        structured_meta.update(
            {
                key: value
                for key, value in partial_structured_meta.items()
                if key in selected_keys
                and isinstance(value, dict)
                and has_required_fields(value, minimal_only=minimal_only)
            }
        )
        print(
            f"Resume from structured checkpoint: {structured_partial_output_path}, "
            f"cached={len(structured_meta)}",
            flush=True,
        )
    elif resume and os.path.exists(structured_output_path):
        existing_structured_meta = load_json(structured_output_path)
        if not isinstance(existing_structured_meta, dict):
            raise ValueError(f"{structured_output_path} 必须是 dict[str, Any] 格式")
        structured_meta.update(
            {
                key: value
                for key, value in existing_structured_meta.items()
                if key in selected_keys
                and isinstance(value, dict)
                and has_required_fields(value, minimal_only=minimal_only)
            }
        )
        print(
            f"Resume from structured output: {structured_output_path}, cached={len(structured_meta)}",
            flush=True,
        )

    result_meta: Dict[str, str] = {}
    for key, value in structured_meta.items():
        if isinstance(value, dict):
            try:
                result_meta[key] = build_training_text(value)
            except Exception:
                pass

    if resume and os.path.exists(partial_output_path):
        partial_meta = load_json(partial_output_path)
        if not isinstance(partial_meta, dict):
            raise ValueError(f"{partial_output_path} 必须是 dict[str, str] 格式")
        for key, value in partial_meta.items():
            if key in selected_keys and key not in result_meta:
                result_meta[key] = value
        print(
            f"Resume from prompt checkpoint: {partial_output_path}, cached={len(result_meta)}",
            flush=True,
        )

    keys = [key for key in selected_keys if key not in structured_meta]

    print(
        f"Start recaption: base={base}, samples={len(keys)}, task_type={task_type}, "
        f"service_spec={service_spec}, output_name={output_name}",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_one,
                base,
                key,
                meta[key],
                task_type,
                service_spec,
                max_retries,
                minimal_only,
            ): key
            for key in keys
        }
        for completed_count, future in enumerate(
            tqdm(as_completed(futures), total=len(futures)),
            start=1,
        ):
            key = futures[future]
            try:
                _, value, json_data = future.result()
                if value is not None:
                    result_meta[key] = value
                if json_data is not None:
                    structured_meta[key] = json_data
            except Exception as exc:
                print(f"{key} generated an exception: {exc}", flush=True)
            if save_every > 0 and completed_count % save_every == 0:
                dump_json(result_meta, partial_output_path)
                dump_json(structured_meta, structured_partial_output_path)
                print(
                    f"Checkpoint saved: prompts={len(result_meta)}, structured={len(structured_meta)}",
                    flush=True,
                )

    if keep_original_on_failure:
        for key in selected_keys:
            if key not in result_meta:
                result_meta[key] = meta[key]

    dump_json(result_meta, output_path)
    dump_json(structured_meta, structured_output_path)
    if os.path.exists(partial_output_path):
        os.remove(partial_output_path)
    if os.path.exists(structured_partial_output_path):
        os.remove(structured_partial_output_path)
    print(
        f"Recaption finished: prompts={len(result_meta)} / {len(selected_keys)} -> {output_path}; "
        f"structured={len(structured_meta)} -> {structured_output_path}",
        flush=True,
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recaption prompts for sref/cref benchmarks.")
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--output_name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--structured_output_name", default=DEFAULT_STRUCTURED_OUTPUT_NAME)
    parser.add_argument("--service_spec", default=DEFAULT_SERVICE_SPEC)
    parser.add_argument(
        "--task_type",
        default=TASK_TYPE_STYLE_TRANSFER,
        choices=[TASK_TYPE_STYLE_TRANSFER, TASK_TYPE_IDENTITY_STYLE],
    )
    parser.add_argument("--max_workers", type=int, default=32)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--keep_original_on_failure", type=int, choices=[0, 1], default=1)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--resume", type=int, choices=[0, 1], default=1)
    parser.add_argument("--minimal_only", type=int, choices=[0, 1], default=0)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(
        base=args.base,
        output_name=args.output_name,
        structured_output_name=args.structured_output_name,
        service_spec=args.service_spec,
        task_type=args.task_type,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
        limit=args.limit,
        keep_original_on_failure=bool(args.keep_original_on_failure),
        save_every=args.save_every,
        resume=bool(args.resume),
        minimal_only=bool(args.minimal_only),
    )
