import os
import argparse
import base64
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Any, Dict, Optional, Tuple

import PIL.Image
from openai import OpenAI
from tqdm import tqdm


DEFAULT_DIR = "/Users/leolan/Documents/trae_projects/docs/assets/cref_sref_flux_lora_part1"
DEFAULT_SERVICE_SPEC = "qwen3vlw8a8@http://stepcloud-apisix-gateway-eval.i-stepfun.com/Qwen3-VL-235B-A22B-W8A8/v1"
DEFAULT_OUTPUT = "triplet_captions.json"

CAPTION_PROMPT = """
image_1 是主体参考图，image_2 是目标图。两张图中是同一个主体。

请先识别两张图中共同的主体（人物、角色、生物等），然后用英文（20-50词）描述 image_2 的画面。
以主体作为主语（用 the + 主体名词），描述它在 image_2 中的动作、姿态、场景。
不要提及风格、画风、色调。不要用"变化""转变"等对比性词汇。直接描述 image_2 画面即可。

只输出 JSON，无额外文字：
{"prompt": "（英文，20-50词）"}
"""


def image_to_base64(image_path: str, fmt: str = "WEBP", quality: int = 95) -> str:
    with open(image_path, "rb") as f:
        pil_image = PIL.Image.open(f).copy()
    pil_image = pil_image.convert("RGB")
    buf = BytesIO()
    if fmt.upper() == "JPEG":
        pil_image.save(buf, format=fmt, quality=quality)
    else:
        pil_image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def as_image_message(image_path: str, image_format: str = "WEBP"):
    mime_type = f"image/{image_format.lower()}"
    b64 = image_to_base64(image_path, fmt=image_format)
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{b64}"},
    }


def parse_service_spec(service_spec: str) -> Tuple[str, str]:
    if "@" not in service_spec:
        raise ValueError(f"service_spec 必须是 '<model>@<base_url>' 格式，当前值为: {service_spec}")
    model_name, base_url = service_spec.split("@", 1)
    return model_name.strip(), base_url.strip().rstrip("/")


def extract_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item.get("text", "") for item in content if isinstance(item, dict) and item.get("text")
        )
    raise TypeError(f"不支持的 message.content 类型: {type(content)}")


def try_extract_json(text: str) -> Optional[Dict]:
    if not text:
        return None
    for extract in [
        lambda t: t,
        lambda t: re.search(r"```json\s*(.*?)\s*```", t, re.DOTALL | re.IGNORECASE)
        and re.search(r"```json\s*(.*?)\s*```", t, re.DOTALL | re.IGNORECASE).group(1),
        lambda t: re.search(r"\{.*\}", t, re.DOTALL)
        and re.search(r"\{.*\}", t, re.DOTALL).group(0),
    ]:
        try:
            candidate = extract(text)
            if candidate:
                return json.loads(candidate.strip())
        except (json.JSONDecodeError, AttributeError):
            continue
    return None


def describe_triplet(
    cref_path: str,
    target_path: str,
    service_spec: str,
) -> str:
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
                as_image_message(cref_path),
                as_image_message(target_path),
                {"type": "text", "text": CAPTION_PROMPT},
            ],
        },
    ]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=2048,
        extra_body=dict(chat_template_kwargs=dict(add_vision_id=True)),
        timeout=60 * 15,
    )
    return extract_message_content(response.choices[0].message.content)


def discover_triplets(directory: str) -> Dict[str, Dict[str, str]]:
    triplets: Dict[str, Dict[str, str]] = {}
    for fname in os.listdir(directory):
        for suffix in ("-cref.jpg", "-sref.jpg", "-target.jpg"):
            if fname.endswith(suffix):
                uuid = fname[: -len(suffix)]
                role = suffix[1:-4]  # cref / sref / target
                triplets.setdefault(uuid, {})[role] = os.path.join(directory, fname)
                break
    complete = {
        uid: paths
        for uid, paths in triplets.items()
        if "cref" in paths and "target" in paths
    }
    return complete


def process_one(
    uuid: str,
    cref_path: str,
    target_path: str,
    service_spec: str,
    max_retries: int = 3,
) -> Tuple[str, Optional[str], Optional[Dict]]:
    last_err = None
    for _ in range(max_retries):
        try:
            raw = describe_triplet(cref_path, target_path, service_spec)
            data = try_extract_json(raw)
            if data and data.get("prompt", "").strip():
                return uuid, data["prompt"].strip(), data
            last_err = ValueError(f"JSON 缺少 prompt 字段或为空: {raw[:200]}")
        except Exception as e:
            last_err = e
    print(f"Failed for {uuid} after {max_retries} retries: {last_err}", flush=True)
    return uuid, None, None


def main(
    directory: str = DEFAULT_DIR,
    output: str = DEFAULT_OUTPUT,
    service_spec: str = DEFAULT_SERVICE_SPEC,
    max_workers: int = 16,
    max_retries: int = 3,
    limit: Optional[int] = None,
    resume: bool = True,
    save_every: int = 20,
):
    triplets = discover_triplets(directory)
    uuids = sorted(triplets.keys())
    if limit is not None:
        uuids = uuids[:limit]

    output_path = os.path.join(directory, output)
    partial_path = f"{output_path}.partial"
    detail_path = os.path.join(directory, "triplet_captions_detail.json")
    detail_partial_path = f"{detail_path}.partial"

    results: Dict[str, str] = {}
    details: Dict[str, Dict] = {}

    if resume:
        for p, store in [(partial_path, results), (output_path, results)]:
            if os.path.exists(p) and not store:
                with open(p, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                store.update({k: v for k, v in loaded.items() if k in uuids})
                print(f"Resumed {len(store)} entries from {p}", flush=True)
                break
        for p, store in [(detail_partial_path, details), (detail_path, details)]:
            if os.path.exists(p) and not store:
                with open(p, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                store.update({k: v for k, v in loaded.items() if k in uuids})
                break

    todo = [uid for uid in uuids if uid not in results]
    print(
        f"Triplets found: {len(uuids)}, cached: {len(results)}, todo: {len(todo)}, "
        f"service: {service_spec}",
        flush=True,
    )

    def _save():
        with open(partial_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        with open(detail_partial_path, "w", encoding="utf-8") as f:
            json.dump(details, f, ensure_ascii=False, indent=2)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_one,
                uid,
                triplets[uid]["cref"],
                triplets[uid]["target"],
                service_spec,
                max_retries,
            ): uid
            for uid in todo
        }
        for count, future in enumerate(tqdm(as_completed(futures), total=len(futures)), 1):
            uid = futures[future]
            try:
                _, prompt, data = future.result()
                if prompt:
                    results[uid] = prompt
                if data:
                    details[uid] = data
            except Exception as exc:
                print(f"{uid} exception: {exc}", flush=True)
            if save_every > 0 and count % save_every == 0:
                _save()
                print(f"Checkpoint: {len(results)}/{len(uuids)}", flush=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)
    for p in (partial_path, detail_partial_path):
        if os.path.exists(p):
            os.remove(p)

    print(
        f"Done: {len(results)}/{len(uuids)} -> {output_path}\n"
        f"Details -> {detail_path}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Caption cref->target triplets.")
    parser.add_argument("--dir", default=DEFAULT_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--service_spec", default=DEFAULT_SERVICE_SPEC)
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--save_every", type=int, default=20)
    args = parser.parse_args()
    main(
        directory=args.dir,
        output=args.output,
        service_spec=args.service_spec,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
        limit=args.limit,
        resume=not args.no_resume,
        save_every=args.save_every,
    )
