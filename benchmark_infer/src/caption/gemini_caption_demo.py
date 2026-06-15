#!/usr/bin/env python3
import os
import json
from typing import List

from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import base64

SYSTEM_PROMPT = """
You are a professional image-synthesis prompt generator.
Your task is to analyze the reference image and create one JSON object containing **exactly 10 distinct and diverse** image-generation scenarios based on the image content.
For each scenario, provide a detailed caption in both Chinese and English.

If it is a scene, you can add some characters or objects to interact within the scene, or change the perspective of the scene.
If there are multiple subjects in the picture, you can have them interact with each other or just focus on one of them for your imagination.
If there is a clear subject in the picture, then take this subject as the theme and imagine various actions, behaviors, positions, and environments in which it is located.

Important rules:
- **Diversity is key**: The 10 scenarios must be significantly different from each other (e.g., different actions, environments, lighting, styles, or perspectives).
- The synthesis must not look like pasted or cut-out elements. The result must appear as a newly rendered, seamless picture.
- The subject from the reference image must interact naturally with the new environment or objects.
- The final image must associate with the reference image but offer a fresh perspective or story.
- **Ignore style**: Just consider the content in the picture. If there is a style, don't take it into account.
- **Avoid overly exaggerated imagination**: The captions written should be as reasonable as possible and depict scenarios that could exist in daily life.

Output format:
Return only valid JSON, no extra explanation. The JSON must have this structure:
{
  "scenarios": [
    {
      "id": 1,
      "CN": "Chinese caption for scenario 1",
      "EN": "English caption for scenario 1"
    },
    ...
    {
      "id": 10,
      "CN": "Chinese caption for scenario 10",
      "EN": "English caption for scenario 10"
    }
  ]
}
You MUST give me exactly 10 diverse scenarios!
"""

USER_INSTRUCTION = """
You will be provided with a reference image.
Task:
Analyze the image and return exactly ONE JSON object with the key "scenarios". 
Its value must be an array of exactly 10 objects, where each object represents a unique imagination/scenario.
Each object must contain:
1. "id": integer ID (1-10)
2. "CN": Chinese instruction
3. "EN": English instruction

Ensure the 10 scenarios are highly diverse and creative.
"""


def image_to_bytes(path: str) -> bytes:
    img = Image.open(path).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def main():
    image_path = "/data/benchmark_metrics/assets/style.webp"

    client = genai.Client(
        http_options={
            "api_version": "v1alpha",
            "base_url": "https://YOUR_OPENAI_COMPAT_ENDPOINT/gemini",
        },
        api_key=os.getenv("GEMINI_API_KEY", ""),
    )

    img_bytes = image_to_bytes(image_path)
    image_part = types.Part.from_bytes(
        data=img_bytes,
        mime_type="image/jpeg",
    )

    text_part = types.Part.from_text(text=SYSTEM_PROMPT + "\n" + USER_INSTRUCTION)

    resp = client.models.generate_content(
        model="gemini-3-pro-native",
        contents=[image_part, text_part],
        config=types.GenerateContentConfig(
            max_output_tokens=4096,
        ),
    )

    raw = "".join(part.text or "" for part in resp.parts)
    raw_clean = raw.strip()
    if "```json" in raw_clean:
        raw_clean = raw_clean.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in raw_clean:
        raw_clean = raw_clean.split("```", 1)[1].split("```", 1)[0].strip()

    data = json.loads(raw_clean)
    scenarios: List[dict] = data.get("scenarios", [])

    captions_en = [s.get("EN", "") for s in scenarios]
    captions_zh = [s.get("CN", "") for s in scenarios]

    print("EN captions:")
    for i, c in enumerate(captions_en, 1):
        print(i, c)

    print("\nZH captions:")
    for i, c in enumerate(captions_zh, 1):
        print(i, c)


if __name__ == "__main__":
    main()
