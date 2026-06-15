from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image


def get_dimensions(img: Any) -> tuple[int, int, int]:
    """
    Minimal replacement for `torchvision.transforms.functional.get_dimensions`.

    Returns:
        (channels, height, width)
    """
    if isinstance(img, Image.Image):
        w, h = img.size
        if img.mode in ("RGB", "YCbCr"):
            c = 3
        elif img.mode in ("RGBA",):
            c = 4
        else:
            c = 1
        return c, h, w

    if torch.is_tensor(img):
        if img.ndim < 2:
            raise ValueError(f"Unsupported tensor shape for get_dimensions: {tuple(img.shape)}")
        if img.ndim == 2:
            h, w = img.shape
            return 1, int(h), int(w)
        c, h, w = img.shape[-3], img.shape[-2], img.shape[-1]
        return int(c), int(h), int(w)

    raise TypeError(f"Unsupported type for get_dimensions: {type(img)}")


def to_tensor(img: Image.Image) -> torch.Tensor:
    """
    Minimal replacement for `torchvision.transforms.functional.to_tensor`.

    - Converts PIL.Image to float32 torch tensor in range [0, 1]
    - Output shape: (C, H, W)
    """
    if not isinstance(img, Image.Image):
        raise TypeError(f"to_tensor expects PIL.Image, got {type(img)}")

    arr = np.array(img, dtype=np.uint8)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor.to(dtype=torch.float32).div_(255.0)


def to_pil_image(tensor: torch.Tensor) -> Image.Image:
    """
    Minimal replacement for `torchvision.transforms.functional.to_pil_image`.

    Accepts:
      - float tensors in [0, 1] or [-1, 1]
      - uint8 tensors in [0, 255]
    """
    if not torch.is_tensor(tensor):
        raise TypeError(f"to_pil_image expects torch.Tensor, got {type(tensor)}")

    t = tensor.detach().cpu()
    if t.ndim == 2:
        t = t.unsqueeze(0)

    if t.ndim != 3:
        raise ValueError(f"to_pil_image expects CHW tensor, got shape {tuple(t.shape)}")

    c, h, w = t.shape
    if c not in (1, 3, 4):
        raise ValueError(f"Unsupported channel count for to_pil_image: {c}")

    if t.dtype != torch.uint8:
        if t.min() < 0:
            t = (t + 1) / 2
        t = t.clamp(0, 1).mul(255).to(torch.uint8)

    arr = t.permute(1, 2, 0).contiguous().numpy()
    if c == 1:
        arr = arr[:, :, 0]
        return Image.fromarray(arr, mode="L")
    if c == 3:
        return Image.fromarray(arr, mode="RGB")
    return Image.fromarray(arr, mode="RGBA")
