from __future__ import annotations

import os
from typing import Any

import torch

from vgo.utils.dist_utils import device_module, device_type


def is_npu() -> bool:
    return device_type == "npu"


def is_cuda() -> bool:
    return device_type == "cuda"


def current_device() -> int:
    if hasattr(device_module, "current_device"):
        return int(device_module.current_device())
    return 0


def empty_cache() -> None:
    if hasattr(device_module, "empty_cache"):
        device_module.empty_cache()


def synchronize(device: Any | None = None) -> None:
    if hasattr(device_module, "synchronize"):
        if device is None:
            device_module.synchronize()
        else:
            device_module.synchronize(device)


def manual_seed(seed: int) -> None:
    torch.manual_seed(seed=seed)
    if is_cuda() and hasattr(torch.cuda, "manual_seed"):
        torch.cuda.manual_seed(seed=seed)
    elif is_npu() and hasattr(torch.npu, "manual_seed"):
        torch.npu.manual_seed(seed=seed)

    os.environ["PYTHONHASHSEED"] = str(seed % 2**32)


__all__ = [
    "device_type",
    "device_module",
    "is_npu",
    "is_cuda",
    "current_device",
    "empty_cache",
    "synchronize",
    "manual_seed",
]
