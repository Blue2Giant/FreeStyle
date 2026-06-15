import math
from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor


def get_noise(num_samples: int, height: int, width: int, device: torch.device, dtype: torch.dtype, seed: int):
    return torch.randn(
        num_samples,
        16,
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )


def time_shift(mu: float, sigma: float, t: Tensor | np.ndarray):
    inverse_t = 1 / ((t == 0) * 1e-6 + t)  # avoid div 0 warning
    return math.exp(mu) / (math.exp(mu) + (inverse_t - 1) ** sigma)


def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    x1: float = 256,
    base_shift: float = 0.5,
    x2: float = 4096,
    max_shift: float = 1.15,
    shift: bool = True,
    align_to_diffusers: bool = False,
) -> list[float]:
    # extra step for zero
    timesteps = (
        np.linspace(1, 0, num_steps + 1).astype(np.float32)
        if align_to_diffusers
        else torch.linspace(1, 0, num_steps + 1)
    )

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(x1=x1, y1=base_shift, x2=x2, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)
    if align_to_diffusers:
        return timesteps
    else:
        return timesteps.tolist()
