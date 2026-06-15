import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import PIL.Image
import torch
from loguru import logger
from torchvision.transforms.functional import get_dimensions


@dataclass
class TargetImageShape:
    shapes: dict[int, list[tuple[int, int]]]
    weights: dict[int, float] | None = None
    init_warn: bool = True
    _random: random.Random | None = None
    is_dist: bool = False

    def init_random(self):
        # init distributed seed
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # FIXME: hard code seed
            self._random = random.Random(42)
        else:
            worker_id = worker_info.id
            global_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            self._random = random.Random(42 + (worker_info.num_workers * global_rank + worker_id))

    @property
    def random(self) -> random.Random:
        if not self.is_dist:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                self._random = None
            self.is_dist = True
        if self._random is None:
            self.init_random()
        return self._random

    def _target_base_size(
        self, height: int, width: int, target_basesizes_weights=None, allowed_base_sizes=None
    ) -> tuple[int, list[int]]:
        if allowed_base_sizes is None:
            allowed_base_sizes = []
            # 选出图片适合的基准尺寸
            for base in self.shapes:
                if height * width >= base * base * 0.8:
                    allowed_base_sizes.append(base)
        else:
            allowed_base_sizes = list(map(int, allowed_base_sizes))
            allowed_base_sizes = [
                base for base in allowed_base_sizes if height * width >= base * base * 0.8 and base in self.shapes
            ]
        # 如果没得选, 选择最小的基准尺寸
        if len(allowed_base_sizes) == 0:
            if self.init_warn:
                logger.warning(
                    f"Warning: No base size is suitable for {height}x{width}, use {min(self.shapes.keys())} instead"
                )
                self.init_warn = False
            allowed_base_sizes.append(min(self.shapes.keys()))

        if target_basesizes_weights is not None:
            weights = target_basesizes_weights
        else:
            weights = self.weights or dict.fromkeys(allowed_base_sizes, 1.0)

        weights = [weights[base] for base in allowed_base_sizes]
        return self.random.choices(allowed_base_sizes, weights=weights)[0], allowed_base_sizes

    def get_suitable_image_shape(
        self, image: torch.Tensor | PIL.Image.Image | tuple[int, int], base_size
    ) -> tuple[int, int]:
        if isinstance(image, (torch.Tensor, PIL.Image.Image)):
            _, height, width = get_dimensions(image)
        else:
            height, width = image
        image_shapes = self.shapes[base_size]
        t_h, t_w = image_shapes[0]
        min_aspect_ratio_error = abs(t_w / t_h - width / height)

        for h, w in image_shapes[1:]:
            error = abs(width / height - w / h)
            if error < min_aspect_ratio_error:
                min_aspect_ratio_error = error
                t_h, t_w = h, w
        return t_h, t_w

    def target_shape(
        self,
        image: torch.Tensor | PIL.Image.Image | tuple[int, int],
        target_basesizes_weights=None,
        allowed_base_sizes=None,
        return_base_size=False,
    ) -> tuple[int, int] | tuple[int, list[int], int, int]:
        if isinstance(image, (torch.Tensor, PIL.Image.Image)):
            _, height, width = get_dimensions(image)
        else:
            height, width = image

        base_size, allowed_base_sizes = self._target_base_size(
            height, width, target_basesizes_weights=target_basesizes_weights, allowed_base_sizes=allowed_base_sizes
        )
        t_h, t_w = self.get_suitable_image_shape(image, base_size=base_size)
        if return_base_size:
            return base_size, allowed_base_sizes, t_h, t_w
        else:
            return t_h, t_w

    def __call__(self, sample: dict) -> Any:
        target_basesizes_weights = None
        allowed_base_sizes = None
        if "__target_basesizes_weights__" in sample:
            target_basesizes_weights = {int(k): float(v) for k, v in sample["__target_basesizes_weights__"].items()}
        if "__allowed_base_sizes__" in sample:
            allowed_base_sizes = sample["__allowed_base_sizes__"]

        # FIXME: cref_mode
        if "cref_mode" in sample and sample["cref_mode"] and isinstance(sample["image"], list):
            target_shapes = [
                self.target_shape(
                    sample["image"][i],
                    target_basesizes_weights=target_basesizes_weights,
                    allowed_base_sizes=allowed_base_sizes,
                )
                for i in range(len(sample["image"]))
            ]
            return target_shapes
        elif isinstance(sample["image"], list):
            return self.target_shape(
                sample["image"][0],
                target_basesizes_weights=target_basesizes_weights,
                allowed_base_sizes=allowed_base_sizes,
            )
        else:
            return self.target_shape(
                sample["image"],
                target_basesizes_weights=target_basesizes_weights,
                allowed_base_sizes=allowed_base_sizes,
            )


def generate_multiple_sizes(image_size: int, step_size=32, range_scale=2 / 5):
    size_min = np.floor(np.sqrt(image_size * image_size * range_scale) / step_size).astype(np.int64) * step_size
    size_all = list(range(size_min, image_size, step_size))
    area = image_size * image_size
    aspect_size = []
    for size in size_all:
        if area % (size * step_size) == 0:
            aspect_size.append(
                (
                    size,
                    np.ceil(area / size / step_size).astype(np.int64) * step_size,
                )
            )
        else:
            aspect_size.append(
                (
                    size,
                    np.ceil(area / size / step_size).astype(np.int64) * step_size,
                )
            )
            aspect_size.append(
                (
                    size,
                    np.floor(area / size / step_size).astype(np.int64) * step_size,
                )
            )

    aspect_size = [*aspect_size, (image_size, image_size)]
    for h, w in aspect_size[::-1]:
        if h == w:
            continue
        aspect_size.append((w, h))
    return np.array(aspect_size).tolist()


def create_router_and_targetsizer(
    batch_size: int | Sequence[int] = 8,
    base_size: int | Sequence[int] = 512,
    base_size_weights: Sequence[int] | Sequence[float] | None = None,
    enable_multi_size: bool = False,
    step_size=32,
):
    batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
    base_size = [base_size] if isinstance(base_size, int) else base_size
    weights = dict.fromkeys(base_size, 1.0) if base_size_weights is None else dict(zip(base_size, base_size_weights))
    assert all(b in weights for b in base_size), f"{weights=}"

    shapes: dict[int, list[tuple[int, int]]] = dict()

    buffer_name_to_size = dict()
    for base, bs in zip(base_size, batch_size):
        shapes[base] = generate_multiple_sizes(base, step_size=step_size) if enable_multi_size else [(base, base)]
        for h, w in shapes[base]:
            buffer_name_to_size[f"{int(h)}x{int(w)}"] = bs

    return TargetImageShape(
        shapes=shapes,
        weights=weights,  # type: ignore
    )
