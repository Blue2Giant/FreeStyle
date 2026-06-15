import itertools
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np
import torch

from vgo.data.vaultloader import TrainingSequence
from vgo.models.modules.distributed_ops import broadcast_tensors


def _current_device() -> torch.device:
    if hasattr(torch, "npu") and torch.npu.is_available():  # type: ignore[attr-defined]
        return torch.device("npu", torch.npu.current_device())  # type: ignore[attr-defined]
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


@dataclass
class PackData:
    sequences: list[TrainingSequence]
    ref_images: list[list[torch.Tensor]] = field(init=False)
    target_images: list[torch.Tensor] = field(init=False)
    text: list[str] | None = None

    # 以下是不需要关心的内容
    _ref_images_flatten: torch.Tensor | None = None
    _target_image_flatten: torch.Tensor | None = None
    data_track_info: Any | None = None

    def __post_init__(self):
        if self._ref_images_flatten is not None:
            assert self._target_image_flatten is not None
            assert self.text is not None
            self.update_images_list()
            return

        self.ref_images = []
        self.target_images = []

        for seq in self.sequences:
            # 只有一个目标图
            assert sum(seq.require_loss) == 1
            # 最后一个是目标图
            assert seq.require_loss[-1] == 1
            assert TrainingSequence.split_message(seq.message_formatter)[-1][0] == "image"

        self.text = ["".join(seq.text) for seq in self.sequences]

        _ref_images = [image.flatten() for seq in self.sequences for image in seq.ref_images[:-1]]
        if len(_ref_images) > 0:
            self._ref_images_flatten = torch.cat(
                [image.flatten() for seq in self.sequences for image in seq.ref_images[:-1]]
            ).float()
        else:
            self._ref_images_flatten = torch.Tensor([]).float()
        self._target_image_flatten = torch.cat([seq.image[-1].flatten() for seq in self.sequences]).float()

        self.update_images_list()

    def update_images_list(self):
        assert self._ref_images_flatten is not None
        assert self._target_image_flatten is not None

        # target_image: list[torch.Tensor] = []
        target_image_size_list = [seq.image[-1].shape[-2:] for seq in self.sequences]
        target_image_pixels = torch.cumsum(torch.tensor([x[0] * x[1] * 3 for x in target_image_size_list]), dim=0)[:-1]
        self.target_images = [
            x.reshape(3, h, w)
            for x, (h, w) in zip(
                self._target_image_flatten.tensor_split(target_image_pixels, dim=0), target_image_size_list
            )
        ]

        ref_images_size_list = [image.shape[-2:] for seq in self.sequences for image in seq.ref_images[:-1]]
        if len(ref_images_size_list) > 0:
            ref_images_pixels = torch.cumsum(torch.tensor([x[0] * x[1] * 3 for x in ref_images_size_list]), dim=0)[:-1]
            ref_images = [
                x.reshape(3, h, w)
                for x, (h, w) in zip(
                    self._ref_images_flatten.tensor_split(ref_images_pixels, dim=0), ref_images_size_list
                )
            ]
            ref_images_num = np.cumsum([0, *[len(seq.ref_images) - 1 for seq in self.sequences]])
            self.ref_images = [ref_images[start:end] for start, end in itertools.pairwise(ref_images_num)]
        else:
            self.ref_images = [[]] * len(self.sequences)

    def pin_memory(self):
        for key, values in self.__dict__.items():
            if isinstance(values, torch.Tensor):
                self.__dict__[key] = values.pin_memory()
        self.update_images_list()
        return self

    def cuda(self):
        device = _current_device()
        for key, values in self.__dict__.items():
            if isinstance(values, torch.Tensor):
                self.__dict__[key] = values.to(device=device, non_blocking=True)
        self.update_images_list()
        return self

    @classmethod
    def sync_tp(cls, obj_from_source, device_mesh):
        if device_mesh.size() == 1:
            return obj_from_source
        _dict_to_broad_cast: dict | None = obj_from_source.__dict__ if device_mesh.get_local_rank() == 0 else None

        if _dict_to_broad_cast is not None:
            _dict_to_broad_cast = {k: v for k, v in _dict_to_broad_cast.items() if k != "data_track_info"}
            # torch.distributed.breakpoint()
            _dict_to_broad_cast["sequences"] = [x.model_dump() for x in _dict_to_broad_cast["sequences"]]
            for x in _dict_to_broad_cast["sequences"]:
                x["sequence_id"] = x["sequence_id"].bytes.hex()
            _dict_to_broad_cast.pop("ref_images")
            _dict_to_broad_cast.pop("target_images")

        out = broadcast_tensors(_dict_to_broad_cast, device_mesh=device_mesh)

        if device_mesh.get_local_rank() == 0:
            return obj_from_source
        else:
            for x in out["sequences"]:
                from uuid import UUID

                x["sequence_id"] = UUID(x["sequence_id"])
            out["sequences"] = [TrainingSequence(**x) for x in out["sequences"]]
            out: dict = out
            field_name_set = {k.name for k in fields(PackData) if k.init}
            to_be_pop_key = [k for k in out if k not in field_name_set]
            for k in to_be_pop_key:
                out.pop(k)
            obj: PackData = cls(**out)
            obj.update_images_list()
            return obj

    def to_dict(self):
        return self
