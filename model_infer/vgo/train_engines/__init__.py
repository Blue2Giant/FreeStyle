from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol

import torch
from torch.distributed.device_mesh import DeviceMesh

from vgo.utils.common_utils import GarbageCollection
from vgo.utils.timer import Timer


@dataclass
class StepInfo:
    """Structured information returned from a single training step."""

    lr: float
    loss: float
    global_grad_norm: float | None
    scalar_metrics: dict[str, float] = field(default_factory=dict)
    images: dict[str, torch.Tensor] | None = None


class Engine(Protocol):
    gc_handler: GarbageCollection
    device_mesh: DeviceMesh

    def get_timer(self) -> dict[str, Timer]: ...

    def set_init_train_state(self, train_state): ...

    def save_data_record(self, save_folder: str): ...

    def load_checkpoint(self, load_path): ...

    def save_checkpoint(self, checkpoint_folder: str): ...

    def batch_generator(self) -> Iterator: ...

    def train_one_step(self, loader) -> StepInfo: ...

    def set_logdir(self, logdir, exp_id) -> None:
        """Set the logging directory for the engine.

        Args:
            logdir (str): The directory path where logs will be saved.
        """
        ...
