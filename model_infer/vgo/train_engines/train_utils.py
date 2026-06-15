from dataclasses import dataclass
from typing import Any

from torch.distributed.checkpoint.stateful import Stateful


@dataclass
class TrainState(Stateful):
    """Tracks the state of the training process."""

    global_step: int = 0
    num_seen_samples: int = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "global_step": self.global_step,
            "num_seen_samples": self.num_seen_samples,
        }

    def load_state_dict(self, state_dict):
        self.global_step = state_dict.get("global_step", 0)
        self.num_seen_samples = state_dict.get("num_seen_samples", 0)
