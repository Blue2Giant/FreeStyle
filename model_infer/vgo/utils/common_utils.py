import gc
import os
import random
import time
from collections import namedtuple
from collections.abc import Iterator, Mapping

import megfile
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from safetensors.torch import load_file
from torch._utils import _get_device_module


def _unwrap_state_dict(state_dict):
    if isinstance(state_dict, Mapping):
        for key in ("state_dict", "model"):
            nested_state_dict = state_dict.get(key)
            if isinstance(nested_state_dict, Mapping):
                return nested_state_dict
    return state_dict


def _log_key_list(prefix: str, keys: list[str], limit: int = 20):
    if not keys:
        return

    shown_keys = keys[:limit]
    suffix = ""
    if len(keys) > limit:
        suffix = f"\n\t... ({len(keys) - limit} more)"
    logger.info(f"{prefix} ({len(keys)}):\n\t" + "\n\t".join(shown_keys) + suffix)


def load_state_dict(model, ckpt_path, strict=False, assign=True):
    if megfile.SmartPath(ckpt_path).suffix == ".safetensors":
        state_dict = load_file(ckpt_path, "cpu")
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu")

    state_dict = _unwrap_state_dict(state_dict)

    if strict:
        missing, unexpected = model.load_state_dict(state_dict, strict=True, assign=assign)
        if len(missing) > 0 and len(unexpected) > 0:
            logger.info(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
            logger.info("\n" + "-" * 79 + "\n")
            logger.info(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
        elif len(missing) > 0:
            logger.info(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        elif len(unexpected) > 0:
            logger.info(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
        logger.success(f"Loading checkpoint from {ckpt_path}.")
        return model

    model_state_dict = model.state_dict()
    filtered_state_dict = {}
    mismatched_keys = []
    skipped_unexpected_keys = []

    for key, value in state_dict.items():
        if key not in model_state_dict:
            skipped_unexpected_keys.append(key)
            continue

        target_value = model_state_dict[key]
        if hasattr(value, "shape") and hasattr(target_value, "shape") and tuple(value.shape) != tuple(target_value.shape):
            mismatched_keys.append(f"{key}: ckpt{tuple(value.shape)} != model{tuple(target_value.shape)}")
            continue

        filtered_state_dict[key] = value

    missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False, assign=assign)
    if len(missing) > 0 and len(unexpected) > 0:
        logger.info(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        logger.info("\n" + "-" * 79 + "\n")
        logger.info(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        logger.info(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        logger.info(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    _log_key_list("Skipped unexpected checkpoint keys", skipped_unexpected_keys)
    _log_key_list("Skipped shape-mismatched checkpoint keys", mismatched_keys)
    logger.success(f"Loading checkpoint from {ckpt_path}.")
    return model


GPUMemStats = namedtuple(
    "GPUMemStats",
    [
        "max_active_gib",
        "max_active_pct",
        "max_reserved_gib",
        "max_reserved_pct",
        "num_alloc_retries",
        "num_ooms",
        "power_draw",
    ],
)


class GPUMemoryMonitor:
    """
    Class to monitor GPU memory usage
    """

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
        self.device_module = _get_device_module(self.device.type)

        if hasattr(self.device_module, "get_device_name"):
            self.device_name = self.device_module.get_device_name(self.device)
        else:
            self.device_name = str(self.device)

        if hasattr(self.device_module, "current_device"):
            self.device_index = int(self.device_module.current_device())
        else:
            self.device_index = 0

        if hasattr(self.device_module, "get_device_properties"):
            self.device_capacity = self.device_module.get_device_properties(self.device).total_memory
        else:
            self.device_capacity = 0
        self.device_capacity_gib = self._to_gib(self.device_capacity) if self.device_capacity else 0.0

        if hasattr(self.device_module, "reset_peak_memory_stats"):
            self.device_module.reset_peak_memory_stats(self.device)
        if hasattr(self.device_module, "empty_cache"):
            self.device_module.empty_cache()

    def _to_gib(self, memory_in_bytes):
        # NOTE: GiB (gibibyte) is 1024, vs GB is 1000
        _gib_in_bytes = 1024 * 1024 * 1024
        memory_in_gib = memory_in_bytes / _gib_in_bytes
        return memory_in_gib

    def _to_pct(self, memory):
        return 100 * memory / self.device_capacity

    def get_peak_stats(self):
        if not hasattr(self.device_module, "memory_stats") or self.device_capacity == 0:
            return GPUMemStats(0.0, 0.0, 0.0, 0.0, 0, 0, None)

        cuda_info = self.device_module.memory_stats(self.device)

        max_active = cuda_info.get("active_bytes.all.peak", 0)
        max_active_gib = self._to_gib(max_active)
        max_active_pct = self._to_pct(max_active)

        max_reserved = cuda_info.get("reserved_bytes.all.peak", 0)
        max_reserved_gib = self._to_gib(max_reserved)
        max_reserved_pct = self._to_pct(max_reserved)

        num_retries = int(cuda_info.get("num_alloc_retries", 0))
        num_ooms = int(cuda_info.get("num_ooms", 0))
        power_draw = self.device_module.power_draw() if hasattr(self.device_module, "power_draw") else None

        if num_retries > 0:
            logger.warning(f"{num_retries} CUDA memory allocation retries.")
        if num_ooms > 0:
            logger.warning(f"{num_ooms} CUDA OOM errors thrown.")

        return GPUMemStats(
            max_active_gib,
            max_active_pct,
            max_reserved_gib,
            max_reserved_pct,
            num_retries,
            num_ooms,
            power_draw,
        )

    def reset_peak_stats(self):
        if hasattr(self.device_module, "reset_peak_memory_stats"):
            self.device_module.reset_peak_memory_stats(self.device)
        if hasattr(self.device_module, "reset_accumulated_memory_stats"):
            self.device_module.reset_accumulated_memory_stats(self.device)

    def __str__(self):
        mem_stats = self.get_peak_stats()
        display_str = f"{self.device_name} ({self.device_index}): {self.device_capacity_gib} GiB capacity, "
        display_str += f"{mem_stats.max_reserved_gib} GiB peak, {mem_stats.max_reserved_pct}% peak"
        return f"{display_str}"


def log_params_count(model: nn.Module, k_lambda=None, v_lambda=None, prefix=""):
    # 典型的v_lambda： lambda v: v.requires_grad
    params_count = 0
    for k, v in model.named_parameters():
        if (k_lambda is None or k_lambda(k)) and (v_lambda is None or v_lambda(v)):
            params_count += v.numel()

    logger.info(prefix + f"{params_count / 1024 / 1024 / 1024:.3f}B")


# used to avoid stragglers in garbage collection
class GarbageCollection:
    def __init__(self, gc_freq: int = 1000):
        assert gc_freq > 0, "gc_freq must be a positive integer"
        self.gc_freq = gc_freq
        gc.disable()
        self.collect("Initial GC collection.")

    def run(self, step_count: int):
        if step_count > 1 and step_count % self.gc_freq == 0:
            self.collect("Peforming periodical GC collection.")

    @staticmethod
    def collect(reason: str = "", level=1):
        begin = time.monotonic()
        gc.collect(level)
        if reason:
            logger.info(f"[GC] {reason} {time.monotonic() - begin:.2f} seconds.")


def combine_list(input_list: list[list] | Iterator[list]):
    result = []
    for list_i in input_list:
        result.extend(list_i)
    return result


def convert_precision(model: nn.Module, dtype=torch.float16) -> nn.Module:
    converted_params = 0
    log_per_param = os.environ.get("VGO_LOG_CONVERT_PRECISION_PER_PARAM", "0") == "1"
    for name, module in model.named_modules():
        if "norm" in name:
            continue  # 不改这些层的精度
        for param_name, param in module.named_parameters(
            prefix=name, recurse=False
        ):  # 只改当前模块的 param，不递归子模块
            if log_per_param:
                logger.debug(f"Cast Param {param_name} to {dtype}")
            param.data = param.data.to(dtype)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(dtype)
            converted_params += 1
    logger.info(f"Converted {converted_params} parameter tensors to {dtype}.")
    return model


def is_namedtuple(data):
    """
    Checks if `data` is a `namedtuple` or not. Can have false positives, but only if a user is trying to mimic a
    `namedtuple` perfectly.
    """
    return isinstance(data, tuple) and hasattr(data, "_asdict") and hasattr(data, "_fields")


def honor_type(obj, generator):
    """
    Cast a generator to the same type as obj (list, tuple, or namedtuple)
    """
    # Some objects may not be able to instantiate from a generator directly
    if is_namedtuple(obj):
        return type(obj)(*list(generator))
    else:
        return type(obj)(generator)


def is_torch_tensor(tensor):
    return isinstance(tensor, torch.Tensor)


def recursively_apply(func, data, *args, test_type=is_torch_tensor, error_on_other_type=False, **kwargs):
    """
    Recursively apply a function on a data structure that is a nested list/tuple/dictionary of a given base type.

    Args:
        func (`callable`):
            The function to recursively apply.
        data (nested list/tuple/dictionary of `main_type`):
            The data on which to apply `func`
        *args:
            Positional arguments that will be passed to `func` when applied on the unpacked data.
        main_type (`type`, *optional*, defaults to `torch.Tensor`):
            The base type of the objects to which apply `func`.
        error_on_other_type (`bool`, *optional*, defaults to `False`):
            Whether to return an error or not if after unpacking `data`, we get on an object that is not of type
            `main_type`. If `False`, the function will leave objects of types different than `main_type` unchanged.
        **kwargs (additional keyword arguments, *optional*):
            Keyword arguments that will be passed to `func` when applied on the unpacked data.

    Returns:
        The same data structure as `data` with `func` applied to every object of type `main_type`.
    """
    if isinstance(data, (tuple, list)):
        return honor_type(
            data,
            (
                recursively_apply(
                    func, o, *args, test_type=test_type, error_on_other_type=error_on_other_type, **kwargs
                )
                for o in data
            ),
        )
    elif isinstance(data, Mapping):
        return type(data)(
            {
                k: recursively_apply(
                    func, v, *args, test_type=test_type, error_on_other_type=error_on_other_type, **kwargs
                )
                for k, v in data.items()
            }  # type: ignore
        )
    elif test_type(data):
        return func(data, *args, **kwargs)
    elif error_on_other_type:
        raise TypeError(
            f"Unsupported types ({type(data)}) passed to `{func.__name__}`. Only nested list/tuple/dicts of "
            f"objects that are valid for `{test_type.__name__}` should be passed."
        )
    return data


def listify(data):
    """
    Recursively finds tensors in a nested list/tuple/dictionary and converts them to a list of numbers.

    Args:
        data (nested list/tuple/dictionary of `torch.Tensor`): The data from which to convert to regular numbers.

    Returns:
        The same data structure as `data` with lists of numbers instead of `torch.Tensor`.
    """

    def _convert_to_list(tensor):
        tensor = tensor.detach().cpu()
        if tensor.dtype == torch.bfloat16:
            # As of Numpy 1.21.4, NumPy does not support bfloat16 (see
            # https://github.com/numpy/numpy/blob/a47ecdea856986cd60eabbd53265c2ca5916ad5d/doc/source/user/basics.types.rst ).  # noqa: E501
            # Until Numpy adds bfloat16, we must convert float32.
            tensor = tensor.to(torch.float32)
        return tensor.tolist()

    return recursively_apply(_convert_to_list, data)


def get_commit_id_from_git_dir(repo_path=".") -> str | None:
    git_dir = os.path.join(repo_path, ".git")
    head_path = os.path.join(git_dir, "HEAD")

    if not os.path.exists(head_path):
        return None

    with open(head_path) as f:
        ref = f.read().strip()

    if ref.startswith("ref:"):
        # HEAD 指向一个分支，如 "ref: refs/heads/main"
        ref_path = os.path.join(git_dir, ref[5:])
        if os.path.exists(ref_path):
            with open(ref_path) as f:
                return f.read().strip()
    else:
        # detached HEAD，HEAD 文件中就是 commit id
        return ref

    return None


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed(seed=seed)


DETERMINISTIC_MODE = os.getenv("DETERMINISTIC_MODE", None) == "1"
