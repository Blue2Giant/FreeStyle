# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import itertools
import math
import os
from collections.abc import Callable, Generator, Iterable
from dataclasses import dataclass
from datetime import timedelta
from functools import cached_property, partial
from typing import Any, Union, cast

import torch
import torch.distributed
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
import torch.distributed.tensor
import torch.nn as nn
import torch.torch_version
from loguru import logger
from torch import distributed as dist
from torch._utils import _get_available_device_type, _get_device_module
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.distributed.tensor import (
    DTensor,
    Replicate,
    distribute_module,
)
from torch.distributed.tensor.parallel import ParallelStyle
from torch.distributed.tensor.placement_types import Placement

from vgo.utils.common_utils import combine_list


def get_device_info():
    device_type = _get_available_device_type()
    if device_type is None:
        device_type = "cuda"  # default device_type: cuda
    device_module = _get_device_module(device_type)  # default device_module:torch.cuda
    return device_type, device_module


device_type, device_module = get_device_info()


class WeightOnlyModule(nn.Module):
    def __init__(self, param: nn.Parameter):
        super().__init__()
        self.param = param

    def forward(self, scale=1.0):
        return self.param * scale


def fully_shard_w_optimizer_dtype(optimizer_dtype: torch.dtype | None = None) -> Callable:
    def customize_fully_shard_w_optimizer_dtype(module: nn.Module, **fsdp_kwargs):
        if optimizer_dtype is not None:
            module = module.to(optimizer_dtype)
        fully_shard(module=module, **fsdp_kwargs)

    return customize_fully_shard_w_optimizer_dtype


def set_fsdp_reduce_ops(module, factor: float, *, recurse: bool = True) -> None:
    module = cast(nn.Module, module)
    modules = list(module.modules()) if recurse else [module]
    for module in modules:
        if isinstance(module, FSDPModule):
            state = module._get_fsdp_state()
            if (fsdp_param_group := state._fsdp_param_group) is not None:
                mul_factor = 1.0 / float(factor)
                reduce_op = torch.distributed._make_nccl_premul_sum(mul_factor)
                fsdp_param_group.reduce_scatter_reduce_op = reduce_op


def get_parameters_w_grad(module) -> list[nn.Parameter]:
    # FIXME, 不确定 FSDP2 初始化时如果加上了 ignore params ，是否会造成遗漏
    if isinstance(module, FSDPModule):
        module = cast(nn.Module, module)
        modules = list(module.modules())
        parameters: list[nn.Parameter] = []
        for module in modules:
            if isinstance(module, FSDPModule):
                state = module._get_fsdp_state()
                if (fsdp_param_group := state._fsdp_param_group) is not None:
                    parameters += [x.sharded_param for x in fsdp_param_group.fsdp_params]
        return parameters

    module = cast(nn.Module, module)
    parameters: list[nn.Parameter] = list(module.parameters(recurse=False))
    for sub_module in module.children():
        parameters += get_parameters_w_grad(sub_module)
    return parameters


def get_modules_w_grad(module) -> tuple[bool, list[nn.Module]]:
    module = cast(nn.Module, module)
    contain_fixed = not all(x.requires_grad for x in module.parameters(recurse=False))

    valid_modules: list[nn.Module] = []
    children_contain_fixed = False
    for sub_module in module.children():
        sub_contain_fixed, modules = get_modules_w_grad(sub_module)
        if sub_contain_fixed:
            children_contain_fixed = True
        valid_modules.extend(modules)

    contain_fixed = contain_fixed or children_contain_fixed

    if len(valid_modules) == 0 and not contain_fixed:
        if len(list(module.parameters())) == 0:
            return False, []

    if not contain_fixed:
        return False, [module]
    else:
        return True, valid_modules


def reshard_module(module, *, recurse: bool = True) -> None:
    module = cast(nn.Module, module)
    modules = list(module.modules()) if recurse else [module]
    for module in modules:
        if isinstance(module, FSDPModule):
            module.reshard()


def unshard_module(module, *, recurse: bool = True) -> None:
    module = cast(nn.Module, module)
    modules = list(module.modules()) if recurse else [module]
    for module in modules:
        if isinstance(module, FSDPModule):
            module.unshard()


def set_unshard_in_backward(module, unshard_in_backward: bool, *, recurse: bool = True) -> None:
    module = cast(nn.Module, module)
    modules = list(module.modules()) if recurse else [module]
    for module in modules:
        if isinstance(module, FSDPModule):
            state = module._get_fsdp_state()
            if (fsdp_param_group := state._fsdp_param_group) is not None:
                fsdp_param_group.unshard_in_backward = unshard_in_backward


def set_reshard_after_backward(module, reshard_after_backward: bool, *, recurse: bool = True) -> None:
    module = cast(nn.Module, module)
    modules = list(module.modules()) if recurse else [module]
    for module in modules:
        if isinstance(module, FSDPModule):
            state = module._get_fsdp_state()
            if (fsdp_param_group := state._fsdp_param_group) is not None:
                fsdp_param_group.reshard_after_backward = reshard_after_backward


def set_requires_gradient_sync(self_module, requires_gradient_sync: bool, *, recurse: bool = True) -> None:
    """
    Sets if the module should sync gradients. This can be used to implement
    gradient accumulation *without communication*. For HSDP, this controls
    both reduce-scatter and all-reduce together. This is the equivalence of
    `no_sync` in FSDP1.

    Args:
        requires_gradient_sync (bool): Whether to reduce gradients for the
            module's parameters.
        recurse (bool): Whether to set for all FSDP submodules or just the
            passed-in module.
    """
    self_module = cast(nn.Module, self_module)
    modules = list(self_module.modules()) if recurse else [self_module]
    for module in modules:
        if isinstance(module, FSDPModule):
            state = module._get_fsdp_state()
            if fsdp_param_group := state._fsdp_param_group:
                fsdp_param_group.reduce_grads = requires_gradient_sync
                fsdp_param_group.all_reduce_grads = requires_gradient_sync


def _dist_reduce(
    x: torch.Tensor,
    reduceOp: str,
    mesh: DeviceMesh,
    extra_pg: dist.ProcessGroup | None = None,
) -> float:
    """Perform distributed reduction on a tensor.

    Args:
        x (torch.Tensor): Input tensor.
        reduceOp (str): Reduce operation to perform.
        mesh (DeviceMesh): Device mesh to use for reduction.
        extra_pg (dist.ProcessGroup, optional): Extra process group to use for reduction.
            Defaults to None. If provided, this all_reduce will be called for the extra
            process group, and then the result will be all_reduced for the mesh.
    """
    if isinstance(x, DTensor):
        # functional collectives do not support DTensor inputs
        x = x.full_tensor()

    if extra_pg is not None:
        x = funcol.all_reduce(x, reduceOp=reduceOp, group=extra_pg)

    assert x.numel() == 1  # required by `.item()`
    return funcol.all_reduce(x, reduceOp=reduceOp, group=mesh).item()


def dist_max(
    x: torch.Tensor,
    mesh: DeviceMesh,
    extra_pg: dist.ProcessGroup | None = None,
) -> float:
    return _dist_reduce(x, reduceOp=c10d.ReduceOp.MAX.name, mesh=mesh, extra_pg=extra_pg)


def dist_mean(
    x: torch.Tensor,
    mesh: DeviceMesh,
    extra_pg: dist.ProcessGroup | None = None,
) -> float:
    return _dist_reduce(x, reduceOp=c10d.ReduceOp.AVG.name, mesh=mesh, extra_pg=extra_pg)


def set_determinism(
    world_mesh: DeviceMesh | None,
    device: torch.device,
    seed: int | None = None,
    deterministic: bool = False,
    distinct_seed_mesh_dim: str = "pp",
) -> None:
    """
    Set the same DTensor manual seed for all dimensions in world mesh, but only different seeds
    across dimension denoted by `distinct_seed_mesh_dim`. An example use case is pipeline parallelism,
    where we want to have the same seed across SPMD groups, but different seeds across PP groups.

    Currently, does not set seeds for the CUDA RNG since TorchTitan always uses DTensor for SPMD parallelisms,
    and DTensor manages its own RNG tracker, but we could extend to support both if needed.

    Set Determinism flags for increased reproducibility with loss of performance.
    """
    if deterministic:
        logger.info("Deterministic algorithm enabled (expect perf degradation).")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # env var for deterministic CuBLAS
        # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    if not world_mesh:
        if seed is not None:
            torch.manual_seed(seed)
            os.environ["PYTHONHASHSEED"] = str(seed % 2**32)
            logger.debug(f"Single-process job using seed: {seed}")
        return

    # to ensure we can control which ranks have same or different seeds, all ranks agree on a starting seed.
    # if user provides one, we use this. Otherwise rank 0 rolls the dice and everyone else uses that.
    if seed is None:
        # Extract the seed for torch's main generator on rank 0 and standardizes on using that to build
        # seeds for unique SPMD groups
        seed_tensor = torch.get_rng_state()[:8].to(device)
        torch.distributed.broadcast(seed_tensor, src=0)
        seed = seed_tensor.to("cpu").view(torch.uint64).item()

    # Set distinct seed for each rank in mesh dimensions, with dimension name provdied by `distinct_seed_mesh_dim`
    # For PP + SPMD cases, we want to separate the world into the SPMD mesh and the PP mesh,
    # and choose a unique seed for each rank on the PP mesh.
    # TODO(jianiw): We could further extend this to support mutiple distinct dimensions instead of just one.
    if c10d.get_world_size() > 1 and distinct_seed_mesh_dim in world_mesh.mesh_dim_names:
        distinct_mesh = world_mesh[distinct_seed_mesh_dim]
        seed += distinct_mesh.get_local_rank()
        seed %= 2**64

        logger.debug(
            f"{distinct_seed_mesh_dim} rank {distinct_mesh.get_local_rank()}, \
                  Global rank {c10d.get_rank()} using seed: {seed}"
        )
        duplicate_seed_mesh = list(filter(lambda name: name != distinct_seed_mesh_dim, world_mesh.mesh_dim_names))
        duplicate_seed_mesh = world_mesh[duplicate_seed_mesh] if len(duplicate_seed_mesh) else None
    else:
        duplicate_seed_mesh = world_mesh
        logger.debug(f"Global Rank {c10d.get_rank()} using seed: {seed}")

    # The native RNGs and python RNG may not be important, except
    # for the 1-D PP case, but we seed them for consistency.
    torch.manual_seed(seed)
    # PYTHONHASHSEED can be a decimal number in the range [0, 2**32 - 1]
    os.environ["PYTHONHASHSEED"] = str(seed % 2**32)

    # As long as we are not in the 1-D (PP-only) case, we will have a seed to use for all ranks of the SPMD mesh.
    # IF PP is also used, this seed is unique per PP rank.
    if duplicate_seed_mesh and duplicate_seed_mesh.get_coordinate() is not None:
        torch.distributed.tensor._random.manual_seed(seed, duplicate_seed_mesh)


def create_context_parallel_ctx(
    cp_mesh: DeviceMesh,
    cp_buffers: list[torch.Tensor],
    cp_seq_dims: list[int],
    cp_no_restore_buffers: set[torch.Tensor],
    cp_rotate_method: str,
):
    try:
        from torch.distributed.tensor.experimental import context_parallel
        from torch.distributed.tensor.experimental._attention import set_rotate_method
    except ImportError:
        print(
            f"PyTorch version {torch.__version__} does not include the experimental "
            "Context Parallel API. Please update to a newer version."
        )

    set_rotate_method(cp_rotate_method)
    return context_parallel(
        cp_mesh,
        buffers=cp_buffers,
        buffer_seq_dims=cp_seq_dims,
        no_restore_buffers=cp_no_restore_buffers,
    )


def get_train_context(enable_loss_parallel: bool, enable_compiled_autograd: bool) -> Generator[None, None, None]:
    @contextlib.contextmanager
    def context(cp_context: Generator[None, None, None] | None = None):
        with contextlib.ExitStack() as stack:
            if enable_loss_parallel:
                stack.enter_context(torch.distributed.tensor.parallel.loss_parallel())

            if enable_compiled_autograd:
                stack.enter_context(torch._dynamo.utils.maybe_enable_compiled_autograd(True))

            if cp_context is not None:
                from torch.nn.attention import SDPBackend, sdpa_kernel

                stack.enter_context(
                    sdpa_kernel(
                        [
                            SDPBackend.FLASH_ATTENTION,
                            SDPBackend.EFFICIENT_ATTENTION,
                            SDPBackend.CUDNN_ATTENTION,
                        ]
                    )
                )
                stack.enter_context(cp_context)

            yield

    return context


def init_distributed(init_timeout_seconds, dump_folder, trace_buf_size, enable_cpu_offload):
    def _warn_overwrite_env(env, val):
        if env in os.environ:
            logger.warning(f"ENV[{env}] = {os.environ[env]} will be overridden to {val} based on job config")
        os.environ[env] = val

    def _get_distributed_backend():
        backend = "nccl"
        if device_type in torch.distributed.Backend.default_device_backend_map:
            backend = torch.distributed.Backend.default_device_backend_map.get(device_type)
        if enable_cpu_offload:
            backend = f"{device_type}:{backend},cpu:gloo"
        return backend

    if torch.__version__ >= torch.torch_version.TorchVersion("2.9"):
        TRACE_BUFFER_SIZE = "TORCH_FR_BUFFER_SIZE"
        TRACE_FILE = "TORCH_FR_DUMP_TEMP_FILE"
    else:
        TRACE_BUFFER_SIZE = "TORCH_NCCL_TRACE_BUFFER_SIZE"
        TRACE_FILE = "TORCH_NCCL_DEBUG_INFO_TEMP_FILE"

    DUMP_ON_TIMEOUT = "TORCH_NCCL_DUMP_ON_TIMEOUT"
    ASYNC_ERROR_HANDLING = "TORCH_NCCL_ASYNC_ERROR_HANDLING"
    SKIP_CLEANUP = "3"

    # FlightRecorder is incompatible with =1 mode where watchdog aborts work, must use =3 (skipcleanup)
    # to get flight recorder dumps. See https://github.com/pytorch/pytorch/issues/121055
    # This could be done only when flight recorder is enabled, but its nice to be consistent to avoid subtle
    # behavior differences
    _warn_overwrite_env(ASYNC_ERROR_HANDLING, SKIP_CLEANUP)

    # enable torch nccl flight recorder in the mode that would dump files if timeout is detected
    _warn_overwrite_env(TRACE_BUFFER_SIZE, str(trace_buf_size))
    if trace_buf_size > 0:
        # dump on timeout by default if trace buffer is enabled
        if dump_folder is not None:
            _warn_overwrite_env(DUMP_ON_TIMEOUT, "1")
            dump_dir = dump_folder
            prefix = "comm_trace"
            os.makedirs(dump_dir, exist_ok=True)
            _warn_overwrite_env(TRACE_FILE, f"{dump_dir}/{prefix}")

    torch.distributed.init_process_group(
        backend=_get_distributed_backend(),
        timeout=timedelta(seconds=init_timeout_seconds),
    )


def set_pg_timeouts(timeout, world_mesh):
    """
    Sets the timeout for all PGs in the provided mesh, and the default (world) group.

    Note: synchronizes via a barrier, before changing the timeouts. This is important, because
    otherwise you may face a race where the slow rank has not reached the timeout reduction point
    yet due to slow operations permitted under the old timeout value, but other faster ranks may
    start issuing collectives under the new shorter timeout and then immediately timeout.
    """
    logger.info(f"Synchronizing and adjusting timeout for all ProcessGroups to {timeout}")
    # Ensure that all the ranks have reached the point of setting the new timeout-
    # otherwise, some ranks may issue collectives with the new/shorter timeout and
    # those may time out, before other ranks have finished with initialization done
    # under the old/slow timeout.
    torch.distributed.barrier(device_ids=[device_module.current_device()])
    device_module.synchronize()

    groups = [world_mesh.get_group(mesh_dim) for mesh_dim in range(world_mesh.ndim)]

    # None represents the 'default' PG, not part of the mesh
    groups.append(None)
    for group in groups:
        torch.distributed.distributed_c10d._set_pg_timeout(timeout, group)


class NoParallel(ParallelStyle):
    def __init__(
        self,
        *,
        input_layouts: Placement | None = None,
        output_layouts: Placement | None = None,
        use_local_output: bool = True,
    ):
        super().__init__()
        self.input_layouts = input_layouts or Replicate()
        self.output_layouts = output_layouts or Replicate()
        self.desired_input_layouts = Replicate()
        self.use_local_output = use_local_output

    @staticmethod
    def _prepare_input_fn(input_layout, desired_input_layout, mod, inputs, device_mesh):
        # annotate module input placements/sharding with input_layouts
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(input_tensor, device_mesh, (input_layout,), run_check=False)

        if input_layout != desired_input_layout:
            input_tensor = input_tensor.redistribute(placements=(desired_input_layout,), async_op=True)
        return (input_tensor, *inputs[1:])

    @staticmethod
    def _prepare_output_fn(output_layout, use_local_output, mod, outputs, device_mesh):
        if outputs.placements != (output_layout,):
            outputs = outputs.redistribute(placements=(output_layout,), async_op=True)
        # back to local tensor
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            None,
            partial(self._prepare_input_fn, self.input_layouts, self.desired_input_layouts),  # type: ignore
            partial(self._prepare_output_fn, self.output_layouts, self.use_local_output),
        )


@torch.no_grad()
def clip_grad_norm_(  # noqa: C901
    parameters: torch.Tensor | Iterable[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
    pp_mesh: DeviceMesh | None = None,
    return_each_grad: bool = False,
    parameter_names: list[str] | None = None,
) -> tuple[torch.Tensor, Iterable[tuple[str, float]] | None]:
    """
    Clip the gradient norm of an iterable of parameters.

    Gradient norm clipping requires computing the gradient norm over the entire model.
    `torch.nn.utils.clip_grad_norm_` only computes gradient norm along DP/FSDP/TP dimensions.
    We need to manually reduce the gradient norm across PP stages.
    See https://github.com/pytorch/torchtitan/issues/596 for details.

    Args:
        parameters: an iterable of Tensors or a single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``
        pp_mesh: pipeline parallel device mesh. If not None, will reduce gradient norm across PP stages.

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).

    """
    if return_each_grad:
        assert parameter_names is not None

    parameters = [parameters] if isinstance(parameters, torch.Tensor) else list(parameters)

    # There are some changes on Pytorch 2.7, see https://github.com/pytorch/pytorch/commit/f859722f70d81eeb5191d12637a6ffaf9aa34fef#diff-1cba85327729d9d4d4cbfbade37a1476651bc5faa9a5ac2ab24fb69beed31fc2
    if torch.__version__ >= torch.torch_version.TorchVersion("2.7"):

        def get_tensor_device_mesh_key(x: torch.Tensor | DTensor):
            # 此时 x 为 (name, grad) 格式
            if return_each_grad:
                x = x[1]

            if isinstance(x, DTensor):
                return x.device_mesh.mesh_dim_names
            else:
                return tuple()

        if not return_each_grad:
            grads = [p.grad for p in parameters if p.grad is not None]
        else:
            grads = [(name, p.grad) for name, p in zip(parameter_names, parameters) if p.grad is not None]

        grads.sort(key=get_tensor_device_mesh_key)
        # 根据参数的 device mesh 进行分组
        grads_lists = [list(group) for key, group in itertools.groupby(grads, key=get_tensor_device_mesh_key)]

        # 计算每一个参数的 grad norm
        grads_name_list = None
        each_grad_norm_tuple_list = None
        if return_each_grad:
            grads_name_list = [[x[0] for x in grads_lists_i] for grads_lists_i in grads_lists]
            grads_lists = [[x[1] for x in grads_lists_i] for grads_lists_i in grads_lists]

        norm_tuple_list = [torch.ops.aten._foreach_norm(grads_list_i, 2) for grads_list_i in grads_lists]

        if return_each_grad:
            grads_name_list = combine_list(grads_name_list)  # type: ignore
            each_grad_norm_tuple_list = [torch.stack(norm_tuple) for norm_tuple in norm_tuple_list]
            each_grad_norm_tuple_list = [
                x.full_tensor() if isinstance(x, DTensor) else x for x in each_grad_norm_tuple_list
            ]
            each_grad_norm_tuple_list = torch.cat(each_grad_norm_tuple_list).float()
            each_grad_norm_tuple_list = each_grad_norm_tuple_list.tolist()

        # 计算每一组参数的 grad norm
        norm_list = [torch.linalg.vector_norm(torch.stack(norm_tuple), 2) for norm_tuple in norm_tuple_list]
        total_norm = [x.full_tensor() if isinstance(x, DTensor) else x for x in norm_list]
        total_norm = torch.linalg.vector_norm(torch.stack(total_norm), 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)

        for grads_list_i in grads_lists:
            torch._foreach_mul_(grads_list_i, clip_coef_clamped)
        if not return_each_grad:
            return total_norm, None
        else:
            return total_norm, zip(
                grads_name_list,  # type: ignore
                each_grad_norm_tuple_list,  # type: ignore
            )
    else:
        if return_each_grad:
            raise NotImplementedError("`return_each_grad` is not supported currently.")

        grads = [p.grad for p in parameters if p.grad is not None]
        total_norm = torch.nn.utils.get_total_norm(grads, norm_type, error_if_nonfinite, foreach)

        # If total_norm is a DTensor, the placements must be `torch.distributed._tensor.ops.math_ops._NormPartial`.
        # We can simply reduce the DTensor to get the total norm in this tensor's process group
        # and then convert it to a local tensor.
        # NOTE: It has two purposes:
        #       1. to make sure the total norm is computed correctly when PP is used (see below)
        #       2. to return a reduced total_norm tensor whose .item() would return the correct value
        if isinstance(total_norm, DTensor):
            # Will reach here if any non-PP parallelism is used.
            # If only using PP, total_norm will be a local tensor.

            total_norm = total_norm.full_tensor()

        if pp_mesh is not None:
            if math.isinf(norm_type):
                dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=pp_mesh.get_group())
            else:
                total_norm **= norm_type
                dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
                total_norm **= 1.0 / norm_type

        torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
        return total_norm, None


@dataclass
class Parallelism:
    data_parallel_replicate_degree: int = 1
    tensor_parallel_with_sequenc_parallel_degree: int = 1


@dataclass
class ParallelDims:
    dp: int
    tp_w_sp: int
    world_size: int

    def __post_init__(self):
        self._validate()

    def _validate(self):
        dp, tp_w_sp = (
            self.dp,
            self.tp_w_sp,
        )
        for d in (dp, tp_w_sp):
            assert d >= 1, "Parallelism degree should be >= 1"

        assert dp * tp_w_sp == self.world_size, (
            f"Invalid parallel dims: dp({dp}) * tp_w_sp({tp_w_sp})) != WORLD_SIZE({self.world_size})"
        )

    def build_mesh(self, device_type: str) -> DeviceMesh:
        dims = []
        names = []
        for d, name in zip(
            [self.dp, self.tp_w_sp],
            ["dp", "tp_w_sp"],
        ):
            dims.append(d)
            names.append(name)

        return self._build_mesh(device_type, dims, names, init_device_mesh)

    def _build_mesh(
        self,
        device_type: str,
        dims: list[int],
        names: list[str],
        init_device_mesh_fn: Callable,
    ) -> DeviceMesh:
        logger.info(f"Building {len(dims)}-D device mesh with {names}, {dims}")
        mesh = init_device_mesh_fn(device_type, dims, mesh_dim_names=names)
        return mesh

    @property
    def dp_enabled(self):
        return self.dp > 0

    @property
    def tp_w_sp_enabled(self):
        return self.tp_w_sp > 1

    @cached_property
    def non_data_parallel_size(self):
        return self.tp_w_sp


def average_sync_dict(  # noqa: C901
    data_dict: dict[str, Union[torch.Tensor, float]],
    device_mesh: DeviceMesh,  # 输入严格为 DeviceMesh
    device: torch.device,  # 张量操作的目标设备, 例如 torch.device(device_mesh.device_type)
) -> None:
    """
    简化版:使用打包方式平均并同步字典,其中 device_mesh 始终是 torch.distributed.DeviceMesh。
    字典中所有最终值都将变为 Python float。
    如果输入的张量包含多个元素,则在全局平均后,取其结果张量的均值。

    Args:
        data_dict (Dict[str, Union[torch.Tensor, float]]):
            需要同步的字典。
        device_mesh (DeviceMesh):
            PyTorch DeviceMesh 对象,定义了用于同步的进程。
        device (torch.device):
            张量操作的目标设备 (例如, torch.device(device_mesh.device_type))。
    """
    current_rank = 0
    global_world_size = 1  # 非分布式或预检查时的默认值
    is_distributed_env = dist.is_available() and dist.is_initialized()

    if is_distributed_env:
        current_rank = dist.get_rank()  # 全局 rank
        global_world_size = dist.get_world_size()  # 全局 world_size

    # 非分布式或单进程情况 (无需实际同步)
    if not is_distributed_env or global_world_size == 1:
        for key, value in data_dict.items():
            if isinstance(value, float):
                data_dict[key] = value
            elif isinstance(value, torch.Tensor):
                tensor_val = value.to(device=device, dtype=torch.float32)
                if tensor_val.numel() == 0:
                    data_dict[key] = float("nan")
                elif tensor_val.numel() == 1:
                    data_dict[key] = tensor_val.item()
                else:  # 多元素
                    data_dict[key] = tensor_val.mean().item()
            else:  # 其他类型
                data_dict[key] = float("nan")  # 保证输出为 float
        return

    # 从 DeviceMesh 获取 ProcessGroup 用于 all_reduce
    # global_world_size 用于检查 mesh 是否覆盖了所有进程
    process_group = device_mesh._flatten().get_group()

    tensors_to_pack: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []  # 存储键、原始类型信息、原始形状、元素数量

    if not data_dict:  # 空字典直接返回
        return

    # 1. 准备和打包数据
    for key, value in data_dict.items():
        item_meta = {"key": key, "num_elements": 0}
        current_tensor: torch.Tensor | None = None

        if isinstance(value, float):
            current_tensor = torch.tensor([value], dtype=torch.float32, device=device)
            item_meta["original_was_tensor"] = False
            item_meta["original_shape"] = torch.Size([1])  # 打包为1元素张量
            item_meta["num_elements"] = 1
        elif isinstance(value, torch.Tensor):
            current_tensor = value.to(device=device, dtype=torch.float32)
            item_meta["original_was_tensor"] = True
            item_meta["original_shape"] = current_tensor.shape
            item_meta["num_elements"] = current_tensor.numel()
            if item_meta["num_elements"] == 0:  # 处理空张量
                data_dict[key] = float("nan")
                continue  # 跳过打包
        else:  # 非 float 或 Tensor
            if current_rank == 0:  # 仅在一个 rank 上打印警告
                print(f"警告: 键 '{key}' 的值类型为 {type(value)},非 float 或 torch.Tensor。将赋为 NaN。")
            data_dict[key] = float("nan")
            continue  # 跳过打包

        tensors_to_pack.append(current_tensor.flatten())  # 扁平化后加入列表
        metadata.append(item_meta)

    if not tensors_to_pack:  # 如果所有项都被跳过 (例如,都是空张量或不支持的类型)
        return  # 字典中的值已在上面循环中更新为 NaN (如果适用)

    # 2. 拼接成一个大张量
    packed_tensor = torch.cat(tensors_to_pack)

    # 3. 执行 AllReduce
    # 如果 process_group 为 None, all_reduce 会使用默认的 WORLD 组。
    # 这是 _get_process_group_from_devicemesh_direct 在 mesh 覆盖 WORLD 或作为回退时的预期行为。
    handle = dist.all_reduce(packed_tensor, op=dist.ReduceOp.AVG, group=process_group, async_op=True)
    handle.wait()  # type: ignore # 等待操作完成

    # 4. 解包并更新字典
    current_pos = 0
    for item_meta_info in metadata:
        key = item_meta_info["key"]
        num_elements = item_meta_info["num_elements"]

        # 从同步后的大张量中提取对应部分
        synced_piece_flat = packed_tensor[current_pos : current_pos + num_elements]
        current_pos += num_elements

        final_float_value: float
        if item_meta_info["original_was_tensor"]:
            # 如果原本是张量,恢复其形状
            synced_tensor = synced_piece_flat.reshape(item_meta_info["original_shape"])
            if synced_tensor.numel() > 1:  # 多元素张量
                final_float_value = synced_tensor.mean().item()  # 计算均值
            elif synced_tensor.numel() == 1:  # 单元素张量
                final_float_value = synced_tensor.item()
            else:  # numel is 0 (空张量)
                final_float_value = float("nan")  # 理论上已被处理,作为保险
        else:  # 原本是 Python float (打包成了1元素张量)
            final_float_value = synced_piece_flat.item()

        data_dict[key] = final_float_value

    # 计算每个平均的 loss
    to_be_poped = []
    loss_mean_dict = {}
    for k, v in data_dict.items():
        if "_loss_sum_" in k:
            to_be_poped.append(k)
        if "_token_count_" in k:
            to_be_poped.append(k)
        if "_token_count_" in k and v > 0:
            # loss_mean = data_dict[k.replace("_token_count_", "_loss_sum_")] / data_dict[k]
            loss_mean = data_dict[k[:-13] + "_loss_sum_"] / v
            loss_mean_dict[k[:-14]] = loss_mean
    data_dict.update(loss_mean_dict)
    for k in to_be_poped:
        data_dict.pop(k)


__all__ = ["ParallelDims", "Parallelism", "clip_grad_norm_", "init_distributed"]
