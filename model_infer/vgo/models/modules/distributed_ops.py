import os
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed
import torch.distributed as dist
import torch.distributed.tensor
from loguru import logger
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.device_mesh import DeviceMesh, ProcessGroup

from vgo.utils.common_utils import combine_list


def _current_device() -> torch.device:
    if hasattr(torch, "npu") and torch.npu.is_available():  # type: ignore[attr-defined]
        return torch.device("npu", torch.npu.current_device())  # type: ignore[attr-defined]
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _is_on_current_device(tensor: torch.Tensor, current_device: torch.device) -> bool:
    if tensor.device.type != current_device.type:
        return False
    if current_device.type == "cpu":
        return True
    return tensor.device.index in (None, current_device.index)


def LocalToDTensor(x: torch.Tensor, seq_len: int, device_mesh: DeviceMesh):
    return torch.distributed.tensor.DTensor.from_local(
        local_tensor=x,
        device_mesh=device_mesh,
        placements=[Shard(dim=0)],
        shape=torch.Size(
            (
                seq_len,
                *x.shape[1:],
            )
        ),
        stride=x.stride(),
    )


def LocalReplicateToDTensorReplicate(x: torch.Tensor, device_mesh: DeviceMesh):
    return torch.distributed.tensor.DTensor.from_local(
        local_tensor=x, device_mesh=device_mesh, placements=[Replicate()], run_check=False
    )


def LocalReplicateToDTensorSP(x: torch.Tensor, device_mesh: DeviceMesh):
    x = torch.distributed.tensor.DTensor.from_local(
        local_tensor=x.contiguous(), device_mesh=device_mesh, placements=[Replicate()], run_check=False
    )
    return x.redistribute(device_mesh=device_mesh, placements=[Shard(dim=0)], async_op=True)


def complex_to_device(complex, device, non_blocking=False):
    if isinstance(complex, torch.Tensor):
        return complex.to(device, non_blocking=non_blocking)
    elif isinstance(complex, dict):
        return {k: complex_to_device(v, device, non_blocking=non_blocking) for k, v in complex.items()}
    elif isinstance(complex, (list, tuple)):
        return [complex_to_device(e, device, non_blocking=non_blocking) for e in complex]
    else:
        return complex


def collect_data_according_to_index(batch, selected_index: list[int]):
    output = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            if len(selected_index) > 0:
                output[k] = v[selected_index].contiguous()
            else:
                output[k] = None
        elif isinstance(v, list):
            if len(selected_index) > 0:
                output[k] = [v[i] for i in selected_index]
            else:
                output[k] = []
        elif isinstance(v, dict):
            output[k] = collect_data_according_to_index(v, selected_index)
    return output


def combine_gather_object_list(gather_object_list):
    outputs = {}

    # we assume that rank 0 always have valid values (not None)
    gather_object_list = [x for x in gather_object_list if x is not None]
    world_size = len(gather_object_list)
    for k, v in gather_object_list[0].items():
        if isinstance(v, torch.Tensor):
            tensor_list = [gather_object_list[rank_idx][k] for rank_idx in range(world_size)]
            outputs[k] = torch.cat(tensor_list, dim=0)
        elif isinstance(v, list):
            list_len = len(v)
            out_list = []
            for list_idx in range(list_len):
                tensor_list = [gather_object_list[rank_idx][k][list_idx] for rank_idx in range(world_size)]
                out_list.append(torch.cat(tensor_list, dim=0))
            outputs[k] = out_list
        elif isinstance(v, dict):
            sub_gather_object_list = [gather_object_list[rank_idx][k] for rank_idx in range(world_size)]
            outputs[k] = combine_gather_object_list(sub_gather_object_list)
        elif isinstance(v, tuple):
            tuple_len = len(v)
            out_tuple = []
            for tuple_idx in range(tuple_len):
                tensor_list = [gather_object_list[rank_idx][k][tuple_idx] for rank_idx in range(world_size)]
                out_tuple.append(torch.cat(tensor_list, dim=0))
            outputs[k] = tuple(out_tuple)
        else:
            raise ValueError(f"Unrecognize Value {k}: {v}")
    return outputs


# def gather_object_from_tensor_parallel_group(split_object, device):
#     gather_outputs_list = [None for _ in range(mpu.get_tensor_model_parallel_world_size())]
#     dist.all_gather_object(gather_outputs_list, split_object, group=mpu.get_tensor_model_parallel_group())
#     gather_outputs_list = complex_to_device(gather_outputs_list, device)
#     outputs = combine_gather_object_list(gather_outputs_list)

#     return outputs


class TensorParallelAllGatherGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, is_pad_split, device_mesh: DeviceMesh):
        ctx.device_mesh = device_mesh
        ctx.input_tensor_shape = input_tensor.shape
        ctx.is_pad_split = is_pad_split
        gather_output = [None for _ in range(device_mesh.size())]

        split_input_tensor = None
        if is_pad_split:  # noqa: SIM108
            split_input_tensor = None
        else:
            # some tensors shape[0] do not match the size of split index, e.g. var text input without padding
            # we what to support this kind of input tensors
            split_input_tensor = [input_tensor.contiguous(), input_tensor.shape[0]]

        dist.all_gather_object(gather_output, split_input_tensor, group=device_mesh.get_group())

        # some tensors shape[0] do not match the size of split index, e.g. var text input without padding
        # we what to support this kind of input tensors
        if not is_pad_split:
            gather_input_tensor_len_list = torch.tensor([x[1] for x in gather_output if x is not None])
            first_dim_seq_len = torch.tensor([0, *torch.cumsum(gather_input_tensor_len_list, dim=0).tolist()]).to(
                torch.int64
            )
            ctx.save_for_backward(first_dim_seq_len)

        current_device = _current_device()
        gather_output = [x[0].to(current_device) for x in gather_output if x is not None]
        gather_output = torch.cat(gather_output, dim=0)
        return gather_output

    @staticmethod
    def backward(ctx, grad_output):
        # grad output should be the correct gradients of the loss over the entire squence
        if not ctx.is_pad_split:
            (first_dim_seq_len,) = ctx.saved_tensors
            local_rank = ctx.device_mesh.get_local_rank()
            return (
                grad_output[first_dim_seq_len[local_rank] : first_dim_seq_len[local_rank + 1]],
                None,
                None,
            )
        else:
            return torch.zeros(*ctx.input_tensor_shape, device=grad_output.device, dtype=grad_output.dtype), None, None


def gather_combine_object_from_tensor_parallel_group(split_object, device_mesh: DeviceMesh, asyn_comm=False):
    if device_mesh.size() == 1:
        return split_object

    if os.environ.get("TORCH_NCCL_AVOID_RECORD_STREAMS", "0") == "1":
        if asyn_comm:
            logger.warning(
                "Async Comm in should not be used when `TORCH_NCCL_AVOID_RECORD_STREAMS=1`"
                "in `gather_combine_object_from_tensor_parallel_group`"
            )
        asyn_comm = False

    info_object = {}

    for k, v in split_object.items():
        if isinstance(v, torch.Tensor):
            info_object[k] = list(v.shape)

    full_info_object = [None for _ in range(device_mesh.size())]

    torch.distributed.all_gather_object(full_info_object, info_object, group=device_mesh.get_group())

    current_device = _current_device()

    out_object = {}
    for k, v in split_object.items():
        dim_0_len = sum(x[k][0] for x in full_info_object)  # type: ignore
        dtype = v.dtype
        tensor_shape = full_info_object[0][k][1:]  # type: ignore

        need_move = not _is_on_current_device(v, current_device)
        need_contig = not v.is_contiguous()

        v_local = v.to(current_device, dtype=dtype).contiguous() if need_move or need_contig else v

        out_tensor = torch.empty(
            (dim_0_len, *tensor_shape),  # type: ignore
            dtype=dtype,
            device=current_device,
        )
        out_tensor_list = list(out_tensor.split([x[k][0] for x in full_info_object], dim=0))  # type: ignore

        comm = torch.distributed.all_gather(
            out_tensor_list,
            v_local,
            group=device_mesh.get_group(),
            async_op=asyn_comm,
        )
        if asyn_comm:
            out_object[k] = (out_tensor_list, comm)
        else:
            out_object[k] = torch.cat(out_tensor_list, dim=0)

    if asyn_comm:
        for k, (out_tensor_list, comm) in out_object.items():
            comm.wait()
            out_object[k] = torch.cat(out_tensor_list, dim=0)

    return out_object


def gather_combine_object_from_tensor_parallel_group_w_grad(split_object, split_index, device_mesh: DeviceMesh):
    # currently, we should require that the split object contains Tensor, List[Tensor]
    # we require that the batch size is the first dimension

    is_pad_split = split_index is None

    out_object = {}
    for k, v in split_object.items():
        if isinstance(v, torch.Tensor) or v is None:
            out_object[k] = TensorParallelAllGatherGrad.apply(v, is_pad_split, device_mesh)
        elif isinstance(v, (list, tuple)):
            out_object[k] = []
            for v_i in v:
                out_object[k].append(TensorParallelAllGatherGrad.apply(v_i, is_pad_split, device_mesh))
        else:
            raise NotImplementedError(f"Unreconized Value Type: {type(v)}, Key: {k}")
    return out_object


class ReportTxt(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def forward(ctx, input_):
        logger.info(f"Txt forward: {input_.norm()}")
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        logger.info(f"Txt backward: {grad_output.norm()}")
        return grad_output


class ReportImage(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def forward(ctx, input_):
        logger.info(f"Image forward: {input_.norm()}")
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        logger.info(f"Image backward: {grad_output.norm()}")
        return grad_output


def check_tp_difference(input_: torch.Tensor, device_mesh: DeviceMesh, name=""):
    world_size = device_mesh.size()
    tensor_list = [torch.zeros_like(input_) for _ in range(world_size)]
    torch.distributed.all_gather(tensor_list, tensor=input_, group=device_mesh.get_group())
    tensor_list = torch.concatenate(tensor_list, dim=0).reshape(world_size, -1)
    # notice, this is different from torch.std
    std_ = (
        ((tensor_list.to(torch.float32) - tensor_list.to(torch.float32).mean(dim=0, keepdim=True)) ** 2).mean(dim=0)
        ** 0.5
    ).mean()
    name = f"of `{name}`"
    logger.info(f"Difference level {name}: {std_}, is same: {torch.allclose(std_, torch.zeros_like(std_))}")


class ReportDifference(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, name, device_mesh):
        ctx.op_name = name
        ctx.device_mesh = device_mesh
        logger.info(f"{ctx.op_name} Forward: ")
        check_tp_difference(input_, device_mesh, name=name)
        return input_

    @staticmethod
    def backward(ctx, grad_out):
        logger.info(f"{ctx.op_name} Backward: ")
        check_tp_difference(grad_out, ctx.device_mesh, name=ctx.op_name)
        return grad_out, None, None


def get_line_number():
    # sys._getframe(1) gets the frame object of the caller
    # f_lineno attribute holds the line number
    return sys._getframe(3).f_lineno


class ReportGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, name: str | None = None, quiet: bool = False):
        ctx.op_name = name
        ctx.line_number = get_line_number()
        ctx.quiet = quiet

        tensor_info = {
            "norm": input_.norm().item(),
            "shape": input_.shape,
            "dtype": input_.dtype,
            "DTensor": isinstance(input_, torch.distributed.tensor.DTensor),
            "local shape": input_.to_local().shape if isinstance(input_, torch.distributed.tensor.DTensor) else None,
        }
        import json

        tensor_info = json.dumps({k: str(v) for k, v in tensor_info.items()}, indent=4) if not ctx.quiet else ""

        if ctx.op_name is not None:
            logger.info(
                f"{ctx.op_name} {torch.distributed.get_rank()} LINE: {ctx.line_number} | Forward: {tensor_info}"
            )
        else:
            logger.info(f"{torch.distributed.get_rank()} LINE: {ctx.line_number} | Forward: {tensor_info}")
        return input_

    @staticmethod
    def backward(ctx, grad_out):
        tensor_info = {
            "norm": grad_out.norm().item(),
            "shape": grad_out.shape,
            "dtype": grad_out.dtype,
            "DTensor": isinstance(grad_out, torch.distributed.tensor.DTensor),
            "local shape": grad_out.to_local().shape
            if isinstance(grad_out, torch.distributed.tensor.DTensor)
            else None,
        }
        import json

        tensor_info = json.dumps({k: str(v) for k, v in tensor_info.items()}, indent=4) if not ctx.quiet else ""

        if ctx.op_name is not None:
            logger.info(
                f"{ctx.op_name} {torch.distributed.get_rank()} LINE: {ctx.line_number} | Backward: {tensor_info}"
            )
        else:
            logger.info(f"{torch.distributed.get_rank()} LINE: {ctx.line_number} | Backward: {tensor_info}")

        return grad_out, None, None


class AllReduceGradBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, device_mesh: DeviceMesh):
        ctx.device_mesh = device_mesh
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        dtype = grad_output.dtype
        # Read this https://docs.pytorch.org/docs/stable/notes/extending.html#:~:text=It%20is%20important%20NEVER%20to%20modifythese%20in%2Dplace.
        grad_output = grad_output.clone()

        grad_output = grad_output.float()

        torch.distributed.all_reduce(grad_output, torch.distributed.ReduceOp.SUM, group=ctx.device_mesh.get_group())
        grad_output = grad_output.to(dtype)
        return grad_output, None


# class AllReduceTensorParallel(torch.autograd.Function):
#     """Split the input and keep only the corresponding chuck to the rank."""

#     @staticmethod
#     def forward(ctx, input_):
#         torch.distributed.all_reduce(input_, group=mpu.get_tensor_model_parallel_group())
#         return input_ * (1 / mpu.get_tensor_model_parallel_world_size())

#     @staticmethod
#     def backward(ctx, grad_output):
#         torch.distributed.all_reduce(grad_output, group=mpu.get_tensor_model_parallel_group())
#         return grad_output * (1 / mpu.get_tensor_model_parallel_world_size())


def get_local_start_end_for_tensor_split(seq_len, rank, world_size):
    split_len = seq_len // world_size
    left = seq_len - split_len * world_size
    if rank < left:
        return (split_len + 1) * rank, (split_len + 1) * (rank + 1)
    else:
        prev = left * (split_len + 1)
        return split_len * (rank - left) + prev, split_len * (rank + 1 - left) + prev


def get_local_start_end_for_chunk(seq_len, rank, world_size):
    split_len = seq_len // world_size
    left = seq_len - split_len * world_size
    if left > 0:
        split_len = split_len + 1

    start_idx = min(rank * split_len, seq_len)
    end_idx = min((rank + 1) * split_len, seq_len)
    return start_idx, end_idx


class PrepareSequenceParallel(torch.autograd.Function):
    """Split input to SP"""

    @staticmethod
    def forward(ctx, input_x: torch.Tensor, sequence_dim: int, device_mesh: DeviceMesh):
        ctx.device_mesh = device_mesh
        ctx.input_shape = input_x.shape
        ctx.sequence_dim = sequence_dim
        ctx.device = input_x.device
        ctx.dtype = input_x.dtype

        rank = device_mesh.get_local_rank()
        world_size = device_mesh.size()
        assert world_size <= input_x.shape[sequence_dim]

        return input_x.tensor_split(world_size, dim=sequence_dim)[rank]

    @staticmethod
    def backward(ctx, grad_output):
        world_size = ctx.device_mesh.size()
        grad_input_shape_list = [list(ctx.input_shape) for _ in range(world_size)]
        for i, grad_input_shape_i in enumerate(grad_input_shape_list):
            x, y = get_local_start_end_for_tensor_split(ctx.input_shape[ctx.sequence_dim], i, world_size)
            grad_input_shape_i[ctx.sequence_dim] = y - x

        grad_input_list = [
            torch.empty(*split_grad_input_shape, device=ctx.device, dtype=ctx.dtype)
            for split_grad_input_shape in grad_input_shape_list
        ]

        dist.all_gather(grad_input_list, grad_output, group=ctx.device_mesh.get_group())

        grad_input = torch.cat(grad_input_list, dim=ctx.sequence_dim)

        return grad_input, None, None


class GatherSequenceParallel(torch.autograd.Function):
    """Split input to SP"""

    @staticmethod
    def forward(ctx, input_x: torch.Tensor, sequence_dim: int, device_mesh: DeviceMesh):
        ctx.device_mesh = device_mesh
        ctx.input_shape = input_x.shape
        ctx.sequence_dim = sequence_dim
        ctx.device = input_x.device
        ctx.dtype = input_x.dtype

        world_size = device_mesh.size()

        tensor_shape_list = [None for _ in range(world_size)]

        dist.all_gather_object(tensor_shape_list, input_x.shape, group=device_mesh.get_group())

        gather_tensor_list = [
            torch.empty(*tensor_shape_i, device=input_x.device, dtype=input_x.dtype)  # type: ignore
            for tensor_shape_i in tensor_shape_list
        ]
        input_x = input_x.contiguous()
        dist.all_gather(gather_tensor_list, input_x, group=device_mesh.get_group())

        return torch.cat(gather_tensor_list, dim=sequence_dim)

    @staticmethod
    def backward(ctx, grad_output):
        rank = ctx.device_mesh.get_local_rank()
        world_size = ctx.device_mesh.size()

        grad_input_list = grad_output.tensor_split(world_size, dim=ctx.sequence_dim)

        return grad_input_list[rank], None, None


class AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_x: torch.Tensor, device_mesh: DeviceMesh, scatter_dim: int, gather_dim: int):
        ctx.device_mesh = device_mesh
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.input_shape = input_x.shape
        rank = device_mesh.get_local_rank()
        world_size = device_mesh.size()
        input_x_list = list(input_x.tensor_split(world_size, dim=scatter_dim))
        input_x_list = [x.contiguous() for x in input_x_list]

        # FIXME: remove this comm ops
        # compute shape for gather dims
        tensor_shape_list = [None for _ in range(world_size)]
        dist.all_gather_object(tensor_shape_list, input_x.shape, group=device_mesh.get_group())

        output_x_shape_list = []
        for tensor_shape_i in tensor_shape_list:
            tensor_shape_i = list(tensor_shape_i)
            tensor_shape_i[scatter_dim] = input_x_list[rank].shape[scatter_dim]
            output_x_shape_list.append(tensor_shape_i)
        ctx.output_x_shape_list = output_x_shape_list

        output_x_list = [
            torch.empty(*output_x_shape_i, device=input_x.device, dtype=input_x.dtype)
            for output_x_shape_i in output_x_shape_list
        ]

        dist.all_to_all(output_x_list, input_x_list, group=device_mesh.get_group())
        return torch.cat(output_x_list, dim=gather_dim)

    @staticmethod
    def backward(ctx, grad_output):
        device_mesh = ctx.device_mesh
        scatter_dim = ctx.scatter_dim
        gather_dim = ctx.gather_dim
        world_size = device_mesh.size()

        grad_input_shape_list = [list(ctx.input_shape) for _ in range(world_size)]
        for i, grad_input_shape_i in enumerate(grad_input_shape_list):
            x, y = get_local_start_end_for_tensor_split(ctx.input_shape[ctx.scatter_dim], i, world_size)
            grad_input_shape_i[ctx.scatter_dim] = y - x

        grad_input_list = [
            torch.empty(*split_grad_input_shape, device=grad_output.device, dtype=grad_output.dtype)
            for split_grad_input_shape in grad_input_shape_list
        ]
        seq_lens = [x[gather_dim] for x in ctx.output_x_shape_list]
        grad_output_list: list[torch.Tensor] = list(grad_output.split(seq_lens, dim=gather_dim))
        grad_output_list = [x.contiguous() for x in grad_output_list]

        # we assume all split has same shape
        dist.all_to_all(grad_input_list, grad_output_list, group=device_mesh.get_group())

        return torch.cat(grad_input_list, dim=scatter_dim), None, None, None


class GridAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_x: torch.Tensor,
        device_mesh: DeviceMesh,
        scatter_dim: int,
        gather_dim: int,
        scatter_grid: list[int],
        gather_grid: list[int],
    ):
        ctx.device_mesh = device_mesh
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.scatter_grid = scatter_grid
        ctx.gather_grid = gather_grid

        rank = device_mesh.get_local_rank()
        world_size = device_mesh.size()

        input_x_list = list(input_x.split(scatter_grid, dim=scatter_dim))
        input_x_list = [x.contiguous() for x in input_x_list]

        output_x_shape_list = []
        for i in range(world_size):
            tensor_shape_i = list(input_x.shape)
            tensor_shape_i[gather_dim] = gather_grid[i]
            tensor_shape_i[scatter_dim] = scatter_grid[rank]
            output_x_shape_list.append(tensor_shape_i)

        output_x_list = [
            torch.empty(*output_x_shape_i, device=input_x.device, dtype=input_x.dtype)
            for output_x_shape_i in output_x_shape_list
        ]

        dist.all_to_all(output_x_list, input_x_list, group=device_mesh.get_group())

        out = torch.cat(output_x_list, dim=gather_dim)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        device_mesh = ctx.device_mesh
        scatter_dim = ctx.scatter_dim
        gather_dim = ctx.gather_dim
        scatter_grid = ctx.scatter_grid
        gather_grid = ctx.gather_grid
        rank = device_mesh.get_local_rank()
        world_size = device_mesh.size()

        grad_input_shape_list = [list(grad_output.shape) for _ in range(world_size)]
        for i, grad_input_shape_i in enumerate(grad_input_shape_list):
            grad_input_shape_i[scatter_dim] = scatter_grid[i]
            grad_input_shape_i[gather_dim] = gather_grid[rank]

        grad_input_list = [
            torch.empty(*split_grad_input_shape, device=grad_output.device, dtype=grad_output.dtype)
            for split_grad_input_shape in grad_input_shape_list
        ]
        grad_output_list: list[torch.Tensor] = list(grad_output.split(gather_grid, dim=gather_dim))
        grad_output_list = [x.contiguous() for x in grad_output_list]

        # we assume all split has same shape
        dist.all_to_all(grad_input_list, grad_output_list, group=device_mesh.get_group())

        return torch.cat(grad_input_list, dim=scatter_dim), None, None, None, None, None


class DataRecorder:
    def __init__(
        self,
        save_path,
        world_mesh: DeviceMesh,
        current_iterations: int = 0,
        collect_keys=None,
        silent=True,
    ):
        if collect_keys is None:
            self.collect_keys = {
                "original_size_as_tuple",
                "target_size_as_tuple",
                "__source__",
                "__url__",
                "__key__",
                "task_type",
            }
        else:
            self.collect_keys = collect_keys

        self.save_path = save_path
        self.current_iterations = current_iterations
        self.buffer = None
        self.silent = silent
        self.world_mesh = world_mesh
        self.tp_mesh = self.world_mesh["tp_w_sp"]
        self.dp_mesh = self.world_mesh["dp"]

    def collect_train_data_for_save(self, samples):  # noqa: C901
        if self.tp_mesh.get_local_rank() == 0:
            if isinstance(samples, dict):
                to_be_gather = {k: v for k, v in samples.items() if k in self.collect_keys}
            else:
                _key = [x._key for x in samples.data_track_info]
                _source = [x._source for x in samples.data_track_info]
                _url = [x._url for x in samples.data_track_info]
                to_be_gather = {
                    "__source__": _source,
                    "__key__": _key,
                    "__url__": _url,
                }
                _sequence_id = [x._sequence_id for x in samples.data_track_info]
                if _sequence_id[0] is not None:
                    _sequence_id = [x.bytes for x in _sequence_id]
                    to_be_gather["__sequence_id__"] = _sequence_id
                _loss = [x._loss for x in samples.data_track_info]
                if _loss[0] is not None:
                    to_be_gather["__loss__"] = _loss
                _timestep = [x._timestep for x in samples.data_track_info]
                if _timestep[0] is not None:
                    to_be_gather["__timestep__"] = _timestep
                _timestep = [x._target_width for x in samples.data_track_info]
                if _timestep[0] is not None:
                    to_be_gather["__target_width__"] = _timestep
                _timestep = [x._target_height for x in samples.data_track_info]
                if _timestep[0] is not None:
                    to_be_gather["__target_height__"] = _timestep
                _timestep = [x._choice_id for x in samples.data_track_info]
                if _timestep[0] is not None:
                    to_be_gather["__choice_id__"] = _timestep
            to_be_gather = complex_to_device(to_be_gather, "cpu")
        else:
            to_be_gather = None

        gather_list = [None for _ in range(self.dp_mesh.size())] if dist.get_rank() == 0 else None

        if self.dp_mesh.mesh[0].item() == 0:
            dist.gather_object(to_be_gather, gather_list, dst=0, group=self.dp_mesh.get_group())

        out = None
        if dist.get_rank() == 0:
            out = {}
            for data_item_i in gather_list:  # type: ignore
                if data_item_i is not None:
                    for k, v in data_item_i.items():  # type: ignore
                        if out.get(k) is not None:
                            out[k].append(v)
                        else:
                            out[k] = [v]
            out = {k: combine_list(v) for k, v in out.items()}
            out["global_step"] = [self.current_iterations for _ in range(len(out[next(iter(out.keys()))]))]

        return out

    def _save(self):
        if dist.get_rank() == 0:
            save_folder = self.save_path
            os.makedirs(save_folder, exist_ok=True)
            self.buffer.to_parquet(os.path.join(save_folder, f"{self.current_iterations}.parquet"))  # type: ignore

        self.buffer = None

    def __call__(self, samples):
        self.current_iterations += 1
        collect_data = self.collect_train_data_for_save(samples=samples)

        # FIXME:
        if dist.get_rank() == 0 and not self.silent:
            logger.info(f"{collect_data}")

        if dist.get_rank() == 0:
            if self.buffer is None:
                self.buffer = pd.DataFrame(collect_data)
            else:
                self.buffer = pd.concat([self.buffer, pd.DataFrame(collect_data)])  # type: ignore


def _get_tensor_info(tensors):
    if isinstance(tensors, dict):
        ret = {}
        for name, value in tensors.items():
            ret[name] = _get_tensor_info(value)
        return ret
    elif isinstance(tensors, (list, tuple)):
        ret = []
        for value in tensors:
            ret.append(_get_tensor_info(value))
        return ret
    elif isinstance(tensors, torch.Tensor):
        return ("tensor_info", tensors.shape, tensors.dtype)
    elif tensors is None or isinstance(tensors, (int, float, str, bool, np.ndarray)):
        return tensors
    else:
        raise ValueError(f"Unsupported type {type(tensors)}")


def _convert_tensor(tensors, rank, src_rank, inp_tensor_obj, group: ProcessGroup):
    if isinstance(tensors, dict):
        ret = {}
        for name, value in tensors.items():
            ret[name] = _convert_tensor(
                value, rank, src_rank, inp_tensor_obj[name] if rank == src_rank else None, group
            )
        return ret
    elif isinstance(tensors, tuple) and tensors[0] == "tensor_info":
        _, shape, dtype = tensors
        current_device = _current_device()
        if rank != src_rank:
            _tensor = torch.empty(shape, dtype=dtype, device=current_device)
        else:
            _tensor = inp_tensor_obj.to(current_device, non_blocking=True).contiguous()
        torch.distributed.broadcast(_tensor, src=src_rank, group=group)
        return _tensor
    elif isinstance(tensors, (list, tuple)):
        ret = []
        for idx, value in enumerate(tensors):
            ret.append(
                _convert_tensor(value, rank, src_rank, inp_tensor_obj[idx] if rank == src_rank else None, group)
            )
        return ret
    elif tensors is None or isinstance(tensors, (int, float, str, bool, np.ndarray)):
        return tensors
    else:
        raise ValueError(f"Unsupported type {type(tensors)}")


def broadcast_tensors(inp_tensor_obj, device_mesh: DeviceMesh) -> Any:
    src_rank = device_mesh["tp_w_sp"].mesh[0].item()
    rank = device_mesh["tp_w_sp"].get_rank()

    # transfer tensor meta info
    if rank == src_rank:
        info = _get_tensor_info(inp_tensor_obj)
        tensor_info = [info]
    else:
        tensor_info = [None]

    torch.distributed.broadcast_object_list(tensor_info, src=src_rank, group=device_mesh["tp_w_sp"].get_group())
    tensor_info = tensor_info[0]

    ret_tensor_dict = _convert_tensor(tensor_info, rank, src_rank, inp_tensor_obj, device_mesh["tp_w_sp"].get_group())

    return ret_tensor_dict
