import os
from importlib.util import find_spec
from dataclasses import dataclass, field

import torch
import torch.distributed.tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import Replicate

from vgo.models.modules.distributed_ops import LocalToDTensor, get_local_start_end_for_tensor_split

from .cat_split_seq_ops import cat_seq_optimized, split_seq_by_len_list_optimized, split_seq_optimized


def _current_device() -> torch.device:
    if hasattr(torch, "npu") and torch.npu.is_available():  # type: ignore[attr-defined]
        return torch.device("npu", torch.npu.current_device())  # type: ignore[attr-defined]
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


@dataclass
class VarLenConfig:
    csr_index: torch.Tensor
    seq_lens: torch.Tensor
    _dist_csr_index: torch.Tensor | None = None
    _dist_seq_lens: torch.Tensor | None = None
    _device_mesh: DeviceMesh | None = None
    _pt_seq_lens: torch.Tensor = field(default_factory=torch.Tensor)
    split_index: torch.Tensor = field(default_factory=torch.Tensor)
    reduce_tensor: torch.Tensor = field(default_factory=torch.Tensor)
    gather_index: torch.Tensor | None = None

    def __post_init__(self):
        self._pt_seq_lens = self.csr_index[1:] - self.csr_index[:-1]
        self.seq_lens = self.seq_lens.cpu().long()
        self.split_index = self.csr_index[1:-1].cpu().long()

        self.reduce_tensor = torch.eye(
            self.seq_lens.size(0), device=self.csr_index.device, dtype=torch.float32
        ).repeat_interleave(self._pt_seq_lens, dim=1)  # type: ignore

    def mean_seq(self, input_seq: torch.Tensor):
        eps = 1e-6

        seq_len = self._pt_seq_lens
        while seq_len.ndim < input_seq.ndim:
            seq_len = seq_len[..., None]
        return ((self.reduce_tensor @ input_seq.type_as(self.reduce_tensor)) / (seq_len + eps)).type_as(input_seq)

    def sum_seq(self, input_seq: torch.Tensor):
        return (self.reduce_tensor @ input_seq.type_as(self.reduce_tensor)).type_as(input_seq)

    @classmethod
    def from_seq_lens(cls, seq_lens, device):
        csr_index = torch.IntTensor([0, *seq_lens]).cumsum(dim=0).to(device, dtype=torch.int32)
        seq_lens = torch.tensor(seq_lens).to(dtype=torch.long)
        return cls(csr_index, seq_lens)

    @classmethod
    def from_csr_index(cls, csr_index):
        seq_lens = (csr_index[1:] - csr_index[:-1]).cpu().to(dtype=torch.long)
        return cls(csr_index, seq_lens)

    def set_gather_index(self, gather_index_len: int):
        self.gather_index = torch.arange(
            self.seq_lens.size(0), device=self.csr_index.device, dtype=torch.int64
        ).repeat_interleave(self._pt_seq_lens, dim=0)  # type: ignore

    def set_device_mesh(self, device_mesh: DeviceMesh):
        self._device_mesh = device_mesh
        self.gather_index = torch.distributed.tensor.DTensor.from_local(
            self.gather_index, device_mesh=device_mesh, placements=[Replicate()]
        )
        self.reduce_tensor = torch.distributed.tensor.DTensor.from_local(
            self.reduce_tensor, device_mesh=device_mesh, placements=[Replicate()]
        )
        self._pt_seq_lens = torch.distributed.tensor.DTensor.from_local(
            self._pt_seq_lens, device_mesh=device_mesh, placements=[Replicate()]
        )

    def unset_device_mesh(self):
        if self._device_mesh is not None:
            self._device_mesh = None
            self.gather_index = self.gather_index.full_tensor()  # type: ignore
            self.reduce_tensor = self.reduce_tensor.full_tensor()  # type: ignore
            self._pt_seq_lens = self._pt_seq_lens.full_tensor()  # type: ignore

    @property
    def dist_csr_index(self):
        if self._dist_csr_index is not None:
            return self._dist_csr_index
        assert self._device_mesh is not None, "Please call `set_device_mesh` before get dist_csr_index"

        self._dist_csr_index = chunk_csr_index(
            self.csr_index, self._device_mesh.size(), self._device_mesh.get_local_rank()
        )

        self._dist_seq_lens = self._dist_csr_index[1:] - self._dist_csr_index[:-1]
        return self._dist_csr_index

    def get_dist_csr_index(self, mode="chunk"):
        if self._dist_csr_index is not None:
            return self._dist_csr_index
        assert self._device_mesh is not None, "Please call `set_device_mesh` before get dist_csr_index"
        if mode == "chunk":
            self._dist_csr_index = chunk_csr_index(
                self.csr_index, self._device_mesh.size(), self._device_mesh.get_local_rank()
            )
        elif mode == "tensor_split":
            self._dist_csr_index = tensor_split_csr_index(
                self.csr_index, self._device_mesh.size(), self._device_mesh.get_local_rank()
            )
        else:
            raise ValueError(f"Unrecognized dist mode for csr index {mode=}")

        self._dist_seq_lens = self._dist_csr_index[1:] - self._dist_csr_index[:-1]
        return self._dist_csr_index

    @property
    def dist_seq_lens(self):
        if self._dist_seq_lens is not None:
            return self._dist_seq_lens
        _ = self.dist_csr_index
        return self._dist_seq_lens


def merge_varlen_seqs(first: VarLenConfig, second: VarLenConfig):
    if first.seq_lens.shape[0] != second.seq_lens.shape[0]:
        raise ValueError(
            f"first and second varlen sequences should be of equal length, got {len(first.seq_lens)} and {len(second.seq_lens)}"  # noqa: E501
        )

    merged_seq_lens = first.seq_lens + second.seq_lens
    csr_index = torch.tensor([0, *merged_seq_lens.tolist()]).to(first.csr_index.device, torch.int32).cumsum(dim=0)
    return VarLenConfig(csr_index=csr_index, seq_lens=merged_seq_lens)


# # FIXME: This should be checked as it create multiple concat kernel in the backward process
# def cat_seq(seqs_all: list[torch.Tensor], lens_all: list[torch.Tensor]):
#     split_all = [seqs.tensor_split(lens, dim=0) for seqs, lens in zip(seqs_all, lens_all)]
#     split_rearange_all = []
#     for split_each in zip(*split_all):
#         split_rearange_all += split_each
#     seqs_cat = torch.cat(split_rearange_all, dim=0)  # [b(n'+s), h//tp, 128]
#     return seqs_cat


# def split_seq(seqs: list[torch.Tensor], lens_all: list[torch.Tensor]):
#     seqs = torch.tensor_split(seqs, torch.stack(lens_all).T.flatten().cumsum(dim=0)[:-1])  # type: ignore
#     seqs_tensor_list = [torch.cat(seqs[0::2], dim=0), torch.cat(seqs[1::2], dim=0)]
#     return seqs_tensor_list


# def split_seq_by_len_list(seqs, lens_all):
#     def interleave_lists(list1, list2):
#         return [elem for pair in zip_longest(list1, list2) for elem in pair if elem is not None]

#     seqs = torch.split(seqs, interleave_lists(lens_all[0], lens_all[1]))
#     seqs_tensor_list = [torch.cat(seqs[0::2], dim=0), torch.cat(seqs[1::2], dim=0)]
#     return seqs_tensor_list


cat_seq = cat_seq_optimized
split_seq = split_seq_optimized
split_seq_by_len_list = split_seq_by_len_list_optimized


def _compile_varlen_op(fn):
    if os.environ.get("VGO_DISABLE_VARLEN_OPS_COMPILE", "0") == "1":
        return fn
    if os.environ.get("VGO_DISABLE_TORCH_COMPILE", "0") == "1":
        return fn
    if not hasattr(torch, "compile"):
        return fn
    if hasattr(torch, "npu") and torch.npu.is_available():  # type: ignore[attr-defined]
        return fn
    if not torch.cuda.is_available():
        return fn
    if find_spec("triton") is None:
        return fn
    return torch.compile(fn, dynamic=True, mode="max-autotune-no-cudagraphs")


class VARLEN_SCALE_SHIFT_REDUCE_TENSOR(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, shift, gather_index, reduce_tensor):
        scale = scale + 1
        scale_repeat = torch.gather(scale, dim=0, index=gather_index.unsqueeze(-1).expand(-1, scale.size(1)))
        shift_repeat = torch.gather(shift, dim=0, index=gather_index.unsqueeze(-1).expand(-1, shift.size(1)))
        ctx.save_for_backward(x, scale_repeat, reduce_tensor)
        return x * scale_repeat + shift_repeat

    @staticmethod
    def backward(ctx, grad_out):
        input_x, scale_repeat, reduce_tensor = ctx.saved_tensors

        grad_scale = grad_out * input_x
        grad_scale = torch.matmul(reduce_tensor, grad_scale.to(reduce_tensor.dtype))
        grad_scale = grad_scale.to(grad_out.dtype)

        grad_input = grad_out * scale_repeat

        grad_shift = torch.matmul(reduce_tensor, grad_out.to(reduce_tensor.dtype))
        grad_shift = grad_shift.to(grad_out.dtype)

        return grad_input, grad_scale, grad_shift, None, None


class VARLEN_GATE_REDUCE_TENSOR(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate, gather_index, reduce_tensor):
        gate_repeat = torch.gather(gate, dim=0, index=gather_index.unsqueeze(-1).expand(-1, gate.size(1)))
        ctx.save_for_backward(x, gate_repeat, reduce_tensor)
        return x * gate_repeat

    @staticmethod
    def backward(ctx, grad_out):
        input_x, gate_repeat, reduce_tensor = ctx.saved_tensors

        grad_gate = grad_out * input_x
        grad_gate = torch.matmul(reduce_tensor, grad_gate.to(reduce_tensor.dtype))
        grad_gate = grad_gate.to(grad_out.dtype)

        grad_input = grad_out * gate_repeat

        return grad_input, grad_gate, None, None


@_compile_varlen_op
def _varlen_scale_shift_reduce_tensor(x, scale, shift, gather_index, reduce_tensor):
    return VARLEN_SCALE_SHIFT_REDUCE_TENSOR.apply(x, scale, shift, gather_index, reduce_tensor)


def varlen_scale_shift_reduce_tensor(x, scale, shift, gather_index, reduce_tensor):
    dtype = x.dtype
    if dtype != torch.float32:
        x = x.to(torch.float32)

    if scale.dtype != torch.float32:
        scale = scale.to(torch.float32)
        shift = shift.to(torch.float32)

    out: torch.Tensor = _varlen_scale_shift_reduce_tensor(x, scale, shift, gather_index, reduce_tensor)

    if out.dtype != dtype:
        out = out.to(dtype)

    return out


@_compile_varlen_op
def _varlen_gate_reduce_tensor(x, gate, gather_index, reduce_tensor):
    return VARLEN_GATE_REDUCE_TENSOR.apply(x, gate, gather_index, reduce_tensor)


def varlen_gate_reduce_tensor(x, gate, gather_index, reduce_tensor):
    dtype = x.dtype

    if x.dtype != torch.float32:
        x = x.to(torch.float32)

    if gate.dtype != torch.float32:
        gate = gate.to(torch.float32)

    out: torch.Tensor = _varlen_gate_reduce_tensor(x, gate, gather_index, reduce_tensor)

    if out.dtype != dtype:
        out = out.to(dtype)  # type: ignore

    return out


def varlen_scale_shift(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    varlen_config: VarLenConfig,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    out = varlen_scale_shift_reduce_tensor(x, scale, shift, varlen_config.gather_index, varlen_config.reduce_tensor)  # type: ignore
    if out.dtype != out_dtype:
        out = out.to(out_dtype)
    return out


def dist_varlen_scale_shift(x: torch.distributed.tensor.DTensor, scale, shift, varlen_config: VarLenConfig):
    # x should be SP
    if isinstance(x, torch.distributed.tensor.DTensor):
        x = x.to_local()
    scale = scale + 1
    seq_len = (varlen_config.dist_csr_index[1:] - varlen_config.dist_csr_index[:-1]).tolist()
    split_x = torch.split(x, seq_len, dim=0)
    out = torch._foreach_mul(split_x, torch.chunk(scale, scale.shape[0], dim=0))
    out = torch._foreach_add(out, torch.chunk(shift, shift.shape[0], dim=0))
    out = torch.cat(out, dim=0)
    out = LocalToDTensor(out, varlen_config.csr_index[-1], varlen_config._device_mesh)
    return out


def varlen_gate(
    x: torch.Tensor,
    gate: torch.Tensor,
    varlen_config: VarLenConfig,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    out = varlen_gate_reduce_tensor(x, gate, varlen_config.gather_index, varlen_config.reduce_tensor)  # type: ignore
    if out.dtype != out_dtype:
        out = out.to(out_dtype)
    return out


def dist_varlen_gate(x, gate, varlen_config: VarLenConfig):
    if isinstance(x, torch.distributed.tensor.DTensor):
        x = x.to_local()
    seq_len = (varlen_config.dist_csr_index[1:] - varlen_config.dist_csr_index[:-1]).tolist()
    split_x = torch.split(x, seq_len, dim=0)
    out = torch._foreach_mul(split_x, torch.chunk(gate, gate.shape[0], dim=0))

    out = torch.cat(out, dim=0)
    out = LocalToDTensor(out, varlen_config.csr_index[-1], varlen_config._device_mesh)
    return out


def get_csr_index_from_mask(mask: torch.Tensor):
    seq_lens = mask.sum(dim=-1)
    zeros_shape = list(seq_lens.shape)
    zeros_shape[-1] = 1
    offsets = torch.cat((seq_lens.new_zeros(zeros_shape), seq_lens), -1).cumsum_(-1)
    return offsets.to(torch.int32)


def get_csr_index_from_batchsizeXtokennum(batch_size, token_num):
    return torch.arange(0, batch_size + 1, dtype=torch.int32, device=_current_device()) * token_num


def get_gather_index_from_mask(mask):
    assert mask.ndim == 2, "Mask should be 2D shape"
    len_i = mask.sum(dim=1).to(torch.int64)
    gather_index = torch.arange(len(len_i), device=mask.device, dtype=torch.int64).repeat_interleave(len_i, dim=0)
    return gather_index


def get_reduce_tensor_from_mask(mask):
    seq_lens = mask.sum(dim=1).to(torch.int32)
    return (
        torch.eye(len(seq_lens), dtype=torch.float32, device=mask.device)
        .repeat_interleave(seq_lens, dim=1)
        .contiguous()
    )


def chunk_csr_index_new(csr_index, chunk_count: int, chunk_rank: int):
    seq_lens = csr_index[-1].item()
    chunk_lens = (seq_lens + chunk_count - 1) // chunk_count

    chunk_start_idx = chunk_lens * chunk_rank
    chunk_end_idx = min(chunk_lens * (chunk_rank + 1), seq_lens)
    csr_index = torch.where(csr_index > chunk_start_idx, csr_index, chunk_start_idx)
    csr_index = torch.where(csr_index < chunk_end_idx, csr_index, chunk_end_idx)
    csr_index -= chunk_start_idx
    return csr_index


def chunk_csr_index(csr_index, chunk_count: int, chunk_rank: int):
    total_seq_len = csr_index[-1]
    split_seq_len = (total_seq_len + chunk_count - 1) // chunk_count
    rank_start_idx = chunk_rank * split_seq_len
    rank_end_idx = rank_start_idx + split_seq_len

    seq_len = csr_index[1:] - csr_index[:-1]
    reduce_tensor = torch.eye(len(seq_len), device=csr_index.device, dtype=torch.int32).repeat_interleave(
        seq_len, dim=1
    )
    reduce_tensor = reduce_tensor[:, rank_start_idx:rank_end_idx]
    chunk_seq_len = reduce_tensor.sum(dim=1)

    zeros_shape = list(chunk_seq_len.shape)
    zeros_shape[-1] = 1

    offsets = torch.cat((chunk_seq_len.new_zeros(zeros_shape), chunk_seq_len), -1).cumsum_(-1)
    return offsets.to(torch.int32)


def tensor_split_csr_index(csr_index, split_count: int, split_rank: int):
    total_seq_len = csr_index[-1]
    rank_start_idx, rank_end_idx = get_local_start_end_for_tensor_split(total_seq_len, split_rank, split_count)

    seq_len = csr_index[1:] - csr_index[:-1]
    reduce_tensor = torch.eye(len(seq_len), device=csr_index.device, dtype=torch.int32).repeat_interleave(
        seq_len, dim=1
    )
    reduce_tensor = reduce_tensor[:, rank_start_idx:rank_end_idx]
    chunk_seq_len = reduce_tensor.sum(dim=1)

    zeros_shape = list(chunk_seq_len.shape)
    zeros_shape[-1] = 1

    offsets = torch.cat((chunk_seq_len.new_zeros(zeros_shape), chunk_seq_len), -1).cumsum_(-1)
    return offsets.to(torch.int32)


def pad_seq(input_seq, pad_len):
    return torch.cat(
        [input_seq, torch.zeros((pad_len, *input_seq.shape[1:]), device=input_seq.device, dtype=input_seq.dtype)],
        dim=0,
    )
