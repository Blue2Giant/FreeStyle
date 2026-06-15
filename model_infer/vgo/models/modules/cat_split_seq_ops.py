"""
Optimized cat_seq, split_seq, and split_seq_by_len_list using torch.autograd.Function

Key optimizations:
1. Custom backward to fuse split/concat operations
2. Reduce number of CUDA kernels in backward pass
3. Preserve exact gradients while improving efficiency
"""

from itertools import zip_longest

import torch


class CatSeqFunction(torch.autograd.Function):
    """
    Optimized cat_seq with fused backward pass

    The original implementation creates multiple intermediate tensors during forward,
    which leads to multiple split kernels during backward. This custom autograd function
    provides a fused backward that directly computes gradients without tracking intermediates.
    """

    @staticmethod
    def forward(ctx, lens_all_cpu, *seqs_all):
        """
        Args:
            lens_all_cpu: List of CPU tensors with split indices
            *seqs_all: Variable number of input tensors

        Returns:
            Concatenated tensor with interleaved chunks
        """
        # Perform the original forward operation
        split_all = [seqs.tensor_split(lens, dim=0) for seqs, lens in zip(seqs_all, lens_all_cpu)]
        split_rearange_all = []
        for split_each in zip(*split_all):
            split_rearange_all += split_each
        seqs_cat = torch.cat(split_rearange_all, dim=0)

        # Save information for backward
        ctx.num_seqs = len(seqs_all)
        ctx.num_chunks = len(lens_all_cpu[0]) + 1 if len(lens_all_cpu) > 0 else 1
        ctx.split_sizes = [t.shape[0] for t in split_rearange_all]

        return seqs_cat

    @staticmethod
    def backward(ctx, grad_output):
        """
        Fused backward pass that directly computes gradients for each input sequence
        """
        num_seqs = ctx.num_seqs
        num_chunks = ctx.num_chunks
        split_sizes = ctx.split_sizes

        # Split grad_output according to forward concatenation
        grad_splits = torch.split(grad_output, split_sizes, dim=0)

        # Rearrange gradients: [seq0_chunk0, seq1_chunk0, ..., seq0_chunk1, seq1_chunk1, ...]
        # Back to: [[seq0_chunk0, seq0_chunk1, ...], [seq1_chunk0, seq1_chunk1, ...], ...]
        grad_seqs_all = []
        for seq_idx in range(num_seqs):
            seq_chunks = [grad_splits[chunk_idx * num_seqs + seq_idx] for chunk_idx in range(num_chunks)]
            grad_seq = torch.cat(seq_chunks, dim=0)
            grad_seqs_all.append(grad_seq)

        # Return None for lens_all_cpu, then gradients for each sequence
        return (None, *tuple(grad_seqs_all))


class SplitSeqFunction(torch.autograd.Function):
    """
    Optimized split_seq with fused backward pass

    The original implementation uses tensor_split + multiple cat operations,
    leading to multiple concat kernels in backward. This provides a fused backward.
    """

    @staticmethod
    def forward(ctx, seqs, lens_all_cpu):
        """
        Args:
            seqs: Input tensor to split
            lens_all_cpu: List of two CPU tensors with split lengths

        Returns:
            Tuple of two tensors (even and odd sequences concatenated)
        """
        # Compute split indices
        split_indices = torch.stack(lens_all_cpu).T.flatten().cumsum(dim=0)[:-1]

        # Perform split
        seqs_split = torch.tensor_split(seqs, split_indices.cpu())

        # Concatenate even and odd sequences
        seqs_even = torch.cat(seqs_split[0::2], dim=0)
        seqs_odd = torch.cat(seqs_split[1::2], dim=0)

        # Save split sizes for backward
        ctx.split_sizes_even = [s.shape[0] for s in seqs_split[0::2]]
        ctx.split_sizes_odd = [s.shape[0] for s in seqs_split[1::2]]

        return seqs_even, seqs_odd

    @staticmethod
    def backward(ctx, grad_even, grad_odd):
        """
        Fused backward pass that directly reconstructs gradient for input
        """
        split_sizes_even = ctx.split_sizes_even
        split_sizes_odd = ctx.split_sizes_odd

        # Split gradients back to chunks
        grad_even_splits = torch.split(grad_even, split_sizes_even, dim=0)
        grad_odd_splits = torch.split(grad_odd, split_sizes_odd, dim=0)

        # Interleave even and odd gradients
        grad_seqs_list = []
        for even_chunk, odd_chunk in zip_longest(grad_even_splits, grad_odd_splits):
            if even_chunk is not None:
                grad_seqs_list.append(even_chunk)
            if odd_chunk is not None:
                grad_seqs_list.append(odd_chunk)

        # Concatenate to form gradient for input
        grad_seqs = torch.cat(grad_seqs_list, dim=0)

        # Return gradient for seqs, None for lens_all_cpu
        return grad_seqs, None


class SplitSeqByLenListFunction(torch.autograd.Function):
    """
    Optimized split_seq_by_len_list with fused backward pass

    Similar to split_seq but takes lists of lengths instead of split indices.
    The original uses torch.split + multiple cat, causing multiple concat kernels in backward.
    """

    @staticmethod
    def forward(ctx, seqs, lens_all):
        """
        Args:
            seqs: Input tensor to split
            lens_all: Tuple of two lists containing chunk sizes

        Returns:
            Tuple of two tensors (even and odd sequences concatenated)
        """

        # Interleave the two length lists
        def interleave_lists(list1, list2):
            return [elem for pair in zip_longest(list1, list2) for elem in pair if elem is not None]

        interleaved_lens = interleave_lists(lens_all[0], lens_all[1])

        # Split according to interleaved lengths
        seqs_split = torch.split(seqs, interleaved_lens, dim=0)

        # Concatenate even and odd sequences
        seqs_even = torch.cat(seqs_split[0::2], dim=0)
        seqs_odd = torch.cat(seqs_split[1::2], dim=0)

        # Save split sizes for backward
        ctx.split_sizes_even = [s.shape[0] for s in seqs_split[0::2]]
        ctx.split_sizes_odd = [s.shape[0] for s in seqs_split[1::2]]

        return seqs_even, seqs_odd

    @staticmethod
    def backward(ctx, grad_even, grad_odd):
        """
        Fused backward pass that directly reconstructs gradient for input
        """
        split_sizes_even = ctx.split_sizes_even
        split_sizes_odd = ctx.split_sizes_odd

        # Split gradients back to chunks
        grad_even_splits = torch.split(grad_even, split_sizes_even, dim=0)
        grad_odd_splits = torch.split(grad_odd, split_sizes_odd, dim=0)

        # Interleave even and odd gradients
        grad_seqs_list = []
        for even_chunk, odd_chunk in zip_longest(grad_even_splits, grad_odd_splits):
            if even_chunk is not None:
                grad_seqs_list.append(even_chunk)
            if odd_chunk is not None:
                grad_seqs_list.append(odd_chunk)

        # Concatenate to form gradient for input
        grad_seqs = torch.cat(grad_seqs_list, dim=0)

        # Return gradient for seqs, None for lens_all
        return grad_seqs, None


# ============================================================================
# Public API functions
# ============================================================================


def cat_seq_optimized(seqs_all: list[torch.Tensor], lens_all: list[torch.Tensor]) -> torch.Tensor:
    """
    Optimized cat_seq with custom backward pass

    Args:
        seqs_all: List of tensors to concatenate
        lens_all: List of split indices (as tensors) for each sequence

    Returns:
        Concatenated tensor with interleaved chunks

    Example:
        >>> seq1 = torch.randn(250, 64, requires_grad=True)
        >>> seq2 = torch.randn(200, 64, requires_grad=True)
        >>> lens1 = torch.tensor([100])  # splits seq1 at [100, 250]
        >>> lens2 = torch.tensor([80])   # splits seq2 at [80, 200]
        >>> result = cat_seq_optimized([seq1, seq2], [lens1, lens2])
        >>> # result: [seq1[:100], seq2[:80], seq1[100:], seq2[80:]]
    """
    # Convert lens to CPU if needed (tensor_split requires CPU indices)
    lens_all_cpu = [lens.cpu() if isinstance(lens, torch.Tensor) else torch.tensor(lens) for lens in lens_all]

    # Apply custom function
    return CatSeqFunction.apply(lens_all_cpu, *seqs_all)


def split_seq_optimized(seqs: torch.Tensor, lens_all: list[torch.Tensor]) -> list[torch.Tensor]:
    """
    Optimized split_seq with custom backward pass

    Args:
        seqs: Input tensor to split
        lens_all: List of two tensors with split lengths

    Returns:
        List of two tensors (even and odd sequences concatenated)

    Example:
        >>> seqs = torch.randn(450, 64, requires_grad=True)
        >>> lens1 = torch.tensor([100])  # even sequence has chunks [100, 150]
        >>> lens2 = torch.tensor([80])   # odd sequence has chunks [80, 120]
        >>> result = split_seq_optimized(seqs, [lens1, lens2])
        >>> # result[0]: concatenation of chunks at indices 0, 2, 4, ...
        >>> # result[1]: concatenation of chunks at indices 1, 3, 5, ...
    """
    # Convert lens to CPU
    lens_all_cpu = [lens.cpu() if isinstance(lens, torch.Tensor) else torch.tensor(lens) for lens in lens_all]

    # Apply custom function
    seqs_even, seqs_odd = SplitSeqFunction.apply(seqs, lens_all_cpu)
    return [seqs_even, seqs_odd]


def split_seq_by_len_list_optimized(seqs: torch.Tensor, lens_all: list) -> list[torch.Tensor]:
    """
    Optimized split_seq_by_len_list with custom backward pass

    Args:
        seqs: Input tensor to split
        lens_all: List of two lists containing chunk sizes

    Returns:
        List of two tensors (even and odd sequences concatenated)

    Example:
        >>> seqs = torch.randn(450, 64, requires_grad=True)
        >>> lens_all = [[100, 150], [80, 120]]  # Lists of chunk sizes
        >>> result = split_seq_by_len_list_optimized(seqs, lens_all)
        >>> # Splits: [100, 80, 150, 120] (interleaved)
        >>> # result[0]: cat([seqs[:100], seqs[180:330]])
        >>> # result[1]: cat([seqs[100:180], seqs[330:]])
    """
    # Convert to tuple for autograd.Function
    # lens_all_tuple = tuple(lens_all)
    lens_all_tuple = [lens.cpu() if isinstance(lens, torch.Tensor) else torch.tensor(lens) for lens in lens_all]

    # Apply custom function
    # seqs_even, seqs_odd = SplitSeqByLenListFunction.apply(seqs, lens_all_tuple)
    seqs_even, seqs_odd = SplitSeqFunction.apply(seqs, lens_all_tuple)
    return [seqs_even, seqs_odd]


# ============================================================================
# Backward compatibility exports
# ============================================================================

cat_seq = cat_seq_optimized
split_seq = split_seq_optimized
split_seq_by_len_list = split_seq_by_len_list_optimized
