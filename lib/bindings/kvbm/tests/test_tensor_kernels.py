# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PyTorch-based regression tests for the CUDA tensor packing kernels.

The goal is to mirror how an ML engineer would use the library, so the tests
act as both verification and documentation.
"""

from typing import List

import pytest
import torch
from kvbm import kernels as ctk


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    """
    Relax tolerances for low-precision dtypes.

    fp16/bf16 round differently from fp32/fp64. Using dtype-aware tolerances
    avoids spurious failures while still guarding against layout mistakes.
    """
    if dtype in (torch.float16, torch.bfloat16):
        return 1e-2, 1e-2
    return 1e-5, 1e-5


def _make_blocks(universal: torch.Tensor, layout: str) -> List[torch.Tensor]:
    """
    Reference implementation for turning a universal tensor into its block stack.

    `layout` controls the per-chunk permutation:
    - "NHD": expect `[nh, nl, no, nt, hd] -> [nt, nh, hd]`
    - "HND": expect `[nh, nl, no, nt, hd] -> [nh, nt, hd]`
    """
    nh, nl, no, nt, hd = universal.shape
    blocks = []
    for layer in range(nl):
        for outer in range(no):
            slice_ = universal[:, layer, outer, :, :].contiguous()
            if layout.upper() == "NHD":
                block = slice_.permute(1, 0, 2).contiguous()
            elif layout.upper() == "HND":
                block = slice_.contiguous()
            else:
                raise ValueError(f"Unsupported layout {layout}")
            blocks.append(block.clone())
    return blocks


def _call_with_backend(func, backend: str, *args):
    """
    Helper to invoke a binding with a backend override, translating
    unsupported backends into pytest skips instead of hard failures.
    """
    try:
        if backend is None:
            func(*args)
        else:
            func(*args, backend=backend)
    except RuntimeError as err:
        if "cudaErrorNotSupported" in str(err):
            pytest.skip(f"{backend} backend not supported on this runtime")
        raise


@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32, torch.float64],
)
def test_block_universal_roundtrip(layout: str, dtype: torch.dtype) -> None:
    """
    Launch `nb` block stacks through block⇄universal kernels and compare
    against pure-PyTorch permutations.

    Shapes:
    - universals: `[nb][nh, nl, no, nt, hd]`
    - blocks:     `[nb][nl * no][nt, nh, hd]` (or `[nh, nt, hd]` for HND)
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for these tests")

    device = torch.device("cuda:0")
    torch.manual_seed(0)

    nh, nl, no, nt, hd = 3, 2, 2, 4, 5
    nb = 3
    universals = [
        torch.randn(nh, nl, no, nt, hd, device=device, dtype=dtype) for _ in range(nb)
    ]

    # Prepare block stacks by permuting each universal tensor with PyTorch ops.
    blocks = [_make_blocks(t, layout) for t in universals]
    outputs = [torch.empty_like(t) for t in universals]

    # Convert block stacks -> universal using the CUDA kernels.
    ctk.block_to_universal(blocks, outputs, layout)
    torch.cuda.synchronize()

    atol, rtol = _tolerances(dtype)
    for produced, expected in zip(outputs, universals):
        assert torch.allclose(produced, expected, atol=atol, rtol=rtol)

    # Zero the inputs and run the reverse direction.
    for block_set in blocks:
        for block in block_set:
            block.zero_()

    ctk.universal_to_block(universals, blocks, layout)
    torch.cuda.synchronize()

    expected_blocks = [_make_blocks(t, layout) for t in universals]
    for produced_set, expected_set in zip(blocks, expected_blocks):
        for produced, expected in zip(produced_set, expected_set):
            assert torch.allclose(produced, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32, torch.float64],
)
def test_operational_roundtrip(dtype: torch.dtype) -> None:
    """
    Validate the block⇄operational fusion path.

    Operational layout flattens `[nt, nh, hd]` into a single `inner` dimension.
    This is useful when `nh` does not need to vary between participants.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for these tests")

    device = torch.device("cuda:0")
    torch.manual_seed(1)

    nh, nl, no, nt, hd = 2, 3, 2, 4, 3
    nb = 2
    universals = [
        torch.randn(nh, nl, no, nt, hd, device=device, dtype=dtype) for _ in range(nb)
    ]

    reference_blocks = [_make_blocks(t, "NHD") for t in universals]
    blocks = [[b.clone() for b in block_set] for block_set in reference_blocks]

    inner = nt * nh * hd
    operationals = [
        torch.empty(nl, no, inner, device=device, dtype=dtype) for _ in range(nb)
    ]

    # Pack block stacks -> operational.
    ctk.block_to_operational(blocks, operationals)
    torch.cuda.synchronize()

    atol, rtol = _tolerances(dtype)
    for operational, ref_blocks in zip(operationals, reference_blocks):
        expected_operational = torch.stack(
            [b.reshape(-1) for b in ref_blocks], dim=0
        ).view(nl, no, inner)
        assert torch.allclose(operational, expected_operational, atol=atol, rtol=rtol)

    # Zero and unpack back into block stacks.
    for block_set in blocks:
        for block in block_set:
            block.zero_()

    ctk.operational_to_block(operationals, blocks)
    torch.cuda.synchronize()

    for produced_set, expected_set in zip(blocks, reference_blocks):
        for produced, expected in zip(produced_set, expected_set):
            assert torch.allclose(produced, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("backend", [None, "auto", "kernel", "async", "batch"])
def test_operational_backends(backend):
    """
    Exercise every backend override. When a backend is unavailable (e.g. batch
    on older runtimes) we skip instead of failing.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for these tests")

    device = torch.device("cuda:0")
    nh, nl, no, nt, hd = 2, 1, 2, 3, 4
    nb = 1
    dtype = torch.float32

    universals = [
        torch.randn(nh, nl, no, nt, hd, device=device, dtype=dtype) for _ in range(nb)
    ]
    blocks = [_make_blocks(t, "NHD") for t in universals]
    operationals = [
        torch.empty(nl, no, nt * nh * hd, device=device, dtype=dtype) for _ in range(nb)
    ]

    _call_with_backend(ctk.block_to_operational, backend, blocks, operationals)
    torch.cuda.synchronize()

    for block in blocks[0]:
        block.zero_()

    _call_with_backend(ctk.operational_to_block, backend, operationals, blocks)
    torch.cuda.synchronize()

    reference = _make_blocks(universals[0], "NHD")
    assert torch.allclose(blocks[0][0], reference[0], atol=1e-5, rtol=1e-5)


def test_universal_shape_mismatch():
    """
    Blocks with the wrong inner shape should trigger a ValueError.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for these tests")

    device = torch.device("cuda:0")
    dtype = torch.float32

    universal = torch.randn(2, 2, 1, 2, 4, device=device, dtype=dtype)
    bad_block = torch.randn(2, 3, 4, device=device, dtype=dtype)  # wrong nt

    with pytest.raises(ValueError):
        ctk.block_to_universal([[bad_block]], [torch.empty_like(universal)], "NHD")


def test_dtype_mismatch_error():
    """
    Mixed dtypes in a batch should raise rather than silently convert.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for these tests")

    device = torch.device("cuda:0")
    universal_f16 = torch.randn(1, 1, 1, 2, 4, device=device, dtype=torch.float16)
    universal_f32 = torch.randn(1, 1, 1, 2, 4, device=device, dtype=torch.float32)

    blocks = [_make_blocks(universal_f16, "NHD"), _make_blocks(universal_f32, "NHD")]

    with pytest.raises(TypeError):
        ctk.block_to_universal(blocks, [universal_f16, universal_f32], "NHD")


def test_non_cuda_tensor_error():
    """
    CPU tensors should be rejected up-front with a helpful message.
    """
    device = torch.device("cpu")
    universal = torch.randn(1, 1, 1, 2, 4, device=device)
    blocks = _make_blocks(universal.cuda(), "NHD")

    with pytest.raises(ValueError):
        ctk.block_to_universal([blocks], [universal], "NHD")


def test_empty_batch_noop():
    """
    An empty batch should succeed without touching CUDA.
    """
    assert ctk.block_to_universal([], [], "NHD") is None
    assert ctk.universal_to_block([], [], "NHD") is None
    assert ctk.block_to_operational([], [], None) is None
    assert ctk.operational_to_block([], [], None) is None
