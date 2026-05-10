# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for `kvbm.v2.vllm.dim_probe`.

These exercise the sentinel probe and `KvBlockLayout` derivation against
hand-rolled fake `AttentionBackend` classes — no GPU, no vLLM import, no
maturin build required beyond having `kvbm` importable.

The shape patterns mirror the canonical answers in vLLM's v1 attention
backends (verified against `vllm/v1/attention/backends/...`).
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

# Make the dim_probe module importable without going through kvbm's
# package __init__ (which requires the maturin-built _core extension).
HERE = os.path.dirname(os.path.abspath(__file__))
DIM_PROBE_PATH = os.path.normpath(
    os.path.join(HERE, "..", "python", "kvbm", "v2", "vllm", "dim_probe.py")
)
spec = importlib.util.spec_from_file_location(
    "_kvbm_dim_probe_under_test", DIM_PROBE_PATH
)
assert spec is not None and spec.loader is not None
dim_probe = importlib.util.module_from_spec(spec)
sys.modules["_kvbm_dim_probe_under_test"] = dim_probe
spec.loader.exec_module(dim_probe)

KvDim = dim_probe.KvDim
KvBlockLayout = dim_probe.KvBlockLayout
probe_kv_dim_layout = dim_probe.probe_kv_dim_layout
derive_block_layout = dim_probe.derive_block_layout
build_dim_layout_from_tensor = dim_probe.build_dim_layout_from_tensor
_FakeBackend = dim_probe._FakeBackend


# --- Shape & stride patterns from vLLM's v1 attention backends ----------------


def _flashattn_shape(num_blocks, block_size, num_kv_heads, head_size):
    # vllm/v1/attention/backends/flash_attn.py:143
    return (2, num_blocks, block_size, num_kv_heads, head_size)


def _flashattn_stride_nhd(include_num_layers_dimension):
    # vllm/v1/attention/backends/flash_attn.py:155-156
    if include_num_layers_dimension:
        return (2, 0, 1, 3, 4, 5)
    return (0, 1, 2, 3, 4)


def _flashattn_stride_hnd(include_num_layers_dimension):
    # vllm/v1/attention/backends/flash_attn.py:157-161
    if include_num_layers_dimension:
        return (2, 4, 0, 1, 3, 5)
    return (0, 1, 3, 2, 4)


def _flashinfer_shape(num_blocks, block_size, num_kv_heads, head_size):
    # vllm/v1/attention/backends/flashinfer.py:351-358
    return (num_blocks, 2, block_size, num_kv_heads, head_size)


def _flashinfer_stride_nhd(include_num_layers_dimension):
    if include_num_layers_dimension:
        return (1, 0, 2, 3, 4, 5)
    return (0, 1, 2, 3, 4)


def _flashinfer_stride_hnd(include_num_layers_dimension):
    if include_num_layers_dimension:
        return (1, 2, 4, 0, 3, 5)
    return (0, 1, 3, 2, 4)


def _mla_indexer_shape(num_blocks, block_size, num_kv_heads, head_size):
    # vllm/v1/attention/backends/mla/indexer.py:107 — assert num_kv_heads == 1.
    # Mirror the real assertion so a probe that forgets to substitute
    # num_kv_heads=1 under use_mla=True fails the same way the real backend
    # would.
    assert (
        num_kv_heads == 1
    ), f"MLA backends require num_kv_heads == 1, got {num_kv_heads}"
    return (num_blocks, block_size, head_size)


def _mla_indexer_stride(include_num_layers_dimension):
    # vllm/v1/attention/backends/mla/indexer.py:117-119 — identity
    return (0, 1, 2, 3) if include_num_layers_dimension else (0, 1, 2)


def _diffkv_shape(num_blocks, block_size, num_kv_heads, head_size):
    # vllm/v1/attention/backends/flash_attn_diffkv.py:48-62
    # Trailing axis is `head_size + head_size_v` — opaque to the prober.
    return (num_blocks, block_size, num_kv_heads, head_size + 64)


# --- Probe label tests --------------------------------------------------------


def test_flashattn_per_layer_labels():
    backend = _FakeBackend(_flashattn_shape, _flashattn_stride_nhd)
    dims = probe_kv_dim_layout(backend)
    assert dims == [
        KvDim.Outer,
        KvDim.Block,
        KvDim.Page,
        KvDim.HeadCount,
        KvDim.HeadSize,
    ]


def test_flashinfer_per_layer_labels():
    backend = _FakeBackend(_flashinfer_shape, _flashinfer_stride_nhd)
    dims = probe_kv_dim_layout(backend)
    assert dims == [
        KvDim.Block,
        KvDim.Outer,
        KvDim.Page,
        KvDim.HeadCount,
        KvDim.HeadSize,
    ]


def test_mla_labels_with_use_mla_hint():
    backend = _FakeBackend(_mla_indexer_shape, _mla_indexer_stride)
    dims = probe_kv_dim_layout(backend, use_mla=True)
    assert dims == [KvDim.Block, KvDim.Page, KvDim.HeadSize]


def test_diffkv_payload_label_for_trailing_axis():
    backend = _FakeBackend(_diffkv_shape, None)
    dims = probe_kv_dim_layout(backend)
    # Trailing axis is `1024 + 64 = 1088`, neither a sentinel nor 2 →
    # labelled `Payload` because it is in trailing position.
    assert dims == [
        KvDim.Block,
        KvDim.Page,
        KvDim.HeadCount,
        KvDim.Payload,
    ]


def test_unknown_non_trailing_axis_raises_with_clear_error():
    # Inject a backend whose shape has an unrecognised axis at position 1.
    def shape(num_blocks, block_size, num_kv_heads, head_size):
        return (num_blocks, 999, block_size, num_kv_heads, head_size)

    backend = _FakeBackend(shape, None)
    with pytest.raises(NotImplementedError) as exc_info:
        probe_kv_dim_layout(backend)
    msg = str(exc_info.value)
    assert "unrecognised non-trailing axis" in msg
    assert "999" in msg
    assert "position 1" in msg


# --- KvBlockLayout derivation tests -------------------------------------------


def test_flashattn_nhd_derives_operational_nhd():
    backend = _FakeBackend(_flashattn_shape, _flashattn_stride_nhd)
    dims = probe_kv_dim_layout(backend)
    assert derive_block_layout(backend, dims) == KvBlockLayout.OperationalNHD


def test_flashattn_hnd_derives_operational_hnd():
    backend = _FakeBackend(_flashattn_shape, _flashattn_stride_hnd)
    dims = probe_kv_dim_layout(backend)
    assert derive_block_layout(backend, dims) == KvBlockLayout.OperationalHND


def test_flashinfer_nhd_derives_operational_nhd():
    backend = _FakeBackend(_flashinfer_shape, _flashinfer_stride_nhd)
    dims = probe_kv_dim_layout(backend)
    assert derive_block_layout(backend, dims) == KvBlockLayout.OperationalNHD


def test_flashinfer_hnd_derives_operational_hnd():
    backend = _FakeBackend(_flashinfer_shape, _flashinfer_stride_hnd)
    dims = probe_kv_dim_layout(backend)
    assert derive_block_layout(backend, dims) == KvBlockLayout.OperationalHND


def test_mla_has_unknown_block_layout_no_head_axis():
    backend = _FakeBackend(_mla_indexer_shape, _mla_indexer_stride)
    dims = probe_kv_dim_layout(backend, use_mla=True)
    # No HeadCount axis — NHD/HND distinction is meaningless.
    assert derive_block_layout(backend, dims) == KvBlockLayout.Unknown


def test_backend_without_stride_order_yields_unknown():
    backend = _FakeBackend(_flashattn_shape, None)  # raises NotImplementedError
    dims = probe_kv_dim_layout(backend)
    assert derive_block_layout(backend, dims) == KvBlockLayout.Unknown


# --- build_dim_layout_from_tensor (the function the worker actually calls) ----


def test_build_dim_layout_uses_tensor_shape_for_sizes():
    backend = _FakeBackend(_flashattn_shape, _flashattn_stride_nhd)
    # Real tensor shape — kernel_block_size may differ from spec.block_size.
    real_shape = (2, 1024, 16, 8, 128)
    dims, sizes = build_dim_layout_from_tensor(backend, tensor_shape=real_shape)
    assert dims == [
        KvDim.Outer,
        KvDim.Block,
        KvDim.Page,
        KvDim.HeadCount,
        KvDim.HeadSize,
    ]
    assert sizes == list(real_shape)


def test_build_dim_layout_rejects_rank_mismatch():
    backend = _FakeBackend(_flashattn_shape, _flashattn_stride_nhd)
    with pytest.raises(ValueError, match="rank"):
        build_dim_layout_from_tensor(backend, tensor_shape=(1, 2, 3))


# --- Sentinels are unique and don't collide with K/V outer (`2`) --------------


def test_sentinels_pairwise_distinct_and_not_two():
    # If this assertion in the module fired, the import would fail —
    # this is a defence-in-depth duplicate.
    assert (
        len(
            {
                dim_probe._S_BLOCKS,
                dim_probe._S_PAGE,
                dim_probe._S_HEAD,
                dim_probe._S_HSZ,
                2,
            }
        )
        == 5
    )


def test_sentinels_satisfy_backend_validation_constraints():
    # FlashAttn/Triton block_size validation:
    assert dim_probe._S_PAGE % 16 == 0
    assert dim_probe._S_BLOCKS % 16 == 0  # passes the same check
    # FlashAttn head_size validation:
    assert dim_probe._S_HSZ % 8 == 0
