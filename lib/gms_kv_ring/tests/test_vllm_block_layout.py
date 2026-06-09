# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E402
"""Unit tests for GMSKVCacheConnectorV1 layer-registration / block-layout
inference. Covers non-MLA (blocks-first + KV-first) and MLA (non-split +
split-block) layouts. Modeled on Pegaflow's `_infer_kv_cache_registration`
test surface; we ported the same shape coverage."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from gpu_memory_service.integrations.vllm.gds_connector_v1 import (
    _build_layers_and_layout,
    _infer_layer_registration,
    _LayerRegistration,
)


def _make_tensor(shape, dtype=torch.float16):
    """Allocate on CPU; we only read shape/stride/element_size."""
    return torch.empty(shape, dtype=dtype, device="cpu")


# ---------- _infer_layer_registration ----------------------------------------


def test_infer_blocks_first_non_mla():
    # vLLM common shape: (num_blocks, block_size, num_heads, head_dim).
    t = _make_tensor((8, 16, 4, 8))
    reg = _infer_layer_registration(t, logical_block_size=16, is_mla=False)
    assert reg == _LayerRegistration(
        num_blocks=8,
        bytes_per_block=16 * 4 * 8 * 2,  # 16 tokens × 4 heads × 8 dim × fp16
        kv_stride_bytes=0,
        physical_blocks_per_logical_block=1,
    )


def test_infer_kv_first_non_mla():
    # vLLM KV-first: (2, num_blocks, block_size, num_heads, head_dim).
    t = _make_tensor((2, 8, 16, 4, 8))
    reg = _infer_layer_registration(t, logical_block_size=16, is_mla=False)
    expected_block_bytes = 16 * 4 * 8 * 2
    expected_kv_stride = 8 * 16 * 4 * 8 * 2  # one full KV-segment in bytes
    assert reg == _LayerRegistration(
        num_blocks=8,
        bytes_per_block=expected_block_bytes,
        kv_stride_bytes=expected_kv_stride,
        physical_blocks_per_logical_block=1,
    )


def test_infer_mla_non_split():
    # MLA, logical == physical: shape (num_blocks, block_size, kv_lora_rank).
    t = _make_tensor((8, 64, 512))
    reg = _infer_layer_registration(t, logical_block_size=64, is_mla=True)
    assert reg.num_blocks == 8
    assert reg.bytes_per_block == 64 * 512 * 2
    assert reg.kv_stride_bytes == 0
    assert reg.physical_blocks_per_logical_block == 1


def test_infer_mla_split_block_fashmla_style():
    # DeepSeek-V3 / FlashMLA: scheduler block 128 tokens, kernel block
    # 64 tokens → 2 physical rows per logical block. Physical tensor
    # has 2× num_blocks rows.
    t = _make_tensor((16, 64, 512))
    reg = _infer_layer_registration(t, logical_block_size=128, is_mla=True)
    assert reg.num_blocks == 8  # 16 / 2
    assert reg.bytes_per_block == 2 * 64 * 512 * 2  # 2 rows coalesced
    assert reg.kv_stride_bytes == 0
    assert reg.physical_blocks_per_logical_block == 2


def test_infer_rejects_indivisible_split():
    t = _make_tensor((16, 70, 512))  # logical 128 % 70 != 0
    with pytest.raises(ValueError, match="multiple of physical block size"):
        _infer_layer_registration(t, logical_block_size=128, is_mla=True)


def test_infer_rejects_misaligned_physical_count():
    # 17 physical rows, ratio 2 → 17 % 2 != 0
    t = _make_tensor((17, 64, 512))
    with pytest.raises(ValueError, match="divisible by physical/logical"):
        _infer_layer_registration(t, logical_block_size=128, is_mla=True)


def test_infer_rejects_zero_logical_block_size():
    t = _make_tensor((8, 16))
    with pytest.raises(ValueError, match="logical block size must be > 0"):
        _infer_layer_registration(t, logical_block_size=0, is_mla=False)


# ---------- _build_layers_and_layout: end-to-end ----------------------------


def _kv_caches(shape, n_layers=3):
    return {f"model.layers.{i}.self_attn": _make_tensor(shape) for i in range(n_layers)}


def test_build_blocks_first_layout_emits_one_entry_per_layer():
    caches = _kv_caches((8, 16, 4, 8))
    layers, layout, n_layers, block_bytes = _build_layers_and_layout(
        caches,
        logical_block_size=16,
        is_mla=False,
    )
    assert n_layers == 3
    assert block_bytes == 16 * 4 * 8 * 2
    entries = layout(3)
    assert len(entries) == 3
    for L, (layer_idx, offset, size) in enumerate(entries):
        assert layer_idx == L
        assert offset == 3 * block_bytes
        assert size == block_bytes


def test_build_kv_first_layout_emits_two_entries_per_layer():
    caches = _kv_caches((2, 8, 16, 4, 8))
    layers, layout, n_layers, block_bytes = _build_layers_and_layout(
        caches,
        logical_block_size=16,
        is_mla=False,
    )
    assert n_layers == 3
    expected_block_bytes = 16 * 4 * 8 * 2
    expected_kv_stride = 8 * 16 * 4 * 8 * 2
    assert block_bytes == expected_block_bytes

    entries = layout(3)
    # 3 layers × 2 KV segments = 6 entries
    assert len(entries) == 6
    for L in range(n_layers):
        k_entry = entries[2 * L]
        v_entry = entries[2 * L + 1]
        assert k_entry == (L, 3 * expected_block_bytes, expected_block_bytes)
        assert v_entry == (
            L,
            expected_kv_stride + 3 * expected_block_bytes,
            expected_block_bytes,
        )


def test_build_mla_split_block_layout_coalesces_physical_rows():
    # DeepSeek-V3 / FlashMLA: 8 logical blocks, ratio 2, so 16 physical rows.
    # Each layout entry for one logical block covers the 2 physical rows.
    caches = _kv_caches((16, 64, 512))
    layers, layout, n_layers, block_bytes = _build_layers_and_layout(
        caches,
        logical_block_size=128,
        is_mla=True,
    )
    assert n_layers == 3
    assert block_bytes == 2 * 64 * 512 * 2

    entries = layout(2)
    assert len(entries) == 3
    for L in range(n_layers):
        layer_idx, offset, size = entries[L]
        assert layer_idx == L
        assert offset == 2 * block_bytes
        # One contiguous coalesced range; not two separate physical entries.
        assert size == block_bytes


def test_build_legacy_path_when_block_size_unknown():
    # Legacy callers (no logical_block_size) keep the axis-max
    # heuristic; behavior must be identical to pre-MLA shipped code.
    # The heuristic picks `max(shape[0], shape[1])` as num_blocks.
    # Use realistic production-shaped tensor where num_blocks > block_size.
    caches = _kv_caches((512, 16, 4, 8))
    layers, layout, n_layers, block_bytes = _build_layers_and_layout(caches)
    assert n_layers == 3
    layer_bytes = 512 * 16 * 4 * 8 * 2
    assert block_bytes == layer_bytes // 512
    entries = layout(0)
    assert len(entries) == 3


def test_build_rejects_empty_kv_caches():
    with pytest.raises(ValueError, match="empty kv_caches dict"):
        _build_layers_and_layout({})


def test_build_rejects_hybrid_layouts():
    caches = {
        "model.layers.0": _make_tensor((8, 16, 4, 8)),
        "model.layers.1": _make_tensor((8, 16, 4, 16)),  # different
    }
    with pytest.raises(NotImplementedError, match="hybrid"):
        _build_layers_and_layout(caches, logical_block_size=16, is_mla=False)
