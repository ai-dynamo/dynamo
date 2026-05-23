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
select_fc_variant = dim_probe.select_fc_variant
select_fc_for_model = dim_probe.select_fc_for_model
FC_INELIGIBLE_NO_BACKENDS = dim_probe.FC_INELIGIBLE_NO_BACKENDS
FC_INELIGIBLE_HYBRID_BACKENDS = dim_probe.FC_INELIGIBLE_HYBRID_BACKENDS
FC_INELIGIBLE_BACKEND_NO_MATCH = dim_probe.FC_INELIGIBLE_BACKEND_NO_MATCH
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


# --- Cross-layer (fully-contiguous, include_num_layers=True) ------------------
#
# vLLM hands `register_cross_layers_kv_cache` the underlying *physical
# contiguous allocation* — `tensor.shape` is in physical byte order, not
# the post-permute attention view. probe_kv_dim_layout(..., include_num_layers=True)
# must therefore return labels in physical order so they pair correctly
# with `tensor.shape`.
#
# Both NHD backends produce the *same* FC physical layout:
#   [num_blocks, num_layers, K/V, page_size, num_kv_heads, head_size]
# — that's the whole point of FullyContiguousLayout (the byte layout is
# stable across backends even though their per-layer logical orderings
# differ).


def test_flashattn_cross_layer_nhd_physical_order():
    backend = _FakeBackend(_flashattn_shape, _flashattn_stride_nhd)
    dims = probe_kv_dim_layout(backend, include_num_layers=True)
    assert dims == [
        KvDim.Block,
        KvDim.Layer,
        KvDim.Outer,
        KvDim.Page,
        KvDim.HeadCount,
        KvDim.HeadSize,
    ]


def test_flashinfer_cross_layer_nhd_physical_order():
    backend = _FakeBackend(_flashinfer_shape, _flashinfer_stride_nhd)
    dims = probe_kv_dim_layout(backend, include_num_layers=True)
    # FlashInfer NHD cross-layer produces the SAME physical FC layout as
    # FA NHD — proving the FC contract is backend-stable.
    assert dims == [
        KvDim.Block,
        KvDim.Layer,
        KvDim.Outer,
        KvDim.Page,
        KvDim.HeadCount,
        KvDim.HeadSize,
    ]


def test_flashattn_cross_layer_hnd_physical_order():
    backend = _FakeBackend(_flashattn_shape, _flashattn_stride_hnd)
    dims = probe_kv_dim_layout(backend, include_num_layers=True)
    # FA HND stride_order = (2, 4, 0, 1, 3, 5) — HeadCount lands at
    # physical position 1, which is incompatible with FullyContiguousLayout.
    # The FC byte-order assertion in worker.py rejects this.
    assert dims == [
        KvDim.Block,
        KvDim.HeadCount,
        KvDim.Layer,
        KvDim.Outer,
        KvDim.Page,
        KvDim.HeadSize,
    ]


def test_flashinfer_cross_layer_hnd_physical_order():
    backend = _FakeBackend(_flashinfer_shape, _flashinfer_stride_hnd)
    dims = probe_kv_dim_layout(backend, include_num_layers=True)
    # FlashInfer HND stride_order = (1, 2, 4, 0, 3, 5).
    assert dims == [
        KvDim.Block,
        KvDim.Outer,
        KvDim.HeadCount,
        KvDim.Layer,
        KvDim.Page,
        KvDim.HeadSize,
    ]


def test_flashattn_cross_layer_nhd_derives_operational_nhd():
    backend = _FakeBackend(_flashattn_shape, _flashattn_stride_nhd)
    dims = probe_kv_dim_layout(backend, include_num_layers=True)
    assert (
        derive_block_layout(backend, dims, include_num_layers=True)
        == KvBlockLayout.OperationalNHD
    )


def test_flashattn_cross_layer_hnd_derives_operational_hnd():
    backend = _FakeBackend(_flashattn_shape, _flashattn_stride_hnd)
    dims = probe_kv_dim_layout(backend, include_num_layers=True)
    assert (
        derive_block_layout(backend, dims, include_num_layers=True)
        == KvBlockLayout.OperationalHND
    )


def test_flashinfer_cross_layer_nhd_derives_operational_nhd():
    backend = _FakeBackend(_flashinfer_shape, _flashinfer_stride_nhd)
    dims = probe_kv_dim_layout(backend, include_num_layers=True)
    assert (
        derive_block_layout(backend, dims, include_num_layers=True)
        == KvBlockLayout.OperationalNHD
    )


def test_flashinfer_cross_layer_hnd_derives_operational_hnd():
    backend = _FakeBackend(_flashinfer_shape, _flashinfer_stride_hnd)
    dims = probe_kv_dim_layout(backend, include_num_layers=True)
    assert (
        derive_block_layout(backend, dims, include_num_layers=True)
        == KvBlockLayout.OperationalHND
    )


def test_build_dim_layout_cross_layer_nhd_pairs_physical_sizes():
    backend = _FakeBackend(_flashattn_shape, _flashattn_stride_nhd)
    # Cross-layer FC physical shape: [num_blocks, num_layers, K/V, page, h, d].
    real_shape = (1024, 32, 2, 16, 8, 128)
    dims, sizes = build_dim_layout_from_tensor(
        backend, tensor_shape=real_shape, include_num_layers=True
    )
    # Labels are physical-order, paired index-for-index with the
    # contiguous tensor's shape.
    assert dims == [
        KvDim.Block,
        KvDim.Layer,
        KvDim.Outer,
        KvDim.Page,
        KvDim.HeadCount,
        KvDim.HeadSize,
    ]
    assert sizes == list(real_shape)
    # Concretely: Block axis carries num_blocks, Layer axis carries num_layers.
    assert sizes[dims.index(KvDim.Block)] == 1024
    assert sizes[dims.index(KvDim.Layer)] == 32


def test_probe_cross_layer_rejects_backend_without_cross_layer_stride_order():
    # _FakeBackend with stride_order=None raises NotImplementedError when
    # `get_kv_cache_stride_order` is called — simulates a backend that
    # supports per-layer but not cross-layer.
    backend = _FakeBackend(_flashattn_shape, None)
    with pytest.raises(RuntimeError, match="include_num_layers=True requires"):
        probe_kv_dim_layout(backend, include_num_layers=True)


# --- select_fc_variant: which backends map to which FC byte ordering ----------


def test_select_fc_variant_fa_nhd_is_operational_nhd():
    backend = _FakeBackend(_flashattn_shape, _flashattn_stride_nhd)
    assert select_fc_variant(backend) == KvBlockLayout.OperationalNHD


def test_select_fc_variant_flashinfer_nhd_is_operational_nhd():
    backend = _FakeBackend(_flashinfer_shape, _flashinfer_stride_nhd)
    # FlashInfer NHD produces the same cross-layer physical layout as FA
    # NHD — that's the whole point of FullyContiguousLayout.
    assert select_fc_variant(backend) == KvBlockLayout.OperationalNHD


def test_select_fc_variant_fa_hnd_maps_to_universal():
    # FA HND's natural cross-layer physical layout
    # `[Block, HeadCount, Layer, Outer, Page, HeadSize]` happens to match
    # FullyContiguousLayout's `Universal` variant exactly. No new variant
    # or kernel work needed — FA HND can register as FC Universal directly.
    backend = _FakeBackend(_flashattn_shape, _flashattn_stride_hnd)
    assert select_fc_variant(backend) == KvBlockLayout.Universal


def test_select_fc_variant_flashinfer_hnd_is_none():
    # FlashInfer HND produces
    # `[Block, Outer, HeadCount, Layer, Page, HeadSize]` — Layer at byte
    # position 3 doesn't match any of the 3 supported FC orderings. Falls
    # back to per-layer (LW).
    backend = _FakeBackend(_flashinfer_shape, _flashinfer_stride_hnd)
    assert select_fc_variant(backend) is None


def test_select_fc_variant_mla_is_none():
    # MLA's 3-dim per-layer shape lacks Outer/HeadCount — FC variants all
    # require both.
    backend = _FakeBackend(_mla_indexer_shape, _mla_indexer_stride)
    assert select_fc_variant(backend) is None


def test_select_fc_variant_backend_without_cross_layer_stride_returns_none():
    # A backend that raises on `get_kv_cache_stride_order(include_num_layers=True)`
    # is non-FC-eligible — auto-fallback to LW. No exception leaks.
    backend = _FakeBackend(_flashattn_shape, None)
    assert select_fc_variant(backend) is None


def test_select_fc_variant_unknown_axis_returns_none():
    # A backend whose per-layer shape has an axis the prober can't
    # identify (NotImplementedError from probe_kv_dim_layout) — treat
    # as non-FC-eligible rather than crashing.
    def shape(num_blocks, block_size, num_kv_heads, head_size):
        return (num_blocks, 999, block_size, num_kv_heads, head_size)

    backend = _FakeBackend(shape, _flashattn_stride_nhd)
    assert select_fc_variant(backend) is None


# --- select_fc_for_model: whole-model contract (hybrid handling) --------------
#
# The whole-model wrapper has to distinguish three "no FC" reasons so the
# connector can log them differently. These tests pin the contract; the
# connector-level integration is exercised by disagg-smoke.


def test_select_fc_for_model_single_fa_nhd_picks_operational_nhd():
    backends = [_FakeBackend(_flashattn_shape, _flashattn_stride_nhd)]
    variant, reason = select_fc_for_model(backends)
    assert variant == KvBlockLayout.OperationalNHD
    assert reason is None


def test_select_fc_for_model_single_fa_hnd_picks_universal():
    backends = [_FakeBackend(_flashattn_shape, _flashattn_stride_hnd)]
    variant, reason = select_fc_for_model(backends)
    assert variant == KvBlockLayout.Universal
    assert reason is None


def test_select_fc_for_model_empty_backends_returns_no_backends_reason():
    variant, reason = select_fc_for_model([])
    assert variant is None
    assert reason == FC_INELIGIBLE_NO_BACKENDS


def test_select_fc_for_model_hybrid_two_compatible_backends_still_falls_back():
    # Both FA NHD and FlashInfer NHD individually map to OperationalNHD,
    # but a model that mixes them is hybrid — vLLM can't allocate a
    # uniform cross-layer cache across distinct backend classes, and the
    # LW path rejects multi-backend models. select_fc_for_model returns
    # HYBRID_BACKENDS so the connector logs the right reason and the
    # failure surfaces in LW with its NotImplementedError.
    backends = [
        _FakeBackend(_flashattn_shape, _flashattn_stride_nhd),
        _FakeBackend(_flashinfer_shape, _flashinfer_stride_nhd),
    ]
    variant, reason = select_fc_for_model(backends)
    assert variant is None
    assert reason == FC_INELIGIBLE_HYBRID_BACKENDS


def test_select_fc_for_model_hybrid_with_incompatible_backend_returns_hybrid_reason():
    # Hybrid detection short-circuits before per-backend probing — even
    # if one of the backends would individually fail (FlashInfer HND →
    # no variant), the reason reported is HYBRID, not BACKEND_NO_MATCH,
    # because the hybrid-ness is the load-bearing constraint.
    backends = [
        _FakeBackend(_flashattn_shape, _flashattn_stride_nhd),
        _FakeBackend(_flashinfer_shape, _flashinfer_stride_hnd),
    ]
    variant, reason = select_fc_for_model(backends)
    assert variant is None
    assert reason == FC_INELIGIBLE_HYBRID_BACKENDS


def test_select_fc_for_model_single_flashinfer_hnd_returns_no_match_reason():
    # Single backend with no FC variant — different log message than
    # hybrid. The connector reports HND2/MLA/missing-stride as the
    # specific reason rather than blaming hybridness.
    backends = [_FakeBackend(_flashinfer_shape, _flashinfer_stride_hnd)]
    variant, reason = select_fc_for_model(backends)
    assert variant is None
    assert reason == FC_INELIGIBLE_BACKEND_NO_MATCH


def test_select_fc_for_model_single_mla_returns_no_match_reason():
    backends = [_FakeBackend(_mla_indexer_shape, _mla_indexer_stride)]
    variant, reason = select_fc_for_model(backends)
    assert variant is None
    assert reason == FC_INELIGIBLE_BACKEND_NO_MATCH
