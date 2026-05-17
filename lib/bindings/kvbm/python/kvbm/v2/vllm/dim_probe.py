# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Probe a vLLM `AttentionBackend` for canonical KV-cache axis labels.

This module replaces shape inference inside the KVBM v2 connector. Rather
than guessing which tensor axis is `num_blocks` / `num_kv_heads` / etc., we
call `attn_backend.get_kv_cache_shape(...)` with **sentinel** values for
each labelled dimension and read the position of each sentinel in the
returned shape. The result is an axis-by-axis label list (`KvDim`) plus an
NHD/HND classification (`KvBlockLayout`) derived from the per-layer stride
order.

Mirrors the structural pattern in vLLM's NIXL connector — see
`vllm/distributed/kv_transfer/kv_connector/utils.py:323` (`TpKVTopology`).

Sentinel choice
---------------
Sentinels must satisfy backend validation:
* FlashAttention/Triton reject `block_size % 16 != 0`
  (`vllm/v1/attention/backends/flash_attn.py:141`,
  `vllm/v1/attention/backends/triton_attn.py:311`).
* FlashAttention rejects `head_size % 8 != 0`
  (`vllm/v1/attention/backends/flash_attn.py:175`).

They must also be pairwise distinct, none equal to ``2`` (the K/V outer
axis), and far outside any plausible real value so a backend cannot
accidentally produce one of them as a *computed* axis (DiffKV's
``head_size + head_size_v``, TurboQuant's ``slot_size_aligned``, FP8
DS-MLA's ``656``).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Sequence


class KvDim(str, Enum):
    """Per-axis label sent across the FFI boundary as a string.

    Values match the Rust `kvbm_common::KvDim` variants exactly; the
    binding parses them by name (see
    `lib/bindings/kvbm/src/v2/connector/worker/mod.rs::parse_kv_dim`).
    """

    Block = "Block"
    Layer = "Layer"
    Outer = "Outer"
    Page = "Page"
    HeadCount = "HeadCount"
    HeadSize = "HeadSize"
    Payload = "Payload"


class KvBlockLayout(str, Enum):
    """Per-block dim ordering, sent across the FFI as a string.

    Values match `kvbm_common::KvBlockLayout` variant names. `Custom` is
    intentionally not exposed — universal/operational/unknown cover the
    cases vLLM's standard backends produce.
    """

    OperationalNHD = "OperationalNHD"
    OperationalHND = "OperationalHND"
    Universal = "Universal"
    Unknown = "Unknown"


# Sentinel values for the probe. See module docstring for rationale.
_S_BLOCKS = 1675664  # 104729 (prime) * 16 — satisfies block_size % 16 == 0
_S_PAGE = 4096  # multiple of 16; far above real block_size (≤ 256)
_S_HEAD = 10007  # prime; num_kv_heads has no alignment constraint
_S_HSZ = 1024  # multiple of 8; far above real head_size (≤ 256)

_SENTINEL_TO_DIM: dict[int, KvDim] = {
    _S_BLOCKS: KvDim.Block,
    _S_PAGE: KvDim.Page,
    _S_HEAD: KvDim.HeadCount,
    _S_HSZ: KvDim.HeadSize,
}

# All four sentinels and 2 (K/V outer) must be pairwise distinct so that
# axis labelling is unambiguous.
assert (
    len({_S_BLOCKS, _S_PAGE, _S_HEAD, _S_HSZ, 2}) == 5
), "dim_probe sentinels collided — bump the values in dim_probe.py"


def probe_kv_dim_layout(
    backend: type,
    *,
    cache_dtype_str: str = "auto",
    use_mla: bool = False,
) -> list[KvDim]:
    """Return the per-axis `KvDim` labels for tensors produced by `backend`.

    The returned list matches the **logical** axis order the connector sees
    after vLLM's `permute(inv_order)` view (per-layer registration only —
    cross-layer is a Milestone 2 concern with different physics).

    Args:
        backend: An `AttentionBackend` subclass (from
            `vllm.distributed.kv_transfer.kv_connector.utils.get_current_attn_backends`).
        cache_dtype_str: Forwarded to `get_kv_cache_shape` — required for
            backends with FP8 inline-scale padding (Triton).
        use_mla: When `True`, an axis size of `1` is labelled `Outer`
            (the MLA fused K/V latent). Without this hint a leading `1`
            would be ambiguous.

    Raises:
        NotImplementedError: If a non-trailing axis cannot be matched to
            any sentinel (the backend's shape uses dims this prober does
            not recognise — file a bug). Trailing unmatched axes are
            labelled `Payload` to support DiffKV/TurboQuant.
    """
    # MLA backends assert ``num_kv_heads == 1`` (see
    # ``vllm/v1/attention/backends/mla/indexer.py:107``); passing the
    # sentinel ``_S_HEAD`` would trip that assertion. The MLA shape has
    # no ``HeadCount`` axis to identify, so substituting the constant ``1``
    # is safe — a ``HeadCount`` axis cannot be present in the result.
    probe_num_kv_heads = 1 if use_mla else _S_HEAD
    probed: Sequence[int] = backend.get_kv_cache_shape(
        _S_BLOCKS,
        _S_PAGE,
        probe_num_kv_heads,
        _S_HSZ,
        cache_dtype_str=cache_dtype_str,
    )

    dims: list[KvDim] = []
    for i, value in enumerate(probed):
        is_last = i == len(probed) - 1
        dim = _SENTINEL_TO_DIM.get(value)
        if dim is not None:
            dims.append(dim)
        elif value == 2:
            dims.append(KvDim.Outer)
        elif value == 1 and use_mla:
            dims.append(KvDim.Outer)
        elif is_last:
            # DiffKV head_size + head_size_v, TurboQuant slot_size_aligned,
            # FP8 DS-MLA 656 — opaque per-token payload.
            dims.append(KvDim.Payload)
        else:
            raise NotImplementedError(
                f"Backend {backend.__name__}: unrecognised non-trailing axis "
                f"size {value} at position {i} in shape {tuple(probed)}; "
                f"file a bug — KVBM does not understand this layout"
            )
    return dims


def derive_block_layout(
    backend: type,
    dims: list[KvDim],
) -> KvBlockLayout:
    """Classify the per-block dimension order as NHD vs HND.

    Per-layer `tensor.shape()` is logical (after `permute(inv_order)`), so
    the labels alone do not distinguish the two memory layouts. We read
    `backend.get_kv_cache_stride_order(False)` and inspect the **physical**
    position of `Page` relative to `HeadCount`:
    - `page_phys < head_phys` → NHD (tokens innermost-but-one)
    - `page_phys > head_phys` → HND (heads innermost-but-one)
    - missing stride order or missing `HeadCount` → `Unknown`
    """
    try:
        stride_order = backend.get_kv_cache_stride_order(
            include_num_layers_dimension=False
        )
    except (AttributeError, NotImplementedError):
        return KvBlockLayout.Unknown

    if KvDim.HeadCount not in dims or KvDim.Page not in dims:
        return KvBlockLayout.Unknown

    head_logical = dims.index(KvDim.HeadCount)
    page_logical = dims.index(KvDim.Page)
    head_phys = stride_order.index(head_logical)
    page_phys = stride_order.index(page_logical)
    return (
        KvBlockLayout.OperationalNHD
        if page_phys < head_phys
        else KvBlockLayout.OperationalHND
    )


def build_dim_layout_from_tensor(
    backend: type,
    *,
    tensor_shape: Sequence[int],
    cache_dtype_str: str = "auto",
    use_mla: bool = False,
) -> tuple[list[KvDim], list[int]]:
    """One-shot: probe `backend` for labels, then pair them with the
    actual `tensor_shape` for sizes.

    Using `tensor.shape()` (not `kv_cache_spec.block_size` / `num_blocks`)
    sidesteps vLLM's `kernel_block_size != spec.block_size` case
    (`kv_connector_model_runner_mixin.py:235-238`): whatever sizes the
    tensor actually carries are the authoritative ones for KVBM's layout
    arithmetic.

    Returns:
        ``(dims, sizes)`` where ``len(dims) == len(sizes) == len(tensor_shape)``.

    Raises:
        ValueError: If the probed dim count does not match the tensor
            rank — indicates a bug in either the backend or this prober.
    """
    dims = probe_kv_dim_layout(
        backend, cache_dtype_str=cache_dtype_str, use_mla=use_mla
    )
    if len(dims) != len(tensor_shape):
        raise ValueError(
            f"probed {len(dims)} axes ({[d.value for d in dims]}) but tensor "
            f"has rank {len(tensor_shape)} (shape {tuple(tensor_shape)})"
        )
    sizes = [int(s) for s in tensor_shape]
    return dims, sizes


# ---------- Test-only fakes ----------------------------------------------------
#
# These aren't part of the public API — they let us exercise the probe in a
# unit test without importing vLLM.


class _FakeBackend:
    """Minimal `AttentionBackend` stand-in for unit tests."""

    __name__ = "_FakeBackend"

    def __init__(self, shape_fn: Any, stride_order: Any | None = None):
        self._shape_fn = shape_fn
        self._stride_order = stride_order

    def get_kv_cache_shape(
        self,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return self._shape_fn(num_blocks, block_size, num_kv_heads, head_size)

    def get_kv_cache_stride_order(
        self, include_num_layers_dimension: bool = False
    ) -> tuple[int, ...]:
        if self._stride_order is None:
            raise NotImplementedError
        return self._stride_order(include_num_layers_dimension)
