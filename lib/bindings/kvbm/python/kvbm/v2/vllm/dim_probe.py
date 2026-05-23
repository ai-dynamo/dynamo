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
    include_num_layers: bool = False,
) -> list[KvDim]:
    """Return the per-axis `KvDim` labels for tensors produced by `backend`.

    Per-layer (default): labels the **logical** post-permute shape returned
    by `get_kv_cache_shape(...)`. The per-layer kv_cache tensors vLLM
    passes to `register_kv_caches` are post-permute views over physically
    contiguous memory (non-row-major strides for HND); Rust's
    `relabel_to_physical_order` reorders by stride at NIXL bind time.

    Cross-layer (``include_num_layers=True``): used by the fully-contiguous
    registration path (`register_cross_layers_kv_cache`). vLLM hands us
    the underlying *physical contiguous allocation* (not the post-permute
    attention view), so the returned labels are in **physical byte
    order**. We build the cross-layer logical label list (per-layer with
    `KvDim.Layer` prepended at position 0), then apply the cross-layer
    `get_kv_cache_stride_order(include_num_layers_dimension=True)`
    permutation so the labels match `tensor.shape` index-for-index.

    Args:
        backend: An `AttentionBackend` subclass (from
            `vllm.distributed.kv_transfer.kv_connector.utils.get_current_attn_backends`).
        cache_dtype_str: Forwarded to `get_kv_cache_shape` — required for
            backends with FP8 inline-scale padding (Triton).
        use_mla: When `True`, an axis size of `1` is labelled `Outer`
            (the MLA fused K/V latent). Without this hint a leading `1`
            would be ambiguous.
        include_num_layers: When `True`, prepend `KvDim.Layer` and return
            labels in physical byte order (matches vLLM's contiguous FC
            allocation). Requires the backend to support
            `get_kv_cache_stride_order(include_num_layers_dimension=True)`.

    Raises:
        NotImplementedError: If a non-trailing axis cannot be matched to
            any sentinel (the backend's shape uses dims this prober does
            not recognise — file a bug). Trailing unmatched axes are
            labelled `Payload` to support DiffKV/TurboQuant.
        RuntimeError: If `include_num_layers=True` and the backend does
            not implement `get_kv_cache_stride_order(include_num_layers_dimension=True)`.
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
    if include_num_layers:
        # vLLM's cross-layer allocation: build the logical label list
        # (Layer prepended at position 0), then apply the cross-layer
        # stride-order permutation so the result is in *physical byte
        # order* — matching the contiguous tensor's `.shape`.
        logical_dims = [KvDim.Layer] + dims
        try:
            stride_order = backend.get_kv_cache_stride_order(
                include_num_layers_dimension=True
            )
        except (AttributeError, NotImplementedError) as e:
            raise RuntimeError(
                f"include_num_layers=True requires "
                f"{backend.__name__}.get_kv_cache_stride_order"
                f"(include_num_layers_dimension=True); got {type(e).__name__}: {e}"
            ) from e
        if len(stride_order) != len(logical_dims):
            raise RuntimeError(
                f"{backend.__name__} cross-layer stride_order length "
                f"{len(stride_order)} != logical dim count {len(logical_dims)} "
                f"(logical={[d.value for d in logical_dims]}, "
                f"stride_order={stride_order})"
            )
        # stride_order[i] is the logical-position index whose axis lands at
        # physical position i. So physical_dims[i] = logical_dims[stride_order[i]].
        dims = [logical_dims[stride_order[i]] for i in range(len(stride_order))]
    return dims


def derive_block_layout(
    backend: type,
    dims: list[KvDim],
    *,
    include_num_layers: bool = False,
) -> KvBlockLayout:
    """Classify the per-block dimension order as NHD vs HND.

    Compares the **physical** byte positions of `Page` and `HeadCount`:
    - `page_phys < head_phys` → NHD (tokens innermost-but-one)
    - `page_phys > head_phys` → HND (heads innermost-but-one)
    - missing stride order or missing `HeadCount` → `Unknown`

    Per-layer (default): ``dims`` is in logical order, so per-layer
    `get_kv_cache_stride_order(False)` is queried to map logical →
    physical positions.

    Cross-layer (``include_num_layers=True``): ``dims`` is already in
    physical byte order (see `probe_kv_dim_layout`), so positions in
    ``dims`` *are* physical positions — no stride_order lookup needed.

    Args:
        backend: the `AttentionBackend` subclass.
        dims: the labelled axis list returned by
            ``probe_kv_dim_layout`` (must use the same
            ``include_num_layers`` value passed here).
        include_num_layers: ``True`` for the cross-layer / FC path.
    """
    if KvDim.HeadCount not in dims or KvDim.Page not in dims:
        return KvBlockLayout.Unknown

    if include_num_layers:
        # `dims` is already physical for the cross-layer path.
        head_phys = dims.index(KvDim.HeadCount)
        page_phys = dims.index(KvDim.Page)
    else:
        try:
            stride_order = backend.get_kv_cache_stride_order(
                include_num_layers_dimension=False
            )
        except (AttributeError, NotImplementedError):
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


# Cross-layer physical byte orderings supported by
# `FullyContiguousLayout` in `kvbm-physical`. Each tuple matches the byte
# layout that one `KvBlockLayout` variant produces in `layout_view()`. A
# backend whose cross-layer physical labels equal one of these tuples can
# register via the FC path with the matched `KvBlockLayout`; anything else
# falls back to per-layer (LW).
#
# Source of truth: `lib/kvbm-physical/src/layout/fully_contiguous.rs:266-395`.
#
# Byte-exact GPU validation that the existing `universal_from_block` /
# `block_from_universal` planner kernels operate correctly on FC operands
# (which is what makes Tier 1's "no new kernel work needed" claim hold)
# lives in `lib/kvbm-physical/src/transfer/tests/planner_path.rs`:
#   - `use_planner_round_trip_nhd_via_universal`
#   - `use_planner_round_trip_hnd_via_universal`
#   - `use_planner_round_trip_nhd_to_hnd`
#   - `use_planner_round_trip_hnd_to_nhd`
# Each fills an FC tensor with a deterministic pattern, transfers through
# the cross-layout kernel, and asserts byte-equality after the roundtrip.
_FC_PHYSICAL_ORDERINGS: tuple[tuple[tuple[KvDim, ...], KvBlockLayout], ...] = (
    (
        (
            KvDim.Block,
            KvDim.Layer,
            KvDim.Outer,
            KvDim.Page,
            KvDim.HeadCount,
            KvDim.HeadSize,
        ),
        KvBlockLayout.OperationalNHD,
    ),
    (
        (
            KvDim.Block,
            KvDim.Layer,
            KvDim.Outer,
            KvDim.HeadCount,
            KvDim.Page,
            KvDim.HeadSize,
        ),
        KvBlockLayout.OperationalHND,
    ),
    (
        (
            KvDim.Block,
            KvDim.HeadCount,
            KvDim.Layer,
            KvDim.Outer,
            KvDim.Page,
            KvDim.HeadSize,
        ),
        KvBlockLayout.Universal,
    ),
)


# Reason `select_fc_for_model` returns `None`. Strings (not exceptions)
# so the caller can format human-readable log lines without re-deriving
# what went wrong.
FC_INELIGIBLE_NO_BACKENDS = "no_attention_backends"
FC_INELIGIBLE_HYBRID_BACKENDS = "hybrid_attention_backends"
FC_INELIGIBLE_BACKEND_NO_MATCH = "backend_no_fc_variant"


def select_fc_for_model(
    backends: list[type],
    *,
    cache_dtype_str: str = "auto",
) -> tuple[KvBlockLayout | None, str | None]:
    """Decide whether a *whole model* can register through FC, and which variant.

    Returns ``(variant, None)`` when FC is viable and ``(None, reason)`` when
    LW must be used. The reason is one of the ``FC_INELIGIBLE_*`` string
    constants so callers can format consistent log messages.

    KVBM v2 does NOT currently support hybrid models (multiple distinct
    attention backends) in either the FC or LW registration paths —
    ``register_kv_caches`` bails on ``len(self._attn_backends) != 1`` and
    ``register_cross_layers_kv_cache`` mirrors that check. Returning
    ``(None, FC_INELIGIBLE_HYBRID_BACKENDS)`` from here keeps the failure
    site in LW (one authoritative error message) instead of vLLM trying
    FC, failing to allocate uniform, and falling back into LW's rejection
    by an indirect route.

    Hybrid kv_cache_groups on a *single* backend are NOT detected here
    (this helper sees only the dedup'd backend list, not the eventual
    kv_cache_config); those still surface as the LW
    ``NotImplementedError``.
    """
    if not backends:
        return (None, FC_INELIGIBLE_NO_BACKENDS)
    if len(backends) > 1:
        return (None, FC_INELIGIBLE_HYBRID_BACKENDS)
    variant = select_fc_variant(backends[0], cache_dtype_str=cache_dtype_str)
    if variant is None:
        return (None, FC_INELIGIBLE_BACKEND_NO_MATCH)
    return (variant, None)


def select_fc_variant(
    backend: type,
    *,
    cache_dtype_str: str = "auto",
) -> KvBlockLayout | None:
    """Return the FC variant a backend's cross-layer layout maps to, or `None`.

    Probes the backend with `include_num_layers=True` to get its physical
    cross-layer label order, then matches against the byte orderings that
    `FullyContiguousLayout` natively supports. The returned `KvBlockLayout`
    is what should be threaded to Rust's
    `PendingLayoutMode::FullyContiguous { block_layout }`.

    Returns `None` (meaning "use the per-layer LW path") when:
    - the backend lacks
      `get_kv_cache_stride_order(include_num_layers_dimension=True)`,
    - the cross-layer physical order doesn't match any supported FC variant
      (e.g. FlashInfer HND's `[Block, Outer, HeadCount, Layer, Page, HeadSize]`),
    - the backend is MLA (3-dim shape, no `Outer`/`HeadCount`).

    Args:
        backend: An `AttentionBackend` subclass.
        cache_dtype_str: Forwarded to `get_kv_cache_shape`.

    Callers should treat `None` as "this backend cannot use FC; fall back
    to per-layer registration via `register_kv_caches`".
    """
    try:
        physical_dims = probe_kv_dim_layout(
            backend,
            cache_dtype_str=cache_dtype_str,
            use_mla=False,
            include_num_layers=True,
        )
    except Exception:
        # Anything that prevents us from characterising the backend means
        # it's not FC-eligible — caller falls back to per-layer. Examples:
        # missing cross-layer stride_order (RuntimeError), unrecognised
        # non-trailing axis (NotImplementedError), MLA assertion failure
        # when probed without `use_mla=True` (AssertionError from
        # `mla/indexer.py:107`), or a backend whose probe call rejects
        # the sentinel dtype.
        return None

    physical_tuple = tuple(physical_dims)
    for ordering, variant in _FC_PHYSICAL_ORDERINGS:
        if physical_tuple == ordering:
            return variant
    return None


def build_dim_layout_from_tensor(
    backend: type,
    *,
    tensor_shape: Sequence[int],
    cache_dtype_str: str = "auto",
    use_mla: bool = False,
    include_num_layers: bool = False,
) -> tuple[list[KvDim], list[int]]:
    """One-shot: probe `backend` for labels, then pair them with the
    actual `tensor_shape` for sizes.

    Using `tensor.shape()` (not `kv_cache_spec.block_size` / `num_blocks`)
    sidesteps vLLM's `kernel_block_size != spec.block_size` case
    (`kv_connector_model_runner_mixin.py:235-238`): whatever sizes the
    tensor actually carries are the authoritative ones for KVBM's layout
    arithmetic.

    Args:
        include_num_layers: When ``True``, the probe prepends ``KvDim.Layer``
            for the cross-layer (FC) registration path; ``tensor_shape``
            must then begin with ``num_layers``.

    Returns:
        ``(dims, sizes)`` where ``len(dims) == len(sizes) == len(tensor_shape)``.

    Raises:
        ValueError: If the probed dim count does not match the tensor
            rank — indicates a bug in either the backend or this prober.
    """
    dims = probe_kv_dim_layout(
        backend,
        cache_dtype_str=cache_dtype_str,
        use_mla=use_mla,
        include_num_layers=include_num_layers,
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
