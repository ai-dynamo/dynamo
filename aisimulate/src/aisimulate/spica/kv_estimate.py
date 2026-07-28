# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KV-cache feasibility for a parallel shape, via AIConfigurator's
``estimate_kv_cache`` (aiconfigurator#1159).

A parallel shape is *feasible* for a workload iff its per-rank KV-cache capacity
(in tokens, AFTER the quantized weights + the backend's activation reservations
are subtracted from VRAM) can hold at least one longest sequence::

    total_kv_size_tokens > max_seq_len

This replaces the coarse BF16 weight floor with the backend's real memory model:
it uses the model's actual (often FP8) weights and it distinguishes MoE sharding
strategies the weight floor cannot — at the *same* GPU count a DEP shape can OOM
while TEP / pure-TP fit, because the experts are sharded differently across ranks
(e.g. DeepSeek-V3 on gb200 at 4 GPUs: DEP has no KV budget, TEP holds ~350k tokens).

Native only: the estimate reads the perf database for ``(hardware_sku, backend)``
(:func:`get_latest_database_version` resolves the version). SKUs without a perf
DB raise :class:`NoPerfDatabase`; the naive fallback is intentionally disabled
because it mis-models MoE expert sharding.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable
from typing import Any

from aiconfigurator.sdk.perf_database import get_latest_database_version

from dynamo._internal.aic import AicMemoryEstimatorUnavailableError

from .parallel_enum import ParallelShape

# Runtime knobs the estimate needs; defaults mirror AIC's memory-estimation tests.
DEFAULT_MAX_NUM_TOKENS = 8192
DEFAULT_MAX_BATCH_SIZE = 256
DEFAULT_MEMORY_FRACTION = 0.9


class NoPerfDatabase(RuntimeError):
    """No perf database for this ``(hardware_sku, backend)`` — cannot estimate KV cache."""


def _load_memory_estimator() -> Callable[..., dict[str, Any]]:
    """Load AIC's optional unified KV-cache estimator on first use.

    AIC 0.9, which Dynamo currently supports, does not provide
    ``aiconfigurator.sdk.memory``. Keep ordinary Spica imports working with that
    release, while preserving transitive import failures from an otherwise present
    estimator module.
    """
    try:
        memory = importlib.import_module("aiconfigurator.sdk.memory")
    except ModuleNotFoundError as exc:
        if exc.name == "aiconfigurator.sdk.memory":
            raise AicMemoryEstimatorUnavailableError(
                "aiconfigurator.sdk.memory is required for experimental Spica "
                "KV-cache estimation; install an AIConfigurator release with the "
                "compatible estimator (AIC 0.10 or newer)"
            ) from exc
        raise
    return memory.estimate_kv_cache


def memory_fraction_kind(backend: str) -> str:
    """TRT-LLM budgets KV from *free* memory; vLLM / SGLang from *total*."""
    return "of_free" if backend == "trtllm" else "of_total"


def resolve_backend_version(hardware_sku: str, backend: str) -> str:
    """Latest perf-DB version for the SKU/backend (required by the native estimate)."""
    version = get_latest_database_version(hardware_sku, backend)
    if version is None:
        raise NoPerfDatabase(
            f"no perf database for hardware_sku={hardware_sku!r}, backend={backend!r}; "
            f"KV-cache estimation needs one (pick a supported SKU)"
        )
    return version


def estimate_kv_tokens(
    shape: ParallelShape,
    *,
    model_name: str,
    hardware_sku: str,
    backend: str,
    backend_version: str,
    max_num_tokens: int = DEFAULT_MAX_NUM_TOKENS,
    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
    memory_fraction: float = DEFAULT_MEMORY_FRACTION,
    nextn: int = 0,
) -> int | None:
    """Per-rank KV-cache capacity (in tokens) for ``shape``, or ``None`` when the
    shape leaves no KV budget (weights + activations already fill VRAM -> OOM).

    Genuine estimation errors (bad inputs, unsupported model) propagate.
    """
    try:
        est = _load_memory_estimator()(
            model_name,
            hardware_sku,
            backend,
            backend_version=backend_version,
            max_num_tokens=max_num_tokens,
            max_batch_size=max_batch_size,
            memory_fraction_kind=memory_fraction_kind(backend),
            memory_fraction_value=memory_fraction,
            tp_size=shape.tp,
            pp_size=shape.pp,
            attention_dp_size=shape.dp,
            moe_tp_size=shape.moe_tp,
            moe_ep_size=shape.moe_ep,
            nextn=nextn,
            allow_naive_fallback=False,
        )
    except ValueError as exc:
        msg = str(exc)
        # Shape-specific infeasibility -> skip this shape (don't abort the whole enumeration):
        #  - "no KV budget": weights + activations already fill VRAM (the shape OOMs).
        #  - "Invalid quantized MoE configuration": the shape's moe_tp/moe_ep doesn't evenly
        #    shard the model's (FP8 block-)quantized MoE dims (e.g. MiniMax-M2.5 moe_tp=8:
        #    moe_intermediate_size 1536 / 8 not divisible by weight_block_size 128). This is a
        #    property of *this shape*, not the model, so the enumerator should just drop it.
        if "no KV budget" in msg or "Invalid quantized MoE configuration" in msg:
            return None
        raise
    return int(est["total_kv_size_tokens"])


def feasible_shape_tokens(
    shapes: Iterable[ParallelShape],
    *,
    model_name: str,
    hardware_sku: str,
    backend: str,
    max_seq_len: int,
    backend_version: str | None = None,
    max_num_tokens: int = DEFAULT_MAX_NUM_TOKENS,
    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
    memory_fraction: float = DEFAULT_MEMORY_FRACTION,
) -> dict[ParallelShape, int]:
    """Map each *feasible* shape to its KV-cache token capacity.

    A shape is feasible iff ``estimate_kv_tokens(shape) > max_seq_len``. Shapes
    that OOM (no KV budget) or fall short are omitted. Estimates are computed once
    per distinct shape, so repeated shapes across replica counts are free.
    """
    if backend_version is None:
        backend_version = resolve_backend_version(hardware_sku, backend)
    feasible: dict[ParallelShape, int] = {}
    for shape in dict.fromkeys(shapes):  # dedup, preserve first-seen order
        tokens = estimate_kv_tokens(
            shape,
            model_name=model_name,
            hardware_sku=hardware_sku,
            backend=backend,
            backend_version=backend_version,
            max_num_tokens=max_num_tokens,
            max_batch_size=max_batch_size,
            memory_fraction=memory_fraction,
        )
        if tokens is not None and tokens > max_seq_len:
            feasible[shape] = tokens
    return feasible
