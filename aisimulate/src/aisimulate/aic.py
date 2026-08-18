# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AISimulate-owned AIC KV-capacity materialization.

Both the engine-only and Dynamo replay compositions use this module so their
rank-local KV capacity is derived from the same defaults and AIC argument set.
"""

from __future__ import annotations

from typing import Any

DEFAULT_BACKEND_VERSIONS = {
    "vllm": "0.19.0",
    "sglang": "0.5.10",
    "trtllm": "1.3.0rc10",
}
DEFAULT_GPU_MEMORY_UTILIZATION = 0.9
DEFAULT_MEM_FRACTION_STATIC = 0.88
DEFAULT_FREE_GPU_MEMORY_FRACTION = 0.9

_DEFAULT_AIC_SYSTEM = "h200_sxm"
_DEFAULT_MAX_NUM_BATCHED_TOKENS = 8192
_DEFAULT_BLOCK_SIZES = {"vllm": 64, "sglang": 1, "trtllm": 32}


def materialize_aic_num_gpu_blocks(raw: dict[str, Any]) -> dict[str, Any]:
    """Return engine arguments with rank-local AIC KV capacity materialized."""

    lowered = dict(raw)
    attention_dp = lowered.get("aic_attention_dp_size")
    dp = attention_dp or 1
    configured_dp = lowered.get("dp_size") or 1
    has_aic_config = lowered.get("aic_backend") is not None or attention_dp is not None
    if has_aic_config and configured_dp > 1 and configured_dp != dp:
        raise ValueError(
            "dp_size must match aic_attention_dp_size for AIC-backed replay "
            f"(got dp_size={configured_dp}, aic_attention_dp_size={dp})"
        )
    if attention_dp is not None and dp > 1:
        lowered["dp_size"] = dp

    if lowered.get("num_gpu_blocks") is not None:
        return lowered
    backend = lowered.get("aic_backend")
    if backend is None:
        return lowered
    if not isinstance(backend, str) or backend not in DEFAULT_BACKEND_VERSIONS:
        supported = ", ".join(sorted(DEFAULT_BACKEND_VERSIONS))
        raise ValueError(
            f"AIC KV cache capacity estimation does not support {backend!r}; "
            f"supported backends: {supported}"
        )
    model = lowered.get("aic_model_path")
    if not model:
        raise ValueError(
            "AIC KV cache capacity estimation requires aic_model_path in engine args"
        )

    lowered["num_gpu_blocks"] = estimate_num_gpu_blocks(
        backend_name=backend,
        system=lowered.get("aic_system") or _DEFAULT_AIC_SYSTEM,
        model_path=model,
        tp_size=(
            lowered.get("aic_tp_size") if lowered.get("aic_tp_size") is not None else 1
        ),
        block_size=_resolve_block_size(lowered, backend),
        max_num_batched_tokens=(
            lowered.get("max_num_batched_tokens")
            if lowered.get("max_num_batched_tokens") is not None
            else _DEFAULT_MAX_NUM_BATCHED_TOKENS
        ),
        gpu_memory_utilization=lowered.get("gpu_memory_utilization"),
        mem_fraction_static=lowered.get("mem_fraction_static"),
        free_gpu_memory_fraction=lowered.get("free_gpu_memory_fraction"),
        backend_version=lowered.get("aic_backend_version"),
        pp_size=(
            lowered.get("aic_pp_size") if lowered.get("aic_pp_size") is not None else 1
        ),
        moe_tp_size=lowered.get("aic_moe_tp_size"),
        moe_ep_size=lowered.get("aic_moe_ep_size"),
        attention_dp_size=attention_dp,
        gemm_dtype=lowered.get("aic_gemm_dtype"),
        moe_dtype=lowered.get("aic_moe_dtype"),
        fmha_dtype=lowered.get("aic_fmha_dtype"),
        kv_cache_dtype=lowered.get("aic_kv_cache_dtype"),
        comm_dtype=lowered.get("aic_comm_dtype"),
        systems_path=lowered.get("systems_path"),
    )
    return lowered


def estimate_num_gpu_blocks(
    *,
    backend_name: str,
    system: str,
    model_path: str,
    tp_size: int,
    block_size: int,
    max_num_batched_tokens: int,
    gpu_memory_utilization: float | None = None,
    mem_fraction_static: float | None = None,
    free_gpu_memory_fraction: float | None = None,
    backend_version: str | None = None,
    pp_size: int = 1,
    moe_tp_size: int | None = None,
    moe_ep_size: int | None = None,
    attention_dp_size: int | None = None,
    gemm_dtype: str | None = None,
    moe_dtype: str | None = None,
    fmha_dtype: str | None = None,
    kv_cache_dtype: str | None = None,
    comm_dtype: str | None = None,
    systems_path: str | None = None,
) -> int:
    """Estimate per-rank KV blocks using the replay-wide AIC contract.

    NextN is intentionally absent. AIC currently can return negative KV
    capacity for Eagle when speculative-decoding state is included. Timing
    compilation still receives NextN; only capacity estimation omits it.
    """

    if backend_name not in DEFAULT_BACKEND_VERSIONS:
        supported = ", ".join(sorted(DEFAULT_BACKEND_VERSIONS))
        raise ValueError(
            f"AIC KV cache capacity estimation does not support {backend_name!r}; "
            f"supported backends: {supported}"
        )
    from aiconfigurator_core.sdk.memory import (
        estimate_num_gpu_blocks as aic_estimate_num_gpu_blocks,
    )

    if backend_name == "trtllm":
        memory_fraction_kind = "of_free"
        memory_fraction_value = (
            free_gpu_memory_fraction
            if free_gpu_memory_fraction is not None
            else DEFAULT_FREE_GPU_MEMORY_FRACTION
        )
    elif backend_name == "sglang":
        memory_fraction_kind = "of_total"
        memory_fraction_value = (
            mem_fraction_static
            if mem_fraction_static is not None
            else DEFAULT_MEM_FRACTION_STATIC
        )
    else:
        memory_fraction_kind = "of_total"
        memory_fraction_value = (
            gpu_memory_utilization
            if gpu_memory_utilization is not None
            else DEFAULT_GPU_MEMORY_UTILIZATION
        )

    return int(
        aic_estimate_num_gpu_blocks(
            model_path,
            system,
            backend_name,
            backend_version=(
                backend_version
                if backend_version is not None
                else DEFAULT_BACKEND_VERSIONS[backend_name]
            ),
            scheduler_block_size=block_size,
            max_num_tokens=max_num_batched_tokens,
            max_batch_size=1,
            memory_fraction_kind=memory_fraction_kind,
            memory_fraction_value=memory_fraction_value,
            tp_size=tp_size,
            pp_size=pp_size,
            attention_dp_size=(
                attention_dp_size if attention_dp_size is not None else 1
            ),
            moe_tp_size=moe_tp_size,
            moe_ep_size=moe_ep_size,
            gemm_quant_mode=_quant_mode_name("gemm", gemm_dtype),
            moe_quant_mode=_quant_mode_name("moe", moe_dtype),
            fmha_quant_mode=_quant_mode_name("fmha", fmha_dtype),
            kvcache_quant_mode=_quant_mode_name("kvcache", kv_cache_dtype),
            comm_quant_mode=_quant_mode_name("comm", comm_dtype),
            systems_path=systems_path,
        )
    )


def _resolve_block_size(raw: dict[str, Any], backend: str) -> int:
    block_size = raw.get("block_size")
    if block_size is not None:
        return int(block_size)
    if backend == "sglang":
        sglang = raw.get("sglang")
        if isinstance(sglang, dict) and sglang.get("page_size") is not None:
            return int(sglang["page_size"])
    return _DEFAULT_BLOCK_SIZES[backend]


def _quant_mode_name(field: str, value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"AIC {field} quant mode must be a string when set")
    normalized = value.strip()
    if not normalized or normalized.lower() in {"auto", "none", "null"}:
        return None
    if normalized == "int4":
        normalized = "int4_wo"

    from aiconfigurator_core.sdk import common

    enum_cls = {
        "gemm": common.GEMMQuantMode,
        "moe": common.MoEQuantMode,
        "fmha": common.FMHAQuantMode,
        "kvcache": common.KVCacheQuantMode,
        "comm": common.CommQuantMode,
    }[field]
    try:
        return enum_cls[normalized].name
    except KeyError:
        allowed = ", ".join(member.name for member in enum_cls)
        raise ValueError(
            f"unsupported AIC {field} quant mode {value!r}; supported values: {allowed}"
        ) from None
