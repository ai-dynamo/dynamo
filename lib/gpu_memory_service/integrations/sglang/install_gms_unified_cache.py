# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)
_BACKEND = "gms"


def _directory_mode() -> str:
    return os.environ.get("GMS_KV_DIRECTORY_MODE", "off").strip().lower()


def _enabled() -> bool:
    # Ring-only mode still gates native page allocation, but must not retain
    # pages in a cache that has no authoritative adoption/reclamation path.
    return _directory_mode() == "authoritative"


def _unsupported_execution_modes(config) -> list[str]:
    reasons = []
    numeric_modes = (
        ("tensor parallelism (requires the TP agreement integration)", ("tp_size",)),
        ("data parallelism", ("dp_size",)),
        ("pipeline parallelism", ("pp_size",)),
        ("decode context parallelism", ("dcp_size",)),
        ("attention context parallelism", ("attn_cp_size", "attention_cp_size")),
    )
    for reason, names in numeric_modes:
        if any(int(getattr(config, name, 1) or 1) > 1 for name in names):
            reasons.append(reason)
    if any(
        bool(getattr(config, name, False))
        for name in ("enable_prefill_cp", "enable_prefill_context_parallel")
    ):
        reasons.append("prefill context parallelism")
    if bool(getattr(config, "enable_dp_attention", False)):
        reasons.append("data-parallel attention")
    disaggregation_mode = str(
        getattr(config, "disaggregation_mode", "null") or "null"
    ).lower()
    if disaggregation_mode not in ("", "none", "null"):
        reasons.append(f"prefill/decode disaggregation ({disaggregation_mode})")
    return reasons


def _validate(ctx) -> None:
    reasons = []
    if ctx.disable_radix_cache:
        reasons.append("disabled radix cache")
    if ctx.is_hybrid_swa:
        reasons.append("sliding-window attention")
    if ctx.is_hybrid_ssm:
        reasons.append("SSM/Mamba state")
    if getattr(ctx, "is_dsa", False):
        reasons.append("dynamic sparse attention")
    if ctx.enable_hierarchical_cache:
        reasons.append("hierarchical cache")
    params = ctx.params
    if int(getattr(params, "tp_world_size", 1) or 1) > 1:
        reasons.append("tensor parallelism (requires the TP agreement integration)")
    reasons.extend(_unsupported_execution_modes(getattr(ctx, "server_args", ctx)))
    allocator = params.token_to_kv_pool_allocator
    if not hasattr(allocator, "_gms_kv_leases_by_page"):
        reasons.append("allocator without GMS leases")
    kvcache = allocator.get_kvcache()
    if getattr(kvcache, "_gms_persistent_kv", False) is not True:
        reasons.append("KV pool outside GMS persistent memory")
    if params.enable_session_radix_cache:
        reasons.append("session radix cache")
    if getattr(getattr(ctx, "server_args", None), "enable_streaming_session", False):
        reasons.append("streaming sessions")
    if params.is_eagle or params.mtp_draft_device_pools:
        reasons.append("speculative decoding")
    if reasons:
        raise ValueError(
            "GMS persistent SGLang HBM currently supports only dense FULL KV: "
            + ", ".join(reasons)
        )


def _factory(ctx):
    _validate(ctx)
    from gpu_memory_service.integrations.sglang.gms_unified_cache import (
        make_gms_unified_cache_class,
    )
    from sglang.srt.mem_cache.unified_cache.components import ComponentType

    ctx.params.tree_components = (ComponentType.FULL,)
    return make_gms_unified_cache_class()(ctx.params)


def install() -> bool:
    if not _enabled():
        return False
    from sglang.srt.mem_cache.registry import (
        get_radix_cache_factory,
        register_radix_cache_backend,
    )

    existing = get_radix_cache_factory(_BACKEND)
    if existing is _factory:
        return False
    if existing is not None:
        raise RuntimeError(f"SGLang cache backend {_BACKEND!r} is already registered")
    register_radix_cache_backend(_BACKEND, _factory)
    return True


def configure(server_args) -> bool:
    if _directory_mode() == "shadow":
        raise ValueError(
            "SGLang persistent KV requires GMS_KV_DIRECTORY_MODE=authoritative; "
            "shadow mode cannot safely retain or adopt native radix-cache pages"
        )
    if not _enabled():
        return False
    custom_pool = os.environ.get("SGLANG_MOONCAKE_CUSTOM_MEM_POOL")
    if custom_pool:
        raise ValueError(
            "GMS persistent KV cannot be combined with "
            f"SGLANG_MOONCAKE_CUSTOM_MEM_POOL={custom_pool!r}"
        )
    from sglang.srt.arg_groups import overrides

    resolving_view = getattr(overrides, "resolving_view", None)
    resolved = (
        resolving_view(server_args) if resolving_view is not None else server_args
    )
    advanced_modes = {
        "disabled radix cache": bool(getattr(resolved, "disable_radix_cache", False)),
        "page-major KV layout": bool(
            getattr(resolved, "enable_page_major_kv_layout", False)
        ),
        "unified memory": bool(getattr(resolved, "enable_unified_memory", False)),
        "hierarchical cache": bool(
            getattr(resolved, "enable_hierarchical_cache", False)
        ),
        "session radix cache": bool(
            getattr(resolved, "enable_session_radix_cache", False)
        ),
        "streaming sessions": bool(
            getattr(resolved, "enable_streaming_session", False)
        ),
        "speculative decoding": getattr(resolved, "speculative_algorithm", None)
        is not None,
    }
    unsupported = [reason for reason, enabled in advanced_modes.items() if enabled]
    unsupported.extend(_unsupported_execution_modes(resolved))
    if unsupported:
        raise ValueError(
            "GMS persistent KV currently supports only dense FULL KV: "
            + ", ".join(unsupported)
        )
    install()
    selected = resolved.radix_cache_backend
    if selected not in (None, _BACKEND):
        raise ValueError(
            "GMS persistent KV cannot be combined with SGLang cache backend "
            f"{selected!r}"
        )
    declare_late_resolution = getattr(overrides, "declare_late_resolution", None)
    if declare_late_resolution is not None:
        declare_late_resolution(server_args, "dynamo.gms", radix_cache_backend=_BACKEND)
    else:
        override = getattr(server_args, "override", None)
        if callable(override):
            override("dynamo.gms", radix_cache_backend=_BACKEND)
        else:
            server_args.radix_cache_backend = _BACKEND
    return True
