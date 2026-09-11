# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)
_BACKEND = "gms"


def _enabled() -> bool:
    return (
        os.environ.get("GMS_KV_DIRECTORY_MODE", "off").lower()
        in (
            "shadow",
            "authoritative",
        )
        or os.environ.get("GMS_SGLANG_ENABLE_KV_RING") == "1"
    )


def _validate(ctx) -> None:
    reasons = []
    if int(getattr(ctx.params, "tp_world_size", 1) or 1) > 1:
        reasons.append("tensor parallelism (requires the TP agreement integration)")
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
    if not hasattr(params.token_to_kv_pool_allocator, "_gms_kv_leases_by_page"):
        reasons.append("allocator without GMS leases")
    if params.enable_session_radix_cache:
        reasons.append("session radix cache")
    if getattr(ctx.server_args, "enable_streaming_session", False):
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
    if not _enabled():
        return False
    install()
    from sglang.srt.arg_groups import overrides

    resolving_view = getattr(overrides, "resolving_view", None)
    resolved = (
        resolving_view(server_args) if resolving_view is not None else server_args
    )
    if int(getattr(resolved, "tp_size", 1) or 1) > 1:
        raise ValueError(
            "GMS persistent KV tensor parallelism requires the TP agreement integration"
        )
    if getattr(resolved, "enable_streaming_session", False):
        raise ValueError(
            "GMS persistent KV does not yet support SGLang streaming sessions"
        )
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
