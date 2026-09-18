# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed installation of SGLang persistent-KV integration hooks."""

from __future__ import annotations

import logging
from collections.abc import Callable

from gpu_memory_service.integrations.sglang.kv_identity import failover_hooks_required

logger = logging.getLogger(__name__)


def _try_install(name: str, installer: Callable[[], object], *, required: bool) -> None:
    try:
        installer()
    except Exception as exc:
        if required:
            raise RuntimeError(
                f"SGLang GMS shared-KV startup could not install {name}; "
                "the SGLang integration API may have changed"
            ) from exc
        logger.warning(
            "[GMS] Optional SGLang %s install failed; continuing without it",
            name,
            exc_info=True,
        )


def verify_kv_failover_hooks() -> None:
    """Reject an incomplete live integration when persistent failover is enabled."""
    if not failover_hooks_required():
        return

    from gpu_memory_service.integrations.sglang import (
        install_gms_unified_cache,
        install_kv_leases,
        install_vmm_ipc_kv,
    )

    missing = [
        name
        for name, installed in (
            (
                "token/page allocator lease hooks",
                install_kv_leases.lease_hooks_installed(),
            ),
            (
                "persistent VMM allocation hooks",
                install_vmm_ipc_kv.persistent_kv_hooks_installed(),
            ),
            (
                "UnifiedRadixCache backend",
                install_gms_unified_cache.cache_backend_installed(),
            ),
        )
        if not installed
    ]
    if missing:
        raise RuntimeError(
            "SGLang GMS shared-KV startup is incomplete; refusing to serve "
            "without required hook(s): "
            + ", ".join(missing)
            + ". Check for SGLang API drift or explicitly disable shared/"
            "authoritative KV failover."
        )


def install_and_verify_kv_failover_hooks() -> None:
    """Install all SGLang KV hooks and verify the live or armed methods."""
    from gpu_memory_service.integrations.sglang import (
        install_gms_unified_cache,
        install_kv_leases,
        install_vmm_ipc_kv,
    )

    required = failover_hooks_required()
    _try_install(
        "persistent VMM hook", install_vmm_ipc_kv.install_lazy, required=required
    )
    _try_install("lease hooks", install_kv_leases.install, required=required)
    _try_install(
        "UnifiedRadixCache backend",
        install_gms_unified_cache.install,
        required=required,
    )
    verify_kv_failover_hooks()
