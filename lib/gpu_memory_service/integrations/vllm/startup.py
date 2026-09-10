# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed installation of vLLM persistent-KV integration hooks."""

from __future__ import annotations

import logging
from collections.abc import Callable

from gpu_memory_service.integrations.vllm.kv_identity import failover_hooks_required

logger = logging.getLogger(__name__)


def _try_install(name: str, installer: Callable[[], object], *, required: bool) -> None:
    try:
        installer()
    except Exception as exc:
        if required:
            raise RuntimeError(
                f"vLLM GMS shared-KV startup could not install {name}; "
                "the vLLM integration API may have changed"
            ) from exc
        logger.warning(
            "[GMS] Optional vLLM %s install failed; continuing without it",
            name,
            exc_info=True,
        )


def verify_kv_failover_hooks() -> None:
    if not failover_hooks_required():
        return
    from gpu_memory_service.integrations.vllm import (
        install_kv_leases,
        install_vmm_ipc_kv,
    )

    missing = [
        name
        for name, installed in (
            (
                "BlockPool/KVCacheManager lease hooks",
                install_kv_leases.lease_hooks_installed(),
            ),
            ("EngineCore process hook", install_kv_leases.engine_core_hook_installed()),
            (
                "persistent VMM allocation and geometry hooks",
                install_vmm_ipc_kv.persistent_kv_hooks_installed(),
            ),
        )
        if not installed
    ]
    if missing:
        raise RuntimeError(
            "vLLM GMS shared-KV startup is incomplete; refusing to serve without "
            "required hook(s): "
            + ", ".join(missing)
            + ". Check for vLLM API drift or explicitly disable "
            "shared/authoritative KV failover."
        )


def install_and_verify_kv_failover_hooks() -> None:
    from gpu_memory_service.integrations.vllm import (
        install_kv_leases,
        install_vmm_ipc_kv,
    )

    required = failover_hooks_required()
    _try_install("lease hooks", install_kv_leases.install, required=required)
    _try_install(
        "EngineCore process hook",
        install_kv_leases.install_engine_core_hook,
        required=required,
    )
    _try_install("persistent VMM hooks", install_vmm_ipc_kv.install, required=required)
    verify_kv_failover_hooks()
