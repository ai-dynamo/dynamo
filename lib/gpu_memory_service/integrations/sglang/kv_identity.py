# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent KV identity helpers for the SGLang GMS integration."""

from __future__ import annotations

import os

from gpu_memory_service.integrations.common.utils import (
    env_enabled_by_default,
    get_gms_persistent_kv_engine_id,
)


def truthy_env(name: str, *, default: bool = False) -> bool:
    return env_enabled_by_default(name, default=default)


def shared_kv_enabled() -> bool:
    return truthy_env(
        "GMS_SGLANG_SHARED_KV",
        default=truthy_env("DYN_GMS_FAILOVER_SHADOW_MODE", default=False),
    )


def private_bootstrap_kv_enabled() -> bool:
    requested = truthy_env(
        "DYN_SGLANG_GMS_PRIVATE_BOOTSTRAP_KV",
        default=truthy_env("GMS_SGLANG_PRIVATE_BOOTSTRAP_KV", default=False),
    )
    if not requested:
        return False
    if not truthy_env("DYN_GMS_FAILOVER_SHADOW_MODE", default=False):
        return True
    return _member_id() != _primary_member_id()


def stable_engine_id(device: int) -> str:
    return get_gms_persistent_kv_engine_id(
        "sglang", device, "GMS_SGLANG_VMM_IPC_ENGINE_ID"
    )


def allocator_tag(device: int) -> str:
    """Process-local Torch allocator tag for this device's KV pool."""
    return f"kv_pool:cuda{int(device)}"


def _member_id() -> str:
    for name in ("ENGINE_ID", "DYN_WORKER_ID", "HOSTNAME"):
        value = os.environ.get(name)
        if value:
            return value
    return str(os.getpid())


def _primary_member_id() -> str:
    return os.environ.get("DYN_GMS_FAILOVER_PRIMARY_ENGINE_ID", "0")


def allocation_engine_id(device: int) -> str:
    engine_id = stable_engine_id(device)
    if private_bootstrap_kv_enabled():
        return f"{engine_id}|bootstrap={_member_id()}"
    return engine_id


def allocation_shared() -> bool:
    return shared_kv_enabled() and not private_bootstrap_kv_enabled()


def promotion_engine_id(device: int) -> str:
    return stable_engine_id(device)


def release_private_bootstrap_kv_pool(manager, engine_id: str, *, logger=None) -> int:
    try:
        allocations = list(manager.list_persistent(engine_id=engine_id))
    except Exception:  # noqa: BLE001
        if logger is not None:
            logger.warning(
                "[GMS] Failed to list private SGLang KV bootstrap allocations for %s",
                engine_id,
                exc_info=True,
            )
        return 0

    released = 0
    for allocation in allocations:
        tag = getattr(allocation, "tag", "")
        if not tag.startswith("kv_pool"):
            continue
        try:
            if manager.release_persistent(engine_id, tag):
                released += 1
        except Exception:  # noqa: BLE001
            if logger is not None:
                logger.warning(
                    "[GMS] Failed to release private SGLang KV bootstrap allocation %s/%s",
                    engine_id,
                    tag,
                    exc_info=True,
                )

    if released and logger is not None:
        logger.info(
            "[GMS] Released %d private SGLang KV bootstrap allocations for %s",
            released,
            engine_id,
        )
    return released
