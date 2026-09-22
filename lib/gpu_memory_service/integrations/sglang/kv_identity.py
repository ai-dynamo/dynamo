# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent KV identity helpers for the SGLang GMS integration."""

from __future__ import annotations

from gpu_memory_service.integrations.common.utils import (
    env_enabled_by_default,
    get_gms_persistent_kv_engine_id,
)


def shared_kv_enabled() -> bool:
    return env_enabled_by_default(
        "GMS_SGLANG_SHARED_KV",
        default=env_enabled_by_default("DYN_GMS_FAILOVER_SHADOW_MODE", default=False),
    )


def reject_removed_private_bootstrap_kv() -> None:
    """Fail closed when a removed private-bootstrap KV knob is still set.

    The client-local scratch pool that made these safe is gone; with them on a
    shadow would allocate against the shared GMS KV pool before it owns the
    failover lock.
    """
    enabled = [
        name
        for name in (
            "DYN_SGLANG_GMS_PRIVATE_BOOTSTRAP_KV",
            "GMS_SGLANG_PRIVATE_BOOTSTRAP_KV",
        )
        if env_enabled_by_default(name, default=False)
    ]
    if enabled:
        raise RuntimeError(
            "SGLang GMS private-bootstrap KV is no longer supported: its "
            "client-local scratch isolation was removed. Unset "
            + ", ".join(enabled)
            + " and use the regular sleeping-shadow failover path instead."
        )


def stable_engine_id(device: int) -> str:
    return get_gms_persistent_kv_engine_id(
        "sglang", device, "GMS_SGLANG_VMM_IPC_ENGINE_ID"
    )


def allocator_tag(device: int) -> str:
    """Process-local Torch allocator tag for this device's KV pool."""
    return f"kv_pool:cuda{int(device)}"


def allocation_engine_id(device: int) -> str:
    reject_removed_private_bootstrap_kv()
    return stable_engine_id(device)


def allocation_shared() -> bool:
    reject_removed_private_bootstrap_kv()
    return shared_kv_enabled()
