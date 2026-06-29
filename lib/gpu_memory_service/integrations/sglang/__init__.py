# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU Memory Service integration for SGLang.

Usage:
    from gpu_memory_service.integrations.sglang import setup_gms

    if server_args.load_format == "gms":
        server_args.load_format = setup_gms(server_args)
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from gpu_memory_service.integrations.sglang.model_loader import GMSModelLoader

logger = logging.getLogger(__name__)

# Module-level GMS lock mode + RO reconnect timeout, set by setup_gms() before
# loader is instantiated. Read by patches.py when creating GMSMemorySaverImpl.
_gms_lock_mode = None
_gms_ro_connect_timeout_ms = None
_gms_initialized = False


def is_gms_active() -> bool:
    """Return True if setup_gms() has been called successfully."""
    return _gms_initialized


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def configure_shared_failover_env() -> None:
    if not (
        _truthy_env("GMS_SGLANG_SHARED_KV")
        and _truthy_env("DYN_GMS_FAILOVER_SHADOW_MODE")
    ):
        return
    os.environ.setdefault("SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK", "0")
    logger.info(
        "[GMS] Disabled SGLang TP memory imbalance check for shared-GMS failover"
    )


def setup_gms(server_args) -> Type["GMSModelLoader"]:
    """Setup GPU Memory Service for SGLang.

    Validates config and returns the GMSModelLoader class.
    Patches are applied automatically when GMSModelLoader is imported.

    Args:
        server_args: SGLang ServerArgs instance.

    Returns:
        GMSModelLoader class to use as load_format.

    Raises:
        ValueError: If incompatible options are enabled.
    """
    # Validate config - GMS provides its own VA-stable unmap/remap for weights
    if getattr(server_args, "enable_weights_cpu_backup", False):
        raise ValueError(
            "Cannot use --enable-weights-cpu-backup with --load-format gms."
        )
    if getattr(server_args, "enable_draft_weights_cpu_backup", False):
        raise ValueError(
            "Cannot use --enable-draft-weights-cpu-backup with --load-format gms."
        )

    # GMS uses SGLang's torch_memory_saver regions as the allocation hook for
    # weights and KV. Force the adapter on for --load-format gms so allocations
    # are captured even when users did not pass --enable-memory-saver.
    server_args.enable_memory_saver = True

    configure_shared_failover_env()

    # The post-ready collective-timeout tightening for SGLang is applied PRECISELY,
    # not via a grace delay: patches.patch_serving_collective_timeout_for_gms wraps
    # Scheduler.run_event_loop (entered only AFTER model load + CUDA-graph capture in
    # Scheduler.__init__), so a tight 2-3s serving timeout can never fire during warmup.
    # That patch is installed at GMSModelLoader import time (model_loader.py), which
    # runs in every rank's scheduler process. Nothing to arm here.

    # Resolve lock mode and RO reconnect timeout from model_loader_extra_config
    # before patches fire.
    global _gms_lock_mode
    global _gms_ro_connect_timeout_ms
    extra = getattr(server_args, "model_loader_extra_config", None)
    if isinstance(extra, str):
        import json

        extra = json.loads(extra) if extra else {}
    extra = extra or {}

    from gpu_memory_service.integrations.common.utils import (
        get_gms_lock_mode,
        get_gms_ro_connect_timeout_ms,
    )

    _gms_lock_mode = get_gms_lock_mode(extra)
    _gms_ro_connect_timeout_ms = get_gms_ro_connect_timeout_ms(extra)

    from gpu_memory_service.integrations.sglang import (
        install_kv_leases,
        install_vmm_ipc_kv,
    )

    install_vmm_ipc_kv.install_lazy()
    install_kv_leases.install()

    # Import triggers patches at module level
    from gpu_memory_service.integrations.sglang.model_loader import GMSModelLoader

    global _gms_initialized
    _gms_initialized = True

    logger.info("[GMS] Using GMSModelLoader...")
    return GMSModelLoader
