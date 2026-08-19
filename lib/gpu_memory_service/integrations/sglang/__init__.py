# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU Memory Service integration for SGLang.

Usage:
    from gpu_memory_service.integrations.sglang import setup_gms

    if server_args.load_format == "gms":
        server_args.override("gms", load_format=setup_gms(server_args))
"""

from __future__ import annotations

import logging
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


def setup_gms(server_args) -> Type["GMSModelLoader"]:
    """Setup GPU Memory Service for SGLang.

    Validates config and returns the GMSModelLoader class.
    Patches are applied automatically when GMSModelLoader is imported.

    Args:
        server_args: SGLang ServerArgs instance.

    Enables the memory saver on ``server_args`` -- GMS releases GPU memory
    through SGLang's memory-saver path -- and leaves ``load_format`` to the
    caller, which is the value this returns.

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

    # SGLang >= 0.5.16 freezes ServerArgs once resolved and offers override()
    # as the mutation point; the guard is opt-in there and unconditional from
    # 0.5.17. This package cannot import Dynamo's SGLang compat helper, so it
    # repeats the same two-line check.
    override = getattr(server_args, "override", None)
    if callable(override):
        override("gms", enable_memory_saver=True)
    else:  # SGLang < 0.5.16 has neither override() nor the guard.
        server_args.enable_memory_saver = True

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

    # Import triggers patches at module level
    from gpu_memory_service.integrations.sglang.model_loader import GMSModelLoader

    global _gms_initialized
    _gms_initialized = True

    logger.info("[GMS] Using GMSModelLoader...")
    return GMSModelLoader
