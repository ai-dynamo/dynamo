# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU Memory Service integration for SGLang."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Type

try:
    from sglang.srt.arg_groups.overrides import declare_resolution
except ImportError:
    # SGLang <0.5.19 predates declarations. Both container pins are now on
    # 0.5.21, so setup_gms() rejects such a build instead of running inert.
    declare_resolution = None

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

    if declare_resolution is not None:
        declare_resolution(server_args, "dynamo.gms", enable_memory_saver=True)
    else:
        # The XPU pin moved off SGLang 0.5.11, so every supported build
        # declares. Assigning enable_memory_saver directly is a silent no-op on
        # builds that resolve config separately from raw ServerArgs, which would
        # leave GMS regions inert -- fail loudly instead.
        raise RuntimeError(
            "GMS requires an SGLang build exposing declare_resolution(); " "none found."
        )

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
