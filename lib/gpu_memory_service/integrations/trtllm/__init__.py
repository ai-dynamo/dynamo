# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU Memory Service integration for TensorRT-LLM.

Usage:
    import json
    from gpu_memory_service.integrations.trtllm import setup_gms

    if config.load_format == "gms":
        raw = config.model_loader_extra_config
        extra = json.loads(raw) if isinstance(raw, str) else (raw or None)
        setup_gms(extra)
"""

from __future__ import annotations

import logging
import os
from typing import Any

from gpu_memory_service.integrations.common import patch_empty_cache
from gpu_memory_service.integrations.common.utils import (
    get_gms_lock_mode as _resolve_lock_mode,
)
from gpu_memory_service.integrations.trtllm.model_loader import (
    get_gms_lock_mode,
    patch_model_loader,
    set_gms_enabled,
    set_gms_lock_mode,
)
from gpu_memory_service.integrations.trtllm.mpi_bootstrap import (
    install_mpi_worker_bootstrap,
    set_extra_config,
)

logger = logging.getLogger(__name__)

__all__ = ["setup_gms", "get_gms_lock_mode"]


def setup_gms(model_loader_extra_config: dict[str, Any] | None = None) -> None:
    """Set up GMS integration for TensorRT-LLM. Call once before creating the engine."""
    extra = model_loader_extra_config or {}
    lock_mode = _resolve_lock_mode(extra)

    set_gms_enabled(True)
    set_gms_lock_mode(lock_mode)

    patch_empty_cache()
    patch_model_loader()

    # Mark GMS enabled in env so MPI worker children can detect it, and stash the
    # extra config so they get the same lock mode. Both must happen before the
    # MPI session bootstrap is installed — the child hook reads them back.
    os.environ["DYN_GMS_TRTLLM_ENABLED"] = "1"
    set_extra_config(extra)
    install_mpi_worker_bootstrap()

    logger.info("[GMS] TensorRT-LLM integration enabled (mode=%s)", lock_mode)
