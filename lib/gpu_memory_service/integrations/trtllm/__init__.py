# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU Memory Service integration for TensorRT-LLM.

Usage:
    from gpu_memory_service.integrations.trtllm import setup_gms

    if config.load_format == "gms":
        setup_gms(config.model_loader_extra_config)
"""

from __future__ import annotations

import logging
from typing import Any

from gpu_memory_service.integrations.common import patch_empty_cache
from gpu_memory_service.integrations.common.utils import (
    get_gms_lock_mode as resolve_gms_lock_mode,
)
from gpu_memory_service.integrations.trtllm.model_loader import (
    get_gms_lock_mode,
    patch_model_loader,
    set_gms_enabled,
    set_gms_lock_mode,
)

logger = logging.getLogger(__name__)


def setup_gms(model_loader_extra_config: dict[str, Any] | None = None) -> None:
    """Setup GPU Memory Service integration for TensorRT-LLM."""
    extra = model_loader_extra_config or {}
    lock_mode = resolve_gms_lock_mode(extra)

    set_gms_enabled(True)
    set_gms_lock_mode(lock_mode)

    patch_empty_cache()
    patch_model_loader()

    logger.info("[GMS] TensorRT-LLM integration enabled")
