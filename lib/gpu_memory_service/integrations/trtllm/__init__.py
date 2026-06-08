# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU Memory Service integration for TensorRT-LLM.

The supported TRT-LLM path is KVCacheManagerV2. GMS owns model weights and
coordinates V2 KV slot leases from Python; the legacy V1 connector hook that
required a patched TRT-LLM wheel is intentionally not part of this runtime path.
"""

from __future__ import annotations

import logging
from typing import Any

from gpu_memory_service.integrations.common import patch_empty_cache
from gpu_memory_service.integrations.common.utils import (
    get_gms_lock_mode as _resolve_lock_mode,
)

logger = logging.getLogger(__name__)

__all__ = [
    "setup_gms",
    "get_gms_lock_mode",
]


def get_gms_lock_mode():
    from gpu_memory_service.integrations.trtllm.model_loader import (
        get_gms_lock_mode as _get_gms_lock_mode,
    )

    return _get_gms_lock_mode()


def setup_gms(model_loader_extra_config: dict[str, Any] | None = None) -> None:
    """Set up GMS integration for TensorRT-LLM. Call once before creating the engine."""
    extra = model_loader_extra_config or {}
    lock_mode = _resolve_lock_mode(extra)

    from gpu_memory_service.integrations.trtllm.model_loader import (
        patch_model_loader,
        set_gms_enabled,
        set_gms_lock_mode,
    )

    set_gms_enabled(True)
    set_gms_lock_mode(lock_mode)

    patch_empty_cache()
    from gpu_memory_service.integrations.trtllm.remote_code_cache import (
        patch_remote_code_cache,
    )

    patch_remote_code_cache()
    patch_model_loader()

    from gpu_memory_service.integrations.trtllm import install_kv_leases_v2

    install_kv_leases_v2.install()

    logger.info("[GMS] TensorRT-LLM integration enabled (mode=%s)", lock_mode)
