# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration constants and environment handling for GPU Memory Service vLLM integration."""

from __future__ import annotations

import os
from typing import Any, Optional

# Single source of truth for environment variable checks
GMS_ENABLED = os.environ.get("GPU_MEMORY_SERVICE_VLLM_AUTO_REGISTER", "").lower() in (
    "1",
    "true",
    "yes",
)

DEFAULT_SOCKET_PATH_TEMPLATE = "/tmp/gpu_memory_service_{device}.sock"

# Keys that GPU Memory Service adds to model_loader_extra_config - must be stripped
# before passing to DefaultModelLoader which may validate unknown keys
GPU_MEMORY_SERVICE_EXTRA_CONFIG_KEYS = frozenset(
    {
        "gpu_memory_service_socket_path",
    }
)


def get_local_rank() -> int:
    """Get the local rank (GPU device index) for the current worker.

    Priority order:
    1. torch.cuda.current_device() if already set (vLLM sets this early)
    2. vLLM's world group local_rank
    3. LOCAL_RANK environment variable
    4. Default to 0
    """
    import torch

    # First check if CUDA device is already set (vLLM sets this in worker init)
    try:
        if torch.cuda.is_initialized():
            current_device = torch.cuda.current_device()
            if current_device != 0 or os.environ.get("LOCAL_RANK", "0") == "0":
                return current_device
    except Exception:
        pass

    # Try vLLM's distributed group
    try:
        from vllm.distributed import get_world_group

        return int(get_world_group().local_rank)
    except Exception:
        pass

    # Fall back to environment variable
    return int(os.environ.get("LOCAL_RANK", "0"))


def resolve_socket_path(
    load_config: Optional[Any] = None, device_index: Optional[int] = None
) -> str:
    """Get socket path from model_loader_extra_config (falls back to default).

    Args:
        load_config: vLLM LoadConfig object with model_loader_extra_config dict.
        device_index: Explicit device index to use for {device} placeholder.
                      If not provided, falls back to get_local_rank().

    Returns:
        Resolved socket path with {local_rank}/{device} placeholders expanded.
    """
    socket_path = None

    # Try model_loader_extra_config
    if load_config is not None:
        extra_config = getattr(load_config, "model_loader_extra_config", None) or {}
        socket_path = extra_config.get("gpu_memory_service_socket_path")

    # Fallback to environment variable or default
    if not socket_path:
        socket_path = os.environ.get(
            "GPU_MEMORY_SERVICE_SOCKET_PATH", DEFAULT_SOCKET_PATH_TEMPLATE
        )

    # Use explicit device_index if provided, otherwise fall back to get_local_rank()
    rank = device_index if device_index is not None else get_local_rank()
    # Support both {local_rank} and {device} placeholders
    if "{local_rank}" in socket_path:
        socket_path = socket_path.format(local_rank=rank)
    if "{device}" in socket_path:
        socket_path = socket_path.format(device=rank)

    return socket_path


def strip_gms_extra_config(load_config: Any) -> Any:
    """Return a copy of load_config with GPU Memory Service keys removed.

    vLLM's DefaultModelLoader may validate model_loader_extra_config and reject
    unknown keys. This strips GPU Memory Service-specific keys so we can delegate
    to DefaultModelLoader.
    """
    from dataclasses import replace

    if load_config is None:
        return load_config

    extra_config = getattr(load_config, "model_loader_extra_config", None) or {}
    if not extra_config:
        return load_config

    # Remove GPU Memory Service keys
    cleaned = {
        k: v
        for k, v in extra_config.items()
        if k not in GPU_MEMORY_SERVICE_EXTRA_CONFIG_KEYS
    }

    return replace(load_config, model_loader_extra_config=cleaned if cleaned else {})
