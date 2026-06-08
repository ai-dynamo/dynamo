# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for GPU Memory Service."""

import logging
import os
import tempfile
from typing import NoReturn

logger = logging.getLogger(__name__)


# Canonical names for GMS-related environment variables. Defined here so
# operator code, launcher code, and engine integration code all reference
# one source of truth — keeping these in lockstep with the Go-side
# constants in deploy/operator/internal/gms/gms.go.
ENV_VMM_GRANULARITY = "DYN_GMS_VMM_GRANULARITY"


def fail(message: str, *args, exc_info=None) -> NoReturn:
    logger.critical(message, *args, exc_info=exc_info)
    logging.shutdown()
    os._exit(1)


_uuid_cache: dict[tuple[int, str | None], str] = {}


def invalidate_uuid_cache() -> None:
    """Clear cached GPU UUIDs. Call after CRIU restore when GPU assignment may change."""
    _uuid_cache.clear()


def _visible_device_token(device: int) -> str | None:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible_devices:
        return None
    tokens = [token.strip() for token in visible_devices.split(",") if token.strip()]
    if device >= len(tokens):
        return None
    return tokens[device]


def get_socket_path(device: int, tag: str = "weights") -> str:
    """Get GMS socket path for the given CUDA device and tag.

    The socket path is based on GPU UUID. ``device`` is a CUDA-visible device
    index, so CUDA_VISIBLE_DEVICES=1 with device=0 resolves to physical GPU1.

    Args:
        device: CUDA-visible device index.

    Returns:
        Socket path
        (e.g., "<tempdir>/gms_GPU-12345678-1234-1234-1234-123456789abc_weights.sock").
    """
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    cache_key = (device, visible_devices)
    uuid = _uuid_cache.get(cache_key)
    if uuid is None:
        import pynvml  # deferred: not available in all environments

        token = _visible_device_token(device)
        pynvml.nvmlInit()
        try:
            if token and token.startswith("GPU-"):
                uuid = token
            else:
                nvml_index = int(token) if token and token.isdigit() else device
                handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_index)
                uuid = pynvml.nvmlDeviceGetUUID(handle)
        finally:
            pynvml.nvmlShutdown()
        _uuid_cache[cache_key] = uuid
    socket_dir = os.environ.get("GMS_SOCKET_DIR") or tempfile.gettempdir()
    return os.path.join(socket_dir, f"gms_{uuid}_{tag}.sock")
