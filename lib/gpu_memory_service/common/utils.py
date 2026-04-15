# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for GPU Memory Service."""

import logging
import os
import tempfile
from typing import NoReturn

logger = logging.getLogger(__name__)


def fail(message: str, *args, exc_info=None) -> NoReturn:
    logger.critical(message, *args, exc_info=exc_info)
    logging.shutdown()
    os._exit(1)


_uuid_cache: dict[int, str] = {}


def invalidate_uuid_cache() -> None:
    """Clear cached GPU UUIDs. Call after CRIU restore when GPU assignment may change."""
    _uuid_cache.clear()


def get_socket_path(device: int, tag: str = "weights") -> str:
    """Get GMS socket path for the given CUDA device and tag.

    The socket path is based on GPU UUID, making it stable across different
    CUDA_VISIBLE_DEVICES configurations. UUIDs are cached per device index.

    Args:
        device: CUDA device index.

    Returns:
        Socket path
        (e.g., "<tempdir>/gms_GPU-12345678-1234-1234-1234-123456789abc_weights.sock").
    """
    uuid = _uuid_cache.get(device)
    if uuid is None:
        import pynvml  # deferred: not available in all environments

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
        finally:
            pynvml.nvmlShutdown()
        _uuid_cache[device] = uuid
    socket_dir = os.environ.get("GMS_SOCKET_DIR") or tempfile.gettempdir()
    return os.path.join(socket_dir, f"gms_{uuid}_{tag}.sock")
