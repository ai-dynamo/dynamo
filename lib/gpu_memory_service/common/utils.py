# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for GPU Memory Service."""

import logging
import os
import sys
import tempfile
from typing import NoReturn

logger = logging.getLogger(__name__)


# Canonical names for GMS-related environment variables. Defined here so
# operator code, launcher code, and engine integration code all reference
# one source of truth — keeping these in lockstep with the Go-side
# constants in deploy/operator/internal/gms/gms.go.
ENV_SCRATCH_KV_ENABLED = "DYN_GMS_SCRATCH_KV_ENABLED"
ENV_VMM_GRANULARITY = "DYN_GMS_VMM_GRANULARITY"

# Production GMS tags: the per-GPU server child and every engine integration
# serve exactly these logical memory pools, one UDS socket per (device, tag).
GMS_TAGS = ("weights", "kv_cache")

_TRUTHY = ("true", "1", "yes")

# sun_path is 104 bytes on Darwin and 108 on Linux, including the terminator.
# Shared by both socket-path builders so the two cannot disagree about it.
AF_UNIX_PATH_LIMIT = 104 if sys.platform == "darwin" else 108


def ensure_af_unix_path_fits(path: str) -> str:
    """Return path, or raise ValueError if it cannot be bound as an AF_UNIX socket.

    bind() reports an over-long path as a bare ``OSError: AF_UNIX path too long``
    from wherever the socket is created, which does not say which path or what
    the limit was. Callers build these from ``GMS_SOCKET_DIR``, so the value is
    operator-supplied and worth naming.
    """
    path_bytes = len(os.fsencode(path))
    if path_bytes >= AF_UNIX_PATH_LIMIT:
        raise ValueError(
            "GMS socket path is too long for AF_UNIX "
            f"({path_bytes} bytes, limit {AF_UNIX_PATH_LIMIT - 1}): {path}"
        )
    return path


def is_truthy_env(name: str) -> bool:
    """True when the named env var is set to a recognized truthy string."""
    return os.environ.get(name, "").lower() in _TRUTHY


def is_scratch_kv_enabled() -> bool:
    """True when this engine should use two-phase (scratch → real) KV allocation."""
    return is_truthy_env(ENV_SCRATCH_KV_ENABLED)


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
    return ensure_af_unix_path_fits(os.path.join(socket_dir, f"gms_{uuid}_{tag}.sock"))


def align_to_granularity(size: int, granularity: int) -> int:
    """Align size up to VMM granularity.

    Args:
        size: Size in bytes
        granularity: Allocation granularity

    Returns:
        Aligned size
    """
    return ((size + granularity - 1) // granularity) * granularity
