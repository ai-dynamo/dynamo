# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for GPU Memory Service."""

import logging
import os
import tempfile
from typing import NoReturn

from gpu_memory_service.core import device as device_identity

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


def invalidate_uuid_cache() -> None:
    """Clear cached GPU UUIDs. Call after CRIU restore when GPU assignment may change."""
    device_identity.invalidate_device_uuid_cache()


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
    uuid = device_identity.get_device_uuid(device)
    socket_dir = os.environ.get("GMS_SOCKET_DIR") or tempfile.gettempdir()
    return os.path.join(socket_dir, f"gms_{uuid}_{tag}.sock")


def align_to_granularity(size: int, granularity: int) -> int:
    """Align size up to VMM granularity.

    Args:
        size: Size in bytes
        granularity: Allocation granularity

    Returns:
        Aligned size
    """
    return ((size + granularity - 1) // granularity) * granularity
