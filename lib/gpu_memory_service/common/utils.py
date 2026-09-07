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
ENV_SCRATCH_KV_ENABLED = "DYN_GMS_SCRATCH_KV_ENABLED"
ENV_VMM_GRANULARITY = "DYN_GMS_VMM_GRANULARITY"

# Production GMS tags: the per-GPU server child and every engine integration
# serve exactly these logical memory pools, one UDS socket per (device, tag).
GMS_TAGS = ("weights", "kv_cache")

_TRUTHY = ("true", "1", "yes", "on")
_FALSEY_ENV_VALUES = frozenset({"", "0", "false", "no", "off"})


def is_truthy_env(name: str) -> bool:
    """True when the named env var is set to a recognized truthy string.

    Use this for opt-in flags: anything unrecognized reads as off.
    """
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def env_enabled_by_default(name: str, *, default: bool = True) -> bool:
    """True unless the named env var explicitly disables the feature.

    Use this for opt-out flags: only a recognized falsey value turns the
    feature off. This is the single implementation shared by the engine
    integrations — do not add per-module copies.
    """
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in _FALSEY_ENV_VALUES


def is_scratch_kv_enabled() -> bool:
    """True when this engine should use two-phase (scratch → real) KV allocation."""
    return is_truthy_env(ENV_SCRATCH_KV_ENABLED)


def fail(message: str, *args, exc_info=None) -> NoReturn:
    logger.critical(message, *args, exc_info=exc_info)
    logging.shutdown()
    os._exit(1)


_uuid_cache: dict[tuple[int, str], str] = {}


def invalidate_uuid_cache() -> None:
    """Clear cached GPU UUIDs. Call after CRIU restore when GPU assignment may change."""
    _uuid_cache.clear()


def nvml_handle_for_cuda_device(pynvml, device: int):
    """Return the NVML handle for a process-visible CUDA ordinal.

    NVML indices are physical-device indices and, unlike CUDA ordinals, are
    not reordered by ``CUDA_VISIBLE_DEVICES``. GMS APIs take CUDA ordinals,
    so translate the numeric or UUID token before using NVML. This helper
    intentionally does not inspect ``NVIDIA_VISIBLE_DEVICES``: container
    runtimes apply that filter to NVML's own device namespace already.
    """
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return pynvml.nvmlDeviceGetHandleByIndex(device)

    visible = [token.strip() for token in raw.split(",") if token.strip()]
    if device < 0 or device >= len(visible):
        raise IndexError(
            f"CUDA device {device} is not visible in CUDA_VISIBLE_DEVICES={raw}"
        )
    token = visible[device]
    try:
        return pynvml.nvmlDeviceGetHandleByIndex(int(token))
    except ValueError:
        return pynvml.nvmlDeviceGetHandleByUUID(token)


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
    visibility = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    cache_key = (device, visibility)
    uuid = _uuid_cache.get(cache_key)
    if uuid is None:
        import pynvml  # deferred: not available in all environments

        pynvml.nvmlInit()
        try:
            handle = nvml_handle_for_cuda_device(pynvml, device)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
        finally:
            pynvml.nvmlShutdown()
        _uuid_cache[cache_key] = uuid
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
