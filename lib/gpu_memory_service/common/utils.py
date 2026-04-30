# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for GPU Memory Service."""

import logging
import os
import tempfile
from typing import NoReturn

logger = logging.getLogger(__name__)

GMS_GPU_UUIDS_ENV = "GMS_GPU_UUIDS"
GMS_GPU_UUID_ENV_PREFIX = "GMS_GPU_UUID_"


# Canonical names for GMS-related environment variables. Defined here so
# operator code, launcher code, and engine integration code all reference
# one source of truth — keeping these in lockstep with the Go-side
# constants in deploy/operator/internal/gms/gms.go.
ENV_SCRATCH_KV_ENABLED = "DYN_GMS_SCRATCH_KV_ENABLED"
ENV_VMM_GRANULARITY = "DYN_GMS_VMM_GRANULARITY"

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


_uuid_cache: dict[int, str] = {}


def invalidate_uuid_cache() -> None:
    """Clear cached GPU UUIDs. Call after CRIU restore when GPU assignment may change."""
    _uuid_cache.clear()


def _split_visible_device_uuids(value: str) -> list[str]:
    return [
        part.strip()
        for part in value.replace(";", ",").split(",")
        if part.strip().startswith(("GPU-", "MIG-"))
    ]


def _get_uuid_from_env(device: int) -> str | None:
    direct = os.environ.get(f"{GMS_GPU_UUID_ENV_PREFIX}{device}")
    if direct:
        direct = direct.strip()
        if direct:
            return direct

    ordered = os.environ.get(GMS_GPU_UUIDS_ENV)
    if ordered:
        uuids = [
            part.strip()
            for part in ordered.replace(";", ",").split(",")
            if part.strip()
        ]
        if device < len(uuids):
            return uuids[device]

    # Some runtimes already expose UUIDs in visible-device order. Treat those
    # as a free fast path, but prefer the explicit GMS_* env vars above.
    for name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        uuids = _split_visible_device_uuids(os.environ.get(name, ""))
        if device < len(uuids):
            return uuids[device]
    return None


def get_socket_path(device: int, tag: str = "weights") -> str:
    """Get GMS socket path for the given CUDA device and tag.

    The socket path is based on GPU UUID, making it stable across different
    CUDA_VISIBLE_DEVICES configurations. UUIDs are cached per device index.
    When the runtime provides GPU UUIDs through GMS_GPU_UUIDS or
    GMS_GPU_UUID_<index>, those env vars are used instead of querying NVML.

    Args:
        device: CUDA device index.

    Returns:
        Socket path
        (e.g., "<tempdir>/gms_GPU-12345678-1234-1234-1234-123456789abc_weights.sock").
    """
    uuid = _uuid_cache.get(device)
    if uuid is None:
        uuid = _get_uuid_from_env(device)
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
