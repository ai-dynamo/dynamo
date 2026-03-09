# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for GPU Memory Service."""

import os
import tempfile

import pynvml

from gpu_memory_service.common.types import RequestedLockType


def _resolve_physical_device(device: int) -> int:
    """Map a CUDA runtime device index to its physical (NVML) device index.

    NVML always enumerates physical GPUs regardless of CUDA_VISIBLE_DEVICES.
    When CUDA_VISIBLE_DEVICES is set, CUDA device 0 may correspond to a
    different NVML device index. This function resolves the mapping by
    comparing UUIDs.

    If CUDA_VISIBLE_DEVICES is not set, the device index is returned as-is.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is None or cvd == "":
        return device

    # Parse CUDA_VISIBLE_DEVICES (supports comma-separated indices or UUIDs)
    entries = [e.strip() for e in cvd.split(",") if e.strip()]
    if device >= len(entries):
        return device

    entry = entries[device]
    try:
        return int(entry)
    except ValueError:
        # Could be a UUID — fall back to identity mapping
        return device


def get_socket_path(device: int) -> str:
    """Get GMS socket path for the given CUDA device.

    The socket path is based on GPU UUID, making it stable across different
    CUDA_VISIBLE_DEVICES configurations. When CUDA_VISIBLE_DEVICES is set,
    the device index is remapped to the physical GPU index before looking
    up the UUID.

    Args:
        device: CUDA device index (runtime-visible).

    Returns:
        Socket path (e.g., "<tempdir>/gms_GPU-12345678-1234-1234-1234-123456789abc.sock").
    """
    physical_device = _resolve_physical_device(device)
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device)
        uuid = pynvml.nvmlDeviceGetUUID(handle)
    finally:
        pynvml.nvmlShutdown()
    return os.path.join(tempfile.gettempdir(), f"gms_{uuid}.sock")


def get_weight_lock_type() -> RequestedLockType:
    """Determine weight GMS lock type from ENGINE_ID.

    ENGINE_ID=0 (default): RW_OR_RO — can load weights from disk or import
        from an existing writer. Backward compatible with single-engine
        deployments where ENGINE_ID is not set.
    ENGINE_ID=1+: RO — import only, blocks until weights are committed.
        Prevents TP>1 deadlocks by ensuring only engine-0 ever holds RW
        locks across devices.
    """
    engine_id = int(os.environ.get("ENGINE_ID", "0"))
    if engine_id == 0:
        return RequestedLockType.RW_OR_RO
    return RequestedLockType.RO
