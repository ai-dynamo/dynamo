# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for GPU Memory Service."""

import os
import tempfile

import pynvml

from gpu_memory_service.common.types import RequestedLockType


def get_socket_path(device: int) -> str:
    """Get GMS socket path for the given CUDA device.

    The socket path is based on GPU UUID, making it stable across different
    CUDA_VISIBLE_DEVICES configurations.

    Args:
        device: CUDA device index.

    Returns:
        Socket path (e.g., "<tempdir>/gms_GPU-12345678-1234-1234-1234-123456789abc.sock").
    """
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
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
