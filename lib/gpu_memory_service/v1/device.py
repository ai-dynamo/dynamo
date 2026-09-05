# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""V1-local CUDA device identity and socket naming."""

from __future__ import annotations

import os
import tempfile
from functools import cache
from uuid import UUID

from gpu_memory_service.common.utils import ensure_af_unix_path_fits
from gpu_memory_service.common.vmm.cuda_utils import cuda_ensure_initialized

try:
    from cuda.bindings import driver as cuda
except ImportError:
    cuda = None


def _check_cuda(result, operation: str) -> None:
    if cuda is None:
        raise RuntimeError(
            "cuda-python is required for GPU Memory Service device identity"
        )
    if result == cuda.CUresult.CUDA_SUCCESS:
        return

    error_result, error_string = cuda.cuGetErrorString(result)
    if error_result == cuda.CUresult.CUDA_SUCCESS and error_string:
        detail = (
            error_string.decode()
            if isinstance(error_string, bytes)
            else str(error_string)
        )
    else:
        detail = f"{result} (cuGetErrorString failed: {error_result})"
    raise RuntimeError(f"CUDA driver call {operation} failed: {detail}")


@cache
def get_device_uuid(device: int) -> str:
    """Return the UUID of a CUDA-visible device ordinal."""
    if cuda is None:
        raise RuntimeError(
            "cuda-python is required for GPU Memory Service device identity"
        )

    cuda_ensure_initialized()
    result, cuda_device = cuda.cuDeviceGet(device)
    _check_cuda(result, "cuDeviceGet")
    result, uuid = cuda.cuDeviceGetUuid(cuda_device)
    _check_cuda(result, "cuDeviceGetUuid")
    return f"GPU-{UUID(bytes=bytes(uuid.bytes))}"


def invalidate_device_uuid_cache() -> None:
    """Clear cached device UUIDs after the visible GPU assignment changes."""
    get_device_uuid.cache_clear()


def get_socket_path(device: int, tag: str = "weights") -> str:
    """Return the V1 socket path for a CUDA-visible device and domain."""
    socket_dir = os.environ.get("GMS_SOCKET_DIR") or tempfile.gettempdir()
    path = os.path.join(
        socket_dir,
        f"gms_{device}_{tag}.sock",
    )
    ensure_af_unix_path_fits(path)
    return path
