# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""V1-local CUDA device identity and socket naming."""

from __future__ import annotations

import os
import sys
import tempfile
from functools import cache
from uuid import UUID

try:
    from cuda.bindings import driver as cuda
except ImportError:
    cuda = None

_AF_UNIX_PATH_LIMIT = 104 if sys.platform == "darwin" else 108


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

    (result,) = cuda.cuInit(0)
    _check_cuda(result, "cuInit")
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
    path_bytes = len(os.fsencode(path))
    if path_bytes >= _AF_UNIX_PATH_LIMIT:
        raise ValueError(
            "GMS socket path is too long for AF_UNIX "
            f"({path_bytes} bytes, limit {_AF_UNIX_PATH_LIMIT - 1}): {path}"
        )
    return path
