# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""V1-local GPU identity and socket naming.

GMS device N is the container's N-th GPU in PCI bus order. N names the
device's checkpoint artifact; its sockets are named by the GPU's UUID, so an
engine finds them from its own GPU whatever order CUDA numbers GPUs in.
"""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from uuid import UUID

from gpu_memory_service.common.vmm.cuda_utils import cuda_ensure_initialized

try:
    from cuda.bindings import driver as cuda
except ImportError:
    cuda = None

_AF_UNIX_PATH_LIMIT = 104 if sys.platform == "darwin" else 108
_PROC_GPUS = Path("/proc/driver/nvidia/gpus")
_DEV = Path("/dev")


@dataclass(frozen=True)
class Gpu:
    bus_id: str
    minor: int
    uuid: str


def _can_open(node: Path) -> bool:
    # Opening, not os.access, so the device cgroup and LSMs decide too.
    try:
        os.close(os.open(node, os.O_RDWR | os.O_CLOEXEC))
    except OSError:
        return False
    return True


def node_gpus(proc_gpus: Path = _PROC_GPUS, dev: Path = _DEV) -> list[Gpu]:
    """GPUs this container can open, in PCI bus order, without starting CUDA."""
    gpus = []
    for entry in proc_gpus.iterdir():
        info = dict(
            line.split(":", 1)
            for line in (entry / "information").read_text().splitlines()
            if ":" in line
        )
        minor = int(info["Device Minor"])
        if _can_open(dev / f"nvidia{minor}"):
            gpus.append(Gpu(entry.name.lower(), minor, info["GPU UUID"].strip()))
    return sorted(gpus, key=lambda gpu: gpu.bus_id)


def gpu_uuid(device: int) -> str:
    """The UUID of GMS device ``device``.

    Lists the container's GPUs, so a process confined to one GPU (see
    ``landlock``) must be given its GPU's UUID instead.
    """
    gpus = node_gpus()
    if not 0 <= device < len(gpus):
        raise ValueError(f"GMS device {device} is not one of {len(gpus)} GPUs")
    return gpus[device].uuid


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


def cuda_ordinal(gpu_uuid: str) -> int:
    """This process's CUDA ordinal of the GPU with ``gpu_uuid``."""
    if cuda is None:
        raise RuntimeError(
            "cuda-python is required for GPU Memory Service device identity"
        )

    cuda_ensure_initialized()
    result, count = cuda.cuDeviceGetCount()
    _check_cuda(result, "cuDeviceGetCount")
    for ordinal in range(count):
        if get_device_uuid(ordinal) == gpu_uuid:
            return ordinal
    raise RuntimeError(f"GPU {gpu_uuid} is not visible to CUDA")


def invalidate_device_uuid_cache() -> None:
    """Clear cached device UUIDs after the visible GPU assignment changes."""
    get_device_uuid.cache_clear()


def get_socket_path(gpu_uuid: str, tag: str = "weights") -> str:
    """Return the V1 socket path for a GPU and domain."""
    socket_dir = os.environ.get("GMS_SOCKET_DIR") or tempfile.gettempdir()
    path = os.path.join(
        socket_dir,
        f"gms_{gpu_uuid}_{tag}.sock",
    )
    path_bytes = len(os.fsencode(path))
    if path_bytes >= _AF_UNIX_PATH_LIMIT:
        raise ValueError(
            "GMS socket path is too long for AF_UNIX "
            f"({path_bytes} bytes, limit {_AF_UNIX_PATH_LIMIT - 1}): {path}"
        )
    return path
