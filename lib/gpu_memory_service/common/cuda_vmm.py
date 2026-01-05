# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared CUDA Virtual Memory Management (VMM) constants and types.

This module provides the CUDA driver API constants and ctypes structures used
by both server (GMSServerMemoryManager) and client (GMSClientMemoryManager) for VMM operations.
"""

import ctypes
from ctypes import Structure, byref, c_int, c_size_t, c_ulonglong, c_void_p

# CUDA VMM constants
CU_MEM_ALLOCATION_TYPE_PINNED = 0x1
CU_MEM_LOCATION_TYPE_DEVICE = 0x1
CU_MEM_ACCESS_FLAGS_PROT_READ = 0x1
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 0x3
CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0x0
CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 0x1

# CUDA types - exported for use by client/server
CUdeviceptr = c_ulonglong
CUmemGenericAllocationHandle = c_ulonglong


class CUmemLocation(Structure):
    """CUDA memory location descriptor."""

    _fields_ = [("type", c_int), ("id", c_int)]


class CUmemAccessDesc(Structure):
    """CUDA memory access descriptor."""

    _fields_ = [("location", CUmemLocation), ("flags", c_int)]


class CUmemAllocationProp(Structure):
    """CUDA memory allocation properties."""

    _fields_ = [
        ("type", c_int),
        ("requestedHandleTypes", c_int),
        ("location", CUmemLocation),
        ("win32HandleMetaData", c_void_p),
        ("allocFlags_compressionType", ctypes.c_uint),
        ("allocFlags_gpuDirectRDMACapable", ctypes.c_uint),
        ("allocFlags_usage", ctypes.c_uint),
        ("allocFlags_reserved", ctypes.c_uint * 4),
    ]


# Lazy-loaded CUDA library handle
_cuda = None


def get_cuda_driver():
    """Get the CUDA driver library handle (lazy-loaded).

    Returns:
        ctypes.CDLL: Handle to libcuda.so.1
    """
    global _cuda
    if _cuda is None:
        _cuda = ctypes.CDLL("libcuda.so.1")
    return _cuda


def check_cuda_result(result: int, name: str) -> None:
    """Check CUDA driver API result and raise on error.

    Args:
        result: CUDA driver API return code
        name: Operation name for error message

    Raises:
        RuntimeError: If result is not CUDA_SUCCESS (0)
    """
    if result != 0:
        cuda = get_cuda_driver()
        err = ctypes.c_char_p()
        cuda.cuGetErrorString(result, byref(err))
        raise RuntimeError(f"{name}: {err.value.decode() if err.value else result}")


def ensure_cuda_initialized() -> None:
    """Ensure CUDA driver is initialized.

    Raises:
        RuntimeError: If cuInit fails
    """
    cuda = get_cuda_driver()
    result = cuda.cuInit(0)
    if result != 0:
        check_cuda_result(result, "cuInit")


def get_allocation_granularity(device: int) -> int:
    """Get VMM allocation granularity for a device.

    Args:
        device: CUDA device index

    Returns:
        Allocation granularity in bytes (typically 2 MiB)
    """
    cuda = get_cuda_driver()
    prop = CUmemAllocationProp()
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    gran = c_size_t()
    check_cuda_result(
        cuda.cuMemGetAllocationGranularity(byref(gran), byref(prop), 0),
        "cuMemGetAllocationGranularity",
    )
    return int(gran.value)


def align_to_granularity(size: int, granularity: int) -> int:
    """Align size up to VMM granularity.

    Args:
        size: Size in bytes
        granularity: Allocation granularity

    Returns:
        Aligned size
    """
    return ((size + granularity - 1) // granularity) * granularity
