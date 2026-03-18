# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA Virtual Memory Management (VMM) utility functions.

This module provides utility functions for CUDA driver API operations
used by both server (GMSServerMemoryManager) and client (GMSClientMemoryManager).
"""

from cuda.bindings import driver as cuda


def check_cuda_result(result: cuda.CUresult, name: str) -> None:
    """Check CUDA driver API result and raise on error.

    Args:
        result: CUDA driver API return code (CUresult enum)
        name: Operation name for error message

    Raises:
        RuntimeError: If result is not CUDA_SUCCESS
    """
    if result != cuda.CUresult.CUDA_SUCCESS:
        err_result, err_str = cuda.cuGetErrorString(result)
        if err_result == cuda.CUresult.CUDA_SUCCESS and err_str:
            err_msg = err_str.decode() if isinstance(err_str, bytes) else str(err_str)
        else:
            err_msg = str(result)
        raise RuntimeError(f"{name}: {err_msg}")


def ensure_cuda_initialized() -> None:
    """Ensure CUDA driver is initialized.

    Raises:
        RuntimeError: If cuInit fails
    """
    (result,) = cuda.cuInit(0)
    check_cuda_result(result, "cuInit")


def get_allocation_granularity(device: int) -> int:
    """Get VMM allocation granularity for a device.

    Args:
        device: CUDA device index

    Returns:
        Allocation granularity in bytes (typically 2 MiB)
    """
    prop = cuda.CUmemAllocationProp()
    prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device
    prop.requestedHandleTypes = (
        cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    )

    result, granularity = cuda.cuMemGetAllocationGranularity(
        prop, cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
    )
    check_cuda_result(result, "cuMemGetAllocationGranularity")
    return int(granularity)


def align_to_granularity(size: int, granularity: int) -> int:
    """Align size up to VMM granularity.

    Args:
        size: Size in bytes
        granularity: Allocation granularity

    Returns:
        Aligned size
    """
    return ((size + granularity - 1) // granularity) * granularity
