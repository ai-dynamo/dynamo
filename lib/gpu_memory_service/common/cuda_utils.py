# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA driver helpers shared by the GMS client and server."""

from __future__ import annotations

import os

from gpu_memory_service.common.locks import GrantedLockType
from gpu_memory_service.common.utils import fail

try:
    from cuda.bindings import driver as cuda
except ImportError:
    # Keep import-time collection working in CPU-only environments and let the
    # first real CUDA call fail with a targeted message instead.
    class _MissingCuda:
        def __getattr__(self, name):
            raise RuntimeError(
                "cuda-python is required for GPU Memory Service CUDA operations"
            )

    cuda = _MissingCuda()


def list_devices() -> list[int]:
    """Return list of CUDA device indices visible to this process via NVML."""
    import pynvml

    pynvml.nvmlInit()
    try:
        count = pynvml.nvmlDeviceGetCount()
    finally:
        pynvml.nvmlShutdown()
    if count == 0:
        raise SystemExit("no nvidia devices found")
    return list(range(count))


def cuda_check_result(result: cuda.CUresult, name: str) -> None:
    if result != cuda.CUresult.CUDA_SUCCESS:
        err_result, err_str = cuda.cuGetErrorString(result)
        if err_result == cuda.CUresult.CUDA_SUCCESS and err_str:
            err_msg = err_str.decode() if isinstance(err_str, bytes) else str(err_str)
        else:
            err_msg = str(result)
        fail("fatal CUDA VMM error in %s: %s", name, err_msg)


def cuda_ensure_initialized() -> None:
    (result,) = cuda.cuInit(0)
    cuda_check_result(result, "cuInit")


def cumem_get_allocation_granularity(device: int) -> int:
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
    cuda_check_result(result, "cuMemGetAllocationGranularity")
    return int(granularity)


def cumem_create_tolerate_oom(size: int, device: int) -> tuple[bool, int]:
    prop = cuda.CUmemAllocationProp()
    prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device
    prop.requestedHandleTypes = (
        cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    )

    result, handle = cuda.cuMemCreate(size, prop, 0)
    if result == cuda.CUresult.CUDA_SUCCESS:
        return True, int(handle)
    if result == cuda.CUresult.CUDA_ERROR_OUT_OF_MEMORY:
        return False, 0
    cuda_check_result(result, "cuMemCreate")
    return False, 0


def cumem_export_to_shareable_handle(handle: int) -> int:
    result, fd = cuda.cuMemExportToShareableHandle(
        handle,
        cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
        0,
    )
    cuda_check_result(result, "cuMemExportToShareableHandle")
    return int(fd)


def align_to_granularity(size: int, granularity: int) -> int:
    """Align size up to VMM granularity.

    Args:
        size: Size in bytes
        granularity: Allocation granularity

    Returns:
        Aligned size
    """
    return ((size + granularity - 1) // granularity) * granularity


def cumem_import_from_shareable_handle_close_fd(fd: int) -> int:
    try:
        result, handle = cuda.cuMemImportFromShareableHandle(
            fd,
            cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
        )
        cuda_check_result(result, "cuMemImportFromShareableHandle")
        return int(handle)
    finally:
        os.close(fd)


def cumem_address_reserve(size: int, granularity: int) -> int:
    result, va = cuda.cuMemAddressReserve(size, granularity, 0, 0)
    cuda_check_result(result, "cuMemAddressReserve")
    return int(va)


def cumem_address_free(va: int, size: int) -> None:
    (result,) = cuda.cuMemAddressFree(va, size)
    cuda_check_result(result, "cuMemAddressFree")


def cumem_map(va: int, size: int, handle: int) -> None:
    (result,) = cuda.cuMemMap(va, size, 0, handle, 0)
    cuda_check_result(result, "cuMemMap")


def cumem_set_access(va: int, size: int, device: int, access: GrantedLockType) -> None:
    access_desc = cuda.CUmemAccessDesc()
    access_desc.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    access_desc.location.id = device
    access_desc.flags = (
        cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READ
        if access == GrantedLockType.RO
        else cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    )
    (result,) = cuda.cuMemSetAccess(va, size, [access_desc], 1)
    cuda_check_result(result, "cuMemSetAccess")


def cumem_unmap(va: int, size: int) -> None:
    (result,) = cuda.cuMemUnmap(va, size)
    cuda_check_result(result, "cuMemUnmap")


def cumem_release(handle: int) -> None:
    (result,) = cuda.cuMemRelease(handle)
    cuda_check_result(result, "cuMemRelease")


def cuda_validate_pointer(va: int) -> None:
    result, _ = cuda.cuPointerGetAttribute(
        cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_DEVICE_POINTER, va
    )
    cuda_check_result(result, "cuPointerGetAttribute")


def cuda_synchronize() -> None:
    (result,) = cuda.cuCtxSynchronize()
    cuda_check_result(result, "cuCtxSynchronize")
