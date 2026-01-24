# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client-side CUDA VMM utilities.

These functions wrap CUDA driver API calls used by the client memory manager
for importing, mapping, and unmapping GPU memory.
"""

from __future__ import annotations

from cuda.bindings import driver as cuda
from gpu_memory_service.common.cuda_vmm_utils import check_cuda_result
from gpu_memory_service.common.types import GrantedLockType


def import_handle_from_fd(fd: int) -> int:
    """Import a CUDA memory handle from a file descriptor.

    Args:
        fd: POSIX file descriptor received via SCM_RIGHTS.

    Returns:
        CUDA memory handle.
    """
    result, handle = cuda.cuMemImportFromShareableHandle(
        fd,
        cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
    )
    check_cuda_result(result, "cuMemImportFromShareableHandle")
    return int(handle)


def reserve_va(size: int, granularity: int) -> int:
    """Reserve virtual address space.

    Args:
        size: Size in bytes (should be aligned to granularity).
        granularity: VMM allocation granularity.

    Returns:
        Reserved virtual address.
    """
    result, va = cuda.cuMemAddressReserve(size, granularity, 0, 0)
    check_cuda_result(result, "cuMemAddressReserve")
    return int(va)


def free_va(va: int, size: int) -> None:
    """Free a virtual address reservation.

    Args:
        va: Virtual address to free.
        size: Size of the reservation.
    """
    (result,) = cuda.cuMemAddressFree(va, size)
    check_cuda_result(result, "cuMemAddressFree")


def map_to_va(va: int, size: int, handle: int) -> None:
    """Map a CUDA handle to a virtual address.

    Args:
        va: Virtual address (must be reserved).
        size: Size of the mapping.
        handle: CUDA memory handle.
    """
    (result,) = cuda.cuMemMap(va, size, 0, handle, 0)
    check_cuda_result(result, "cuMemMap")


def set_access(va: int, size: int, device: int, access: GrantedLockType) -> None:
    """Set access permissions for a mapped region.

    Args:
        va: Virtual address.
        size: Size of the region.
        device: CUDA device index.
        access: Access mode - RO for read-only, RW for read-write.
    """
    acc = cuda.CUmemAccessDesc()
    acc.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    acc.location.id = device
    acc.flags = (
        cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READ
        if access == GrantedLockType.RO
        else cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    )
    (result,) = cuda.cuMemSetAccess(va, size, [acc], 1)
    check_cuda_result(result, "cuMemSetAccess")


def unmap(va: int, size: int) -> None:
    """Unmap a virtual address region.

    Args:
        va: Virtual address to unmap.
        size: Size of the mapping.
    """
    (result,) = cuda.cuMemUnmap(va, size)
    check_cuda_result(result, "cuMemUnmap")


def release_handle(handle: int) -> None:
    """Release a CUDA memory handle.

    Args:
        handle: CUDA memory handle to release.
    """
    (result,) = cuda.cuMemRelease(handle)
    check_cuda_result(result, "cuMemRelease")
