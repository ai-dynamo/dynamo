# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA VMM allocation manager - pure business logic, no threading/transport.

This module contains the GMSServerMemoryManager class which handles physical GPU memory
allocations via CUDA Virtual Memory Management (VMM) API. It creates shareable
memory without mapping it locally (no CUDA context needed on the server).

The GMSServerMemoryManager is NOT thread-safe. Callers must provide external
synchronization (e.g., via LockManager ensuring single-writer access).
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
from uuid import uuid4

from cuda.bindings import driver as cuda
from gpu_memory_service.common.cuda_vmm_utils import (
    align_to_granularity,
    check_cuda_result,
    ensure_cuda_initialized,
    get_allocation_granularity,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AllocationInfo:
    """Information about a single GPU memory allocation.

    Attributes:
        allocation_id: Unique identifier for this allocation
        size: Requested size in bytes
        aligned_size: Actual size after alignment to VMM granularity
        handle: CUmemGenericAllocationHandle value
        tag: User-provided tag for grouping allocations
        epoch_id: Epoch that owns this allocation
        created_at: Timestamp when allocation was created
    """

    allocation_id: str
    size: int
    aligned_size: int
    handle: int
    tag: str
    epoch_id: str
    created_at: float


class AllocationNotFoundError(Exception):
    """Raised when an allocation_id doesn't exist."""

    pass


class GMSServerMemoryManager:
    """GPU Memory Service server-side memory manager.

    Manages CUDA VMM physical memory allocations. This class handles the core
    memory operations using CUDA Virtual Memory Management (VMM) API. It creates
    physical allocations that can be exported as POSIX file descriptors for
    sharing with other processes.

    Key design points:
    - No VA mapping: The memory manager never maps memory to virtual addresses,
      so it doesn't create a CUDA context. This allows it to survive GPU
      driver failures.
    - NOT thread-safe: Callers must provide external synchronization.
      The GMSLocalFSM's RW/RO semantics ensure single-writer access.
    """

    def __init__(self, device: int = 0):
        self._device = device
        self._allocations: Dict[str, AllocationInfo] = {}
        ensure_cuda_initialized()
        self._granularity = get_allocation_granularity(device)
        logger.info(
            f"GMSServerMemoryManager initialized: device={device}, granularity={self._granularity}"
        )

    @property
    def device(self) -> int:
        return self._device

    @property
    def granularity(self) -> int:
        return self._granularity

    @property
    def allocation_count(self) -> int:
        return len(self._allocations)

    @property
    def total_bytes(self) -> int:
        return sum(info.aligned_size for info in self._allocations.values())

    def _get(
        self, allocation_id: str, epoch_id: Optional[str] = None
    ) -> AllocationInfo:
        info = self._allocations.get(allocation_id)
        if info is None:
            raise AllocationNotFoundError(f"Unknown allocation: {allocation_id}")
        if epoch_id is not None and info.epoch_id != epoch_id:
            raise AllocationNotFoundError(
                f"Allocation {allocation_id} is not in epoch {epoch_id}"
            )
        return info

    def _release(self, info: AllocationInfo) -> None:
        (result,) = cuda.cuMemRelease(info.handle)
        if result != cuda.CUresult.CUDA_SUCCESS:
            logger.warning(f"cuMemRelease failed for {info.allocation_id}: {result}")

    def allocate(
        self, size: int, tag: str = "default", epoch_id: str = ""
    ) -> AllocationInfo:
        """Create a physical memory allocation (no VA mapping).

        Uses cuMemCreate to allocate physical GPU memory that can be exported
        as a file descriptor for sharing with other processes.

        Args:
            size: Requested size in bytes (will be aligned up to granularity)
            tag: Tag for grouping allocations (e.g., "weights", "kv_cache")
            epoch_id: Epoch that owns this allocation

        Returns:
            AllocationInfo with allocation_id, aligned_size, handle

        Raises:
            RuntimeError: If CUDA allocation fails
        """
        if not epoch_id:
            raise ValueError("epoch_id must be non-empty")

        aligned_size = align_to_granularity(size, self._granularity)

        prop = cuda.CUmemAllocationProp()
        prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = self._device
        prop.requestedHandleTypes = (
            cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        )

        result, handle = cuda.cuMemCreate(aligned_size, prop, 0)
        check_cuda_result(result, "cuMemCreate")

        # epoch_id is immutable: assigned at allocation creation and never changes.
        info = AllocationInfo(
            allocation_id=str(uuid4()),
            size=size,
            aligned_size=aligned_size,
            handle=int(handle),
            tag=tag,
            epoch_id=epoch_id,
            created_at=time.time(),
        )
        self._allocations[info.allocation_id] = info
        logger.debug(
            f"Allocated {info.allocation_id}: size={size}, aligned={aligned_size}, tag={tag}, epoch={epoch_id}"
        )
        return info

    def export_fd(self, allocation_id: str, epoch_id: Optional[str] = None) -> int:
        """Export allocation as POSIX FD for SCM_RIGHTS transfer.

        The returned file descriptor can be sent to another process via
        Unix domain socket SCM_RIGHTS. The receiving process can then
        import it using cuMemImportFromShareableHandle.

        IMPORTANT: The caller MUST close the returned FD after sendmsg()
        to avoid file descriptor leaks.

        Args:
            allocation_id: ID of allocation to export

        Returns:
            File descriptor (caller owns, must close after sending)

        Raises:
            AllocationNotFoundError: If allocation_id doesn't exist
            RuntimeError: If CUDA export fails
        """
        info = self._get(allocation_id, epoch_id=epoch_id)
        result, fd = cuda.cuMemExportToShareableHandle(
            info.handle,
            cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
            0,
        )
        check_cuda_result(result, "cuMemExportToShareableHandle")
        return int(fd)

    def free(self, allocation_id: str, epoch_id: Optional[str] = None) -> bool:
        """Release physical memory for a single allocation.

        Args:
            allocation_id: ID of allocation to free

        Returns:
            True if allocation existed and was freed, False otherwise
        """
        info = self._allocations.get(allocation_id)
        if info is None:
            return False
        if epoch_id is not None and info.epoch_id != epoch_id:
            logger.debug(
                "Free skipped due to epoch mismatch: allocation_id=%s allocation_epoch=%s requested_epoch=%s",
                allocation_id,
                info.epoch_id,
                epoch_id,
            )
            return False
        self._allocations.pop(allocation_id, None)
        self._release(info)
        logger.debug(f"Freed allocation: {allocation_id}")
        return True

    def clear_all(self) -> int:
        """Release ALL allocations.

        Used by loaders before loading a new model, or during cleanup
        when a writer aborts without committing.

        Returns:
            Number of allocations cleared
        """
        count = len(self._allocations)
        for info in self._allocations.values():
            self._release(info)
        self._allocations.clear()
        logger.info(f"Cleared {count} allocations")
        return count

    def clear_epoch(self, epoch_id: str) -> int:
        """Release all allocations in a specific epoch."""
        to_clear = [
            allocation_id
            for allocation_id, info in self._allocations.items()
            if info.epoch_id == epoch_id
        ]
        for allocation_id in to_clear:
            info = self._allocations.pop(allocation_id)
            self._release(info)
        if to_clear:
            logger.info(f"Cleared {len(to_clear)} allocations from epoch {epoch_id}")
        return len(to_clear)

    def get_allocation(
        self, allocation_id: str, epoch_id: Optional[str] = None
    ) -> AllocationInfo:
        """Get allocation info. Raises AllocationNotFoundError if not found."""
        return self._get(allocation_id, epoch_id=epoch_id)

    def list_allocations(
        self, tag: Optional[str] = None, epoch_id: Optional[str] = None
    ) -> List[AllocationInfo]:
        """List all allocations, optionally filtered by tag."""
        allocations = self._allocations.values()
        if epoch_id is not None:
            allocations = [info for info in allocations if info.epoch_id == epoch_id]
        if tag is None:
            return list(allocations)
        return [info for info in allocations if info.tag == tag]
