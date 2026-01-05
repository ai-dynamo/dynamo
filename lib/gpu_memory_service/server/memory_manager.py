"""CUDA VMM allocation manager - pure business logic, no threading/transport.

This module contains the GMSServerMemoryManager class which handles physical GPU memory
allocations via CUDA Virtual Memory Management (VMM) API. It creates shareable
memory without mapping it locally (no CUDA context needed on the server).

The GMSServerMemoryManager is NOT thread-safe. Callers must provide external
synchronization (e.g., via LockManager ensuring single-writer access).
"""

import logging
import time
from ctypes import byref, c_int, c_size_t, c_ulonglong
from dataclasses import dataclass
from typing import Dict, List, Optional
from uuid import uuid4

from gpu_memory_service.common.cuda_vmm import (
    CU_MEM_ALLOCATION_TYPE_PINNED,
    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
    CU_MEM_LOCATION_TYPE_DEVICE,
    CUmemAllocationProp,
    CUmemGenericAllocationHandle,
    align_to_granularity,
    check_cuda_result,
    ensure_cuda_initialized,
    get_allocation_granularity,
    get_cuda_driver,
)

# Re-export CUmemGenericAllocationHandle for use in cuMemRelease calls
__all__ = ["GMSServerMemoryManager", "AllocationInfo", "AllocationNotFoundError"]

logger = logging.getLogger(__name__)


@dataclass
class AllocationInfo:
    """Information about a single GPU memory allocation.

    Attributes:
        allocation_id: Unique identifier for this allocation
        size: Requested size in bytes
        aligned_size: Actual size after alignment to VMM granularity
        handle: CUmemGenericAllocationHandle value
        tag: User-provided tag for grouping allocations
        created_at: Timestamp when allocation was created
    """

    allocation_id: str
    size: int
    aligned_size: int
    handle: int
    tag: str
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
    - No VA mapping: The manager never maps memory to virtual addresses,
      so it doesn't create a CUDA context. This allows it to survive GPU
      driver failures.
    - NOT thread-safe: Callers must provide external synchronization.
      The GlobalLockFSM's RW/RO semantics ensure single-writer access.
    - Lazy CUDA loading: CUDA library is loaded on first use.

    Example:
        manager = GMSServerMemoryManager(device=0)
        info = manager.allocate(1024 * 1024, tag="weights")
        fd = manager.export_fd(info.allocation_id)
        # Send fd to client via SCM_RIGHTS
        os.close(fd)  # Caller must close after sending
    """

    def __init__(self, device: int = 0):
        """Initialize allocation manager for a CUDA device.

        Args:
            device: CUDA device index (default: 0)

        Raises:
            RuntimeError: If CUDA initialization fails
        """
        self._device = device
        self._allocations: Dict[str, AllocationInfo] = {}

        # Initialize CUDA and get granularity
        ensure_cuda_initialized()
        self._granularity = get_allocation_granularity(device)

        logger.info(
            f"GMSServerMemoryManager initialized: device={device}, granularity={self._granularity}"
        )

    @property
    def device(self) -> int:
        """CUDA device index."""
        return self._device

    @property
    def granularity(self) -> int:
        """VMM allocation granularity for this device (typically 2 MiB)."""
        return self._granularity

    @property
    def allocation_count(self) -> int:
        """Number of active allocations."""
        return len(self._allocations)

    @property
    def total_bytes(self) -> int:
        """Total aligned bytes across all allocations."""
        return sum(info.aligned_size for info in self._allocations.values())

    def _align(self, size: int) -> int:
        """Align size up to VMM granularity."""
        return align_to_granularity(size, self._granularity)

    def allocate(self, size: int, tag: str = "default") -> AllocationInfo:
        """Create a physical memory allocation (no VA mapping).

        Uses cuMemCreate to allocate physical GPU memory that can be exported
        as a file descriptor for sharing with other processes.

        Args:
            size: Requested size in bytes (will be aligned up to granularity)
            tag: Tag for grouping allocations (e.g., "weights", "kv_cache")

        Returns:
            AllocationInfo with allocation_id, aligned_size, handle

        Raises:
            RuntimeError: If CUDA allocation fails
        """
        aligned_size = self._align(size)
        cuda = get_cuda_driver()

        # Set up allocation properties for shareable handle
        prop = CUmemAllocationProp()
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = self._device
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR

        # Create physical allocation (no VA mapping!)
        handle = CUmemGenericAllocationHandle()
        check_cuda_result(
            cuda.cuMemCreate(
                byref(handle), c_size_t(aligned_size), byref(prop), c_ulonglong(0)
            ),
            "cuMemCreate",
        )

        allocation_id = str(uuid4())
        info = AllocationInfo(
            allocation_id=allocation_id,
            size=size,
            aligned_size=aligned_size,
            handle=handle.value,
            tag=tag,
            created_at=time.time(),
        )
        self._allocations[allocation_id] = info

        logger.debug(
            f"Allocated {allocation_id}: size={size}, aligned_size={aligned_size}, tag={tag}"
        )
        return info

    def export_fd(self, allocation_id: str) -> int:
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
        info = self._allocations.get(allocation_id)
        if info is None:
            raise AllocationNotFoundError(f"Unknown allocation: {allocation_id}")

        cuda = get_cuda_driver()
        fd = c_int()
        check_cuda_result(
            cuda.cuMemExportToShareableHandle(
                byref(fd),
                CUmemGenericAllocationHandle(info.handle),
                c_int(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR),
                c_ulonglong(0),
            ),
            "cuMemExportToShareableHandle",
        )
        return fd.value

    def free(self, allocation_id: str) -> bool:
        """Release physical memory for a single allocation.

        Args:
            allocation_id: ID of allocation to free

        Returns:
            True if allocation existed and was freed, False otherwise
        """
        info = self._allocations.pop(allocation_id, None)
        if info is None:
            return False

        cuda = get_cuda_driver()
        cuda.cuMemRelease(CUmemGenericAllocationHandle(info.handle))
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
        cuda = get_cuda_driver()
        for info in self._allocations.values():
            cuda.cuMemRelease(CUmemGenericAllocationHandle(info.handle))
        self._allocations.clear()
        logger.info(f"Cleared {count} allocations")
        return count

    def get_allocation(self, allocation_id: str) -> AllocationInfo:
        """Get allocation info.

        Args:
            allocation_id: ID to look up

        Returns:
            AllocationInfo

        Raises:
            AllocationNotFoundError: If not found
        """
        info = self._allocations.get(allocation_id)
        if info is None:
            raise AllocationNotFoundError(f"Unknown allocation: {allocation_id}")
        return info

    def list_allocations(self, tag: Optional[str] = None) -> List[AllocationInfo]:
        """List all allocations, optionally filtered by tag.

        Args:
            tag: If provided, only return allocations with this tag

        Returns:
            List of AllocationInfo objects
        """
        if tag is None:
            return list(self._allocations.values())
        return [info for info in self._allocations.values() if info.tag == tag]
