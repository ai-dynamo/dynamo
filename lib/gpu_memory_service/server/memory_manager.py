# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA VMM allocation manager - pure business logic, no threading/transport.

This module contains the GMSServerMemoryManager class which handles physical GPU memory
allocations via CUDA Virtual Memory Management (VMM) API. It creates shareable
memory without mapping it locally (no CUDA context needed on the server).

The GMSServerMemoryManager is NOT thread-safe. Callers must provide external
synchronization (e.g., via LockManager ensuring single-writer access).
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional
from uuid import uuid4

from gpu_memory_service.common.cuda_vmm_utils import (
    align_to_granularity,
    cuda_ensure_initialized,
    cumem_create_tolerate_oom,
    cumem_export_to_shareable_handle,
    cumem_get_allocation_granularity,
    cumem_release,
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
    epoch_id: int
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

    def __init__(
        self,
        device: int = 0,
        *,
        allocation_retry_interval: float = 0.5,
        allocation_retry_timeout: Optional[float] = None,
    ):
        if allocation_retry_interval <= 0:
            raise ValueError(
                f"allocation_retry_interval must be > 0, got {allocation_retry_interval}"
            )
        if allocation_retry_timeout is not None and allocation_retry_timeout <= 0:
            raise ValueError(
                f"allocation_retry_timeout must be > 0 when set, got {allocation_retry_timeout}"
            )

        self._device = device
        self._allocations: dict[str, AllocationInfo] = {}
        cuda_ensure_initialized()
        self._granularity = cumem_get_allocation_granularity(device)
        self._allocation_retry_interval = allocation_retry_interval
        self._allocation_retry_timeout = allocation_retry_timeout
        logger.info(
            "GMSServerMemoryManager initialized: device=%d, granularity=%d, "
            "alloc_retry_interval=%.3f, alloc_retry_timeout=%s",
            device,
            self._granularity,
            self._allocation_retry_interval,
            (
                f"{self._allocation_retry_timeout:.3f}"
                if self._allocation_retry_timeout is not None
                else "none"
            ),
        )

    @property
    def device(self) -> int:
        return self._device

    @property
    def allocation_count(self) -> int:
        return len(self._allocations)

    def _retry_timeout_exceeded(self, started_at: float) -> bool:
        if self._allocation_retry_timeout is None:
            return False
        return (time.monotonic() - started_at) >= self._allocation_retry_timeout

    async def allocate(
        self,
        size: int,
        tag: str = "default",
        epoch_id: int = 0,
        is_connected: Optional[Callable[[], bool]] = None,
    ) -> AllocationInfo:
        """Create a physical memory allocation (no VA mapping).

        Uses cuMemCreate to allocate physical GPU memory that can be exported
        as a file descriptor for sharing with other processes.

        On CUDA OOM, this method blocks and retries until allocation succeeds
        (or optional retry timeout is reached).

        Args:
            size: Requested size in bytes (will be aligned up to granularity)
            tag: Tag for grouping allocations (e.g., "weights", "kv_cache")
            epoch_id: Epoch that owns this allocation

        Returns:
            AllocationInfo with allocation_id, aligned_size, handle

        Raises:
            RuntimeError: If CUDA allocation fails
        """
        if size <= 0:
            raise ValueError(f"size must be > 0, got {size}")
        if epoch_id <= 0:
            raise ValueError(f"epoch_id must be > 0, got {epoch_id}")

        aligned_size = align_to_granularity(size, self._granularity)

        started_at = time.monotonic()
        while True:
            if is_connected is not None and not is_connected():
                raise ConnectionAbortedError(
                    "RW client disconnected during allocation retry"
                )

            allocated, handle = cumem_create_tolerate_oom(aligned_size, self._device)
            if allocated:
                break

            if (
                self._allocation_retry_timeout is not None
                and (time.monotonic() - started_at) >= self._allocation_retry_timeout
            ):
                raise TimeoutError(
                    "Timed out waiting for GPU memory: "
                    f"requested_size={size}, aligned_size={aligned_size}, tag={tag}, "
                    f"epoch={epoch_id}, "
                    f"waited_sec={time.monotonic() - started_at:.3f}"
                )

            logger.warning(
                "cuMemCreate OOM for aligned_size=%d bytes, tag=%s, epoch=%s; "
                "retrying in %.3fs",
                aligned_size,
                tag,
                epoch_id,
                self._allocation_retry_interval,
            )
            await asyncio.sleep(self._allocation_retry_interval)

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

    def export_allocation(self, allocation_id: str, epoch_id: int) -> int:
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
        info = self.get_allocation(allocation_id, epoch_id)
        return cumem_export_to_shareable_handle(info.handle)

    def free_allocation(self, allocation_id: str, epoch_id: int) -> bool:
        """Release physical memory for a single allocation.

        Args:
            allocation_id: ID of allocation to free

        Returns:
            True if allocation existed and was freed, False otherwise
        """
        info = self._allocations.get(allocation_id)
        if info is None:
            return False
        if info.epoch_id != epoch_id:
            logger.debug(
                "Free skipped due to epoch mismatch: allocation_id=%s allocation_epoch=%s requested_epoch=%s",
                allocation_id,
                info.epoch_id,
                epoch_id,
            )
            return False
        self._allocations.pop(allocation_id, None)
        cumem_release(info.handle)
        logger.debug(f"Freed allocation: {allocation_id}")
        return True

    def clear_all_allocations(self, epoch_id: int) -> int:
        """Release all allocations in a specific epoch."""
        to_clear = [
            allocation_id
            for allocation_id, info in self._allocations.items()
            if info.epoch_id == epoch_id
        ]
        for allocation_id in to_clear:
            info = self._allocations.pop(allocation_id)
            cumem_release(info.handle)
        if to_clear:
            logger.info(f"Cleared {len(to_clear)} allocations from epoch {epoch_id}")
        return len(to_clear)

    def get_allocation(self, allocation_id: str, epoch_id: int) -> AllocationInfo:
        """Get allocation info. Raises AllocationNotFoundError if not found."""
        info = self._allocations.get(allocation_id)
        if info is None:
            raise AllocationNotFoundError(f"Unknown allocation: {allocation_id}")
        if info.epoch_id != epoch_id:
            raise AllocationNotFoundError(
                f"Allocation {allocation_id} is not in epoch {epoch_id}"
            )
        return info

    def list_allocations(
        self, epoch_id: int, tag: Optional[str] = None
    ) -> list[AllocationInfo]:
        """List all allocations, optionally filtered by tag."""
        allocations = [
            info for info in self._allocations.values() if info.epoch_id == epoch_id
        ]
        if tag is None:
            return allocations
        return [info for info in allocations if info.tag == tag]
