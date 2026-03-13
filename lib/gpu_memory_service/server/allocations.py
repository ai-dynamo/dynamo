# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Server-side CUDA allocation store."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional
from uuid import uuid4

from gpu_memory_service.common.cuda_utils import (
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
    allocation_id: str
    size: int
    aligned_size: int
    handle: int
    tag: str
    epoch_id: int
    layout_slot: int
    created_at: float


class AllocationNotFoundError(Exception):
    """Raised when an allocation_id doesn't exist."""


class GMSAllocationManager:
    """Server-side CUDA VMM allocation store."""

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
        self._next_layout_slot_by_epoch: dict[int, int] = {}
        cuda_ensure_initialized()
        self._granularity = cumem_get_allocation_granularity(device)
        self._allocation_retry_interval = allocation_retry_interval
        self._allocation_retry_timeout = allocation_retry_timeout
        logger.info(
            "GMSAllocationManager initialized: device=%d, granularity=%d, "
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

    async def allocate(
        self,
        size: int,
        epoch_id: int,
        tag: str = "default",
        is_connected: Optional[Callable[[], bool]] = None,
        on_oom: Optional[Callable[[], None]] = None,
    ) -> AllocationInfo:
        if size <= 0:
            raise ValueError(f"size must be > 0, got {size}")
        if epoch_id <= 0:
            raise ValueError(f"epoch_id must be > 0, got {epoch_id}")

        aligned_size = align_to_granularity(size, self._granularity)
        started_at = time.monotonic()
        reported_oom = False
        while True:
            if is_connected is not None and not is_connected():
                raise ConnectionAbortedError(
                    "RW client disconnected during allocation retry"
                )

            allocated, handle = cumem_create_tolerate_oom(aligned_size, self._device)
            if allocated:
                break

            if on_oom is not None and not reported_oom:
                on_oom()
                reported_oom = True

            if self._allocation_retry_timeout is not None:
                waited = time.monotonic() - started_at
                if waited >= self._allocation_retry_timeout:
                    raise TimeoutError(
                        "Timed out waiting for GPU memory: "
                        f"requested_size={size}, aligned_size={aligned_size}, "
                        f"tag={tag}, epoch={epoch_id}, waited_sec={waited:.3f}"
                    )

            logger.warning(
                "cuMemCreate OOM for aligned_size=%d bytes, tag=%s, epoch=%s; retrying in %.3fs",
                aligned_size,
                tag,
                epoch_id,
                self._allocation_retry_interval,
            )
            await asyncio.sleep(self._allocation_retry_interval)

        info = AllocationInfo(
            allocation_id=str(uuid4()),
            size=size,
            aligned_size=aligned_size,
            handle=int(handle),
            tag=tag,
            epoch_id=epoch_id,
            layout_slot=self._next_layout_slot_by_epoch.get(epoch_id, 0),
            created_at=time.time(),
        )
        self._next_layout_slot_by_epoch[epoch_id] = info.layout_slot + 1
        self._allocations[info.allocation_id] = info
        logger.debug(
            "Allocated %s: size=%d, aligned=%d, tag=%s, epoch=%d, slot=%d",
            info.allocation_id,
            size,
            aligned_size,
            tag,
            epoch_id,
            info.layout_slot,
        )
        return info

    def export_allocation(self, allocation_id: str, epoch_id: int) -> int:
        return cumem_export_to_shareable_handle(
            self.get_allocation(allocation_id, epoch_id).handle
        )

    def free_allocation(self, allocation_id: str, epoch_id: int) -> bool:
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
        cumem_release(info.handle)
        self._allocations.pop(allocation_id, None)
        logger.debug("Freed allocation: %s", allocation_id)
        return True

    def clear_all_allocations(self, epoch_id: int) -> int:
        allocation_ids = [
            allocation_id
            for allocation_id, info in self._allocations.items()
            if info.epoch_id == epoch_id
        ]
        for allocation_id in allocation_ids:
            info = self._allocations[allocation_id]
            cumem_release(info.handle)
            self._allocations.pop(allocation_id, None)
        if allocation_ids:
            logger.info(
                "Cleared %d allocations from epoch %d",
                len(allocation_ids),
                epoch_id,
            )
        self._next_layout_slot_by_epoch.pop(epoch_id, None)
        return len(allocation_ids)

    def get_allocation(self, allocation_id: str, epoch_id: int) -> AllocationInfo:
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
        allocations = [
            info for info in self._allocations.values() if info.epoch_id == epoch_id
        ]
        allocations.sort(key=lambda info: info.layout_slot)
        if tag is None:
            return allocations
        return [info for info in allocations if info.tag == tag]
