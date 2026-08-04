# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""V0 allocation catalog over shared physical GMS ownership."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional
from uuid import uuid4

from gpu_memory_service.common.utils import align_to_granularity
from gpu_memory_service.common.vmm import get_vmm
from gpu_memory_service.core.server.allocations import (
    GMSAllocationManager as CoreAllocationManager,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AllocationInfo:
    allocation_id: str
    size: int
    aligned_size: int
    tag: str
    layout_slot: int


class AllocationNotFoundError(Exception):
    """Raised when an allocation ID is not present in the active V0 catalog."""


class GMSAllocationManager:
    """Add V0 IDs, tags, layout slots, and allocation retry to the core."""

    def __init__(
        self,
        device: int = 0,
        *,
        allocation_retry_interval: float = 0.5,
        allocation_retry_timeout: Optional[float] = 60.0,
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
        self._vmm = get_vmm()
        self._physical = CoreAllocationManager(self._vmm, device)
        self._granularity = self._vmm.get_allocation_granularity(device)
        self._allocations: dict[str, AllocationInfo] = {}
        self._next_layout_slot = 0
        self._allocation_retry_interval = allocation_retry_interval
        self._allocation_retry_timeout = allocation_retry_timeout
        logger.info(
            "GMSAllocationManager initialized: device=%d, "
            "granularity=%d, alloc_retry_interval=%.3f, alloc_retry_timeout=%s",
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
        tag: str = "default",
        is_connected: Optional[Callable[[], bool]] = None,
        on_oom: Optional[Callable[[], None]] = None,
    ) -> AllocationInfo:
        if size <= 0:
            raise ValueError(f"size must be > 0, got {size}")

        aligned_size = align_to_granularity(size, self._granularity)
        allocation_id = str(uuid4())
        started_at = time.monotonic()
        reported_oom = False
        while True:
            if is_connected is not None and not is_connected():
                raise ConnectionAbortedError(
                    "RW client disconnected during allocation retry"
                )

            try:
                self._physical.allocate(allocation_id, aligned_size)
                break
            except MemoryError:
                pass

            if on_oom is not None and not reported_oom:
                on_oom()
                reported_oom = True

            if self._allocation_retry_timeout is not None:
                waited = time.monotonic() - started_at
                if waited >= self._allocation_retry_timeout:
                    raise TimeoutError(
                        "Timed out waiting for GPU memory: "
                        f"requested_size={size}, aligned_size={aligned_size}, "
                        f"tag={tag}, waited_sec={waited:.3f}"
                    )

            free_b, total_b = -1, -1
            try:
                free_b, total_b = self._vmm.device_memory_info(self._device)
            except Exception:
                logger.debug(
                    "device memory info failed for device %d",
                    self._device,
                    exc_info=True,
                )
            elapsed = time.monotonic() - started_at
            logger.warning(
                "physical allocation OOM for aligned_size=%d bytes, tag=%s, "
                "elapsed=%.2fs free=%d total=%d; retrying in %.3fs",
                aligned_size,
                tag,
                elapsed,
                free_b,
                total_b,
                self._allocation_retry_interval,
            )
            await asyncio.sleep(self._allocation_retry_interval)

        info = AllocationInfo(
            allocation_id=allocation_id,
            size=size,
            aligned_size=aligned_size,
            tag=tag,
            layout_slot=self._next_layout_slot,
        )
        self._next_layout_slot += 1
        self._allocations[allocation_id] = info
        logger.debug(
            "Allocated %s: size=%d, aligned=%d, tag=%s, slot=%d",
            allocation_id,
            size,
            aligned_size,
            tag,
            info.layout_slot,
        )
        return info

    def export_allocation(self, allocation_id: str) -> int:
        self.get_allocation(allocation_id)
        return self._physical.export(allocation_id)

    def free_allocation(self, allocation_id: str) -> bool:
        if allocation_id not in self._allocations:
            return False
        self._physical.free(allocation_id)
        del self._allocations[allocation_id]
        logger.debug("Freed allocation: %s", allocation_id)
        return True

    def clear_all(self) -> int:
        count = self._physical.clear()
        self._allocations.clear()
        self._next_layout_slot = 0
        if count:
            logger.info("Cleared %d allocations", count)
        return count

    def get_allocation(self, allocation_id: str) -> AllocationInfo:
        try:
            return self._allocations[allocation_id]
        except KeyError:
            raise AllocationNotFoundError(
                f"Unknown allocation: {allocation_id}"
            ) from None

    def list_allocations(self, tag: Optional[str] = None) -> list[AllocationInfo]:
        allocations = sorted(
            self._allocations.values(), key=lambda info: info.layout_slot
        )
        if tag is None:
            return allocations
        return [info for info in allocations if info.tag == tag]
