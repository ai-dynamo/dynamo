# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Server-side CUDA allocation store."""

from __future__ import annotations

import asyncio
import logging
import os
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
class BackingAllocation:
    backing_id: str
    size: int
    aligned_size: int
    handle: int
    export_fd: int
    created_at: float


@dataclass(frozen=True)
class AllocationInfo:
    allocation_id: str
    size: int
    aligned_size: int
    handle: int
    export_fd: int
    tag: str
    layout_slot: int
    created_at: float
    backing_id: str = ""
    backing_offset: int = 0


class AllocationNotFoundError(Exception):
    """Raised when an allocation_id doesn't exist."""


class GMSAllocationManager:
    """Server-side CUDA VMM allocation store."""

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
        self._allocations: dict[str, AllocationInfo] = {}
        self._backings: dict[str, BackingAllocation] = {}
        self._next_layout_slot = 0
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

    async def _allocate_backing(
        self,
        size: int,
        is_connected: Optional[Callable[[], bool]] = None,
        on_oom: Optional[Callable[[], None]] = None,
        *,
        oom_tag: str = "backing",
    ) -> BackingAllocation:
        if size <= 0:
            raise ValueError(f"size must be > 0, got {size}")

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
                        f"tag={oom_tag}, waited_sec={waited:.3f}"
                    )

            # Visibility while retrying. Logged every iteration with elapsed
            # time + free GPU memory, so a stuck retry loop is observable
            # rather than silent.
            try:
                import torch

                free_b, total_b = torch.cuda.mem_get_info(self._device)
            except (ImportError, RuntimeError):
                logger.debug(
                    "torch.cuda.mem_get_info(%d) failed", self._device, exc_info=True
                )
                free_b, total_b = -1, -1
            elapsed = time.monotonic() - started_at
            logger.warning(
                "cuMemCreate OOM for aligned_size=%d bytes, tag=%s, "
                "elapsed=%.2fs free=%d total=%d; retrying in %.3fs",
                aligned_size,
                oom_tag,
                elapsed,
                free_b,
                total_b,
                self._allocation_retry_interval,
            )
            await asyncio.sleep(self._allocation_retry_interval)

        export_fd = int(cumem_export_to_shareable_handle(int(handle)))
        backing = BackingAllocation(
            backing_id=str(uuid4()),
            size=size,
            aligned_size=aligned_size,
            handle=int(handle),
            export_fd=export_fd,
            created_at=time.time(),
        )
        self._backings[backing.backing_id] = backing
        return backing

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
        backing = await self._allocate_backing(
            aligned_size,
            is_connected=is_connected,
            on_oom=on_oom,
            oom_tag=tag,
        )
        info = AllocationInfo(
            allocation_id=backing.backing_id,
            size=size,
            aligned_size=aligned_size,
            handle=backing.handle,
            export_fd=backing.export_fd,
            tag=tag,
            layout_slot=self._next_layout_slot,
            created_at=time.time(),
            backing_id=backing.backing_id,
            backing_offset=0,
        )
        self._next_layout_slot = info.layout_slot + 1
        self._allocations[info.allocation_id] = info
        logger.debug(
            "Allocated %s: size=%d, aligned=%d, tag=%s, slot=%d",
            info.allocation_id,
            size,
            aligned_size,
            tag,
            info.layout_slot,
        )
        return info

    async def create_packed_layout(
        self,
        backing_sizes: list[int],
        placements: list[tuple[int, int, str, int, int]],
        *,
        is_connected: Optional[Callable[[], bool]] = None,
        on_oom: Optional[Callable[[], None]] = None,
    ) -> list[AllocationInfo]:
        """Create slab-backed published allocations for snapshot restore.

        Args:
            backing_sizes: Physical slab sizes.
            placements: Tuples of
                (size, aligned_size, tag, backing_index, backing_offset).

        Returns:
            Published allocation records, one per placement.
        """
        if not placements:
            return []
        if not backing_sizes:
            raise ValueError("packed layout requires at least one backing")

        backings: list[BackingAllocation] = []
        initial_layout_slot = self._next_layout_slot
        try:
            for index, size in enumerate(backing_sizes):
                aligned_size = align_to_granularity(size, self._granularity)
                backings.append(
                    await self._allocate_backing(
                        aligned_size,
                        is_connected=is_connected,
                        on_oom=on_oom,
                        oom_tag=f"packed_restore_slab_{index}",
                    )
                )

            infos: list[AllocationInfo] = []
            for size, aligned_size, tag, backing_index, backing_offset in placements:
                if size <= 0:
                    raise ValueError(f"placement size must be > 0, got {size}")
                if aligned_size <= 0:
                    raise ValueError(
                        f"placement aligned_size must be > 0, got {aligned_size}"
                    )
                if backing_index < 0 or backing_index >= len(backings):
                    raise ValueError(
                        f"placement backing_index {backing_index} out of range "
                        f"for {len(backings)} backing(s)"
                    )
                if backing_offset < 0:
                    raise ValueError(
                        f"placement backing_offset must be >= 0, got {backing_offset}"
                    )
                aligned_size = align_to_granularity(aligned_size, self._granularity)
                if backing_offset % self._granularity:
                    raise ValueError(
                        "placement backing_offset must be aligned to granularity "
                        f"{self._granularity}, got {backing_offset}"
                    )
                backing = backings[backing_index]
                if backing_offset + aligned_size > backing.aligned_size:
                    raise ValueError(
                        "placement exceeds backing allocation: "
                        f"offset={backing_offset} aligned_size={aligned_size} "
                        f"backing_size={backing.aligned_size}"
                    )

                info = AllocationInfo(
                    allocation_id=str(uuid4()),
                    size=int(size),
                    aligned_size=int(aligned_size),
                    handle=backing.handle,
                    export_fd=backing.export_fd,
                    tag=str(tag),
                    layout_slot=self._next_layout_slot,
                    created_at=time.time(),
                    backing_id=backing.backing_id,
                    backing_offset=int(backing_offset),
                )
                self._next_layout_slot = info.layout_slot + 1
                self._allocations[info.allocation_id] = info
                infos.append(info)
            logger.info(
                "Created packed layout: %d published allocations backed by "
                "%d slab allocation(s), %.2f GiB backing",
                len(infos),
                len(backings),
                sum(b.aligned_size for b in backings) / (1 << 30),
            )
            return infos
        except Exception:
            # If any validation fails after slabs were allocated, do not leave
            # hidden slab allocations stranded in an otherwise empty RW layout.
            for info in list(self._allocations.values()):
                if info.backing_id in {backing.backing_id for backing in backings}:
                    self._allocations.pop(info.allocation_id, None)
            for backing in backings:
                self._free_backing_if_unused(backing.backing_id, force=True)
            self._next_layout_slot = initial_layout_slot
            raise

    def export_allocation(self, allocation_id: str) -> int:
        info = self.get_allocation(allocation_id)
        backing = self._get_backing(info.backing_id or info.allocation_id)
        return os.dup(backing.export_fd)

    def _get_backing(self, backing_id: str) -> BackingAllocation:
        backing = self._backings.get(backing_id)
        if backing is None:
            raise AllocationNotFoundError(f"Unknown backing allocation: {backing_id}")
        return backing

    def _free_backing_if_unused(self, backing_id: str, *, force: bool = False) -> None:
        if not force and any(
            info.backing_id == backing_id for info in self._allocations.values()
        ):
            return
        backing = self._backings.pop(backing_id, None)
        if backing is None:
            return
        os.close(backing.export_fd)
        cumem_release(backing.handle)

    def free_allocation(self, allocation_id: str) -> bool:
        info = self._allocations.get(allocation_id)
        if info is None:
            return False
        self._allocations.pop(allocation_id, None)
        self._free_backing_if_unused(info.backing_id or info.allocation_id)
        logger.debug("Freed allocation: %s", allocation_id)
        return True

    def clear_all(self) -> int:
        allocation_ids = list(self._allocations)
        self._allocations.clear()
        backing_ids = list(self._backings)
        for backing_id in backing_ids:
            self._free_backing_if_unused(backing_id, force=True)
        if allocation_ids:
            logger.info("Cleared %d allocations", len(allocation_ids))
        self._next_layout_slot = 0
        return len(allocation_ids)

    def get_allocation(self, allocation_id: str) -> AllocationInfo:
        info = self._allocations.get(allocation_id)
        if info is None:
            raise AllocationNotFoundError(f"Unknown allocation: {allocation_id}")
        return info

    def list_allocations(self, tag: Optional[str] = None) -> list[AllocationInfo]:
        allocations = list(self._allocations.values())
        allocations.sort(key=lambda info: info.layout_slot)
        if tag is None:
            return allocations
        return [info for info in allocations if info.tag == tag]
