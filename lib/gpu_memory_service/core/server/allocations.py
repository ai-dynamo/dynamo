# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Server ownership of physical GMS allocations."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from gpu_memory_service.common.vmm import VMMDevice

from ..errors import AllocationNotFoundError, FatalGMSError, GMSError


@dataclass(frozen=True)
class AllocationInfo:
    aligned_size: int
    handle: int


class GMSAllocationManager:
    """Own retained device allocation handles by opaque allocation ID."""

    def __init__(self, vmm: VMMDevice, device: int):
        self._vmm = vmm
        self._device = device
        self._vmm.ensure_initialized()
        self._granularity = int(self._vmm.get_allocation_granularity(device))
        if self._granularity <= 0:
            raise ValueError("allocation granularity must be positive")
        self._allocations: dict[str, AllocationInfo] = {}
        self._fatal: FatalGMSError | None = None
        self._lock = threading.Lock()

    @property
    def allocation_count(self) -> int:
        with self._lock:
            return len(self._allocations)

    def allocate(self, allocation_id: str, aligned_size: int) -> None:
        with self._lock:
            self._check()
            self._validate_id(allocation_id)
            if aligned_size <= 0 or aligned_size % self._granularity:
                raise GMSError("allocation size is not aligned for this GPU")
            if allocation_id in self._allocations:
                raise GMSError("allocation ID already exists")

            allocated, handle = self._vmm.create_tolerate_oom(
                aligned_size, self._device
            )
            if not allocated:
                raise MemoryError(f"cannot allocate {aligned_size} GPU bytes")
            self._allocations[allocation_id] = AllocationInfo(aligned_size, int(handle))

    def get(self, allocation_id: str) -> AllocationInfo:
        with self._lock:
            self._check()
            self._validate_id(allocation_id)
            try:
                return self._allocations[allocation_id]
            except KeyError:
                raise AllocationNotFoundError("unknown allocation ID") from None

    def list(self) -> list[tuple[str, AllocationInfo]]:
        with self._lock:
            self._check()
            return list(self._allocations.items())

    def export(self, allocation_id: str) -> int:
        with self._lock:
            self._check()
            self._validate_id(allocation_id)
            try:
                allocation = self._allocations[allocation_id]
            except KeyError:
                raise AllocationNotFoundError("unknown allocation ID") from None
            return int(self._vmm.export_to_shareable_handle(allocation.handle))

    def free(self, allocation_id: str) -> None:
        with self._lock:
            self._check()
            self._validate_id(allocation_id)
            try:
                allocation = self._allocations[allocation_id]
            except KeyError:
                raise AllocationNotFoundError("unknown allocation ID") from None
            try:
                self._vmm.release(allocation.handle)
            except Exception as cause:
                raise self._latch("server allocation cleanup failed", cause) from cause
            del self._allocations[allocation_id]

    def clear(self) -> int:
        with self._lock:
            self._check()
            allocation_ids = tuple(self._allocations)
            for allocation_id in allocation_ids:
                allocation = self._allocations[allocation_id]
                try:
                    self._vmm.release(allocation.handle)
                except Exception as cause:
                    raise self._latch(
                        "server allocation cleanup failed", cause
                    ) from cause
                del self._allocations[allocation_id]
            return len(allocation_ids)

    def _check(self) -> None:
        if self._fatal is not None:
            raise self._fatal

    def _latch(self, message: str, cause: Exception) -> FatalGMSError:
        if self._fatal is None:
            self._fatal = FatalGMSError(f"{message}: {cause}")
        return self._fatal

    @staticmethod
    def _validate_id(allocation_id: str) -> None:
        if not allocation_id:
            raise GMSError("allocation ID must not be empty")
