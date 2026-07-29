# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Server ownership of physical GMS allocations."""

from __future__ import annotations

import threading

from gpu_memory_service.common.vmm import VMMDevice


class GMSAllocationManager:
    """Own retained device allocation handles by opaque allocation ID."""

    def __init__(self, vmm: VMMDevice, device: int):
        self._vmm = vmm
        self._device = device
        self._vmm.ensure_initialized()
        self._granularity = int(self._vmm.get_allocation_granularity(device))
        if self._granularity <= 0:
            raise ValueError("allocation granularity must be positive")
        self._allocations: dict[str, tuple[int, int]] = {}
        self._lock = threading.Lock()

    def allocate(self, allocation_id: str, aligned_size: int) -> None:
        with self._lock:
            if not allocation_id:
                raise RuntimeError("allocation ID must not be empty")
            if aligned_size <= 0 or aligned_size % self._granularity:
                raise RuntimeError("allocation size is not aligned for this GPU")
            if allocation_id in self._allocations:
                raise RuntimeError("allocation ID already exists")
            allocated, handle = self._vmm.create_tolerate_oom(
                aligned_size, self._device
            )
            if not allocated:
                raise MemoryError(f"cannot allocate {aligned_size} GPU bytes")
            self._allocations[allocation_id] = (int(handle), aligned_size)

    def export(self, allocation_id: str) -> int:
        with self._lock:
            handle, _aligned_size = self._get(allocation_id)
            return int(self._vmm.export_to_shareable_handle(handle))

    def free(self, allocation_id: str) -> None:
        with self._lock:
            handle, _aligned_size = self._get(allocation_id)
            self._vmm.release(handle)
            del self._allocations[allocation_id]

    def list_allocations(self) -> tuple[tuple[str, int], ...]:
        with self._lock:
            return tuple(
                (allocation_id, aligned_size)
                for allocation_id, (_handle, aligned_size) in self._allocations.items()
            )

    def clear(self) -> int:
        with self._lock:
            allocation_ids = tuple(self._allocations)
            for allocation_id in allocation_ids:
                handle, _aligned_size = self._allocations[allocation_id]
                self._vmm.release(handle)
                del self._allocations[allocation_id]
            return len(allocation_ids)

    def _get(self, allocation_id: str) -> tuple[int, int]:
        if not allocation_id:
            raise RuntimeError("allocation ID must not be empty")
        try:
            return self._allocations[allocation_id]
        except KeyError:
            raise RuntimeError("unknown allocation ID") from None
