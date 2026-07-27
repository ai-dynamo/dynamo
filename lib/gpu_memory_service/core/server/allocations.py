# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Server ownership of physical GMS allocations."""

from __future__ import annotations

import threading

from gpu_memory_service.common.vmm import VMMDevice

from ..errors import GMSError


class GMSAllocationManager:
    """Own retained device allocation handles by opaque allocation ID."""

    def __init__(self, vmm: VMMDevice, device: int):
        self._vmm = vmm
        self._device = device
        self._vmm.ensure_initialized()
        self._granularity = int(self._vmm.get_allocation_granularity(device))
        if self._granularity <= 0:
            raise ValueError("allocation granularity must be positive")
        self._allocations: dict[str, int] = {}
        self._fatal: GMSError | None = None
        self._lock = threading.Lock()

    def allocate(self, allocation_id: str, aligned_size: int) -> None:
        with self._lock:
            self._check()
            if not allocation_id:
                raise GMSError("allocation ID must not be empty")
            if aligned_size <= 0 or aligned_size % self._granularity:
                raise GMSError("allocation size is not aligned for this GPU")
            if allocation_id in self._allocations:
                raise GMSError("allocation ID already exists")
            allocated, handle = self._vmm.create_tolerate_oom(
                aligned_size, self._device
            )
            if not allocated:
                raise MemoryError(f"cannot allocate {aligned_size} GPU bytes")
            self._allocations[allocation_id] = int(handle)

    def export(self, allocation_id: str) -> int:
        with self._lock:
            self._check()
            handle = self._get(allocation_id)
            return int(self._vmm.export_to_shareable_handle(handle))

    def free(self, allocation_id: str) -> None:
        with self._lock:
            self._check()
            handle = self._get(allocation_id)
            try:
                self._vmm.release(handle)
            except Exception as exc:
                raise self._latch("server allocation cleanup failed", exc) from exc
            del self._allocations[allocation_id]

    def clear(self) -> int:
        with self._lock:
            self._check()
            allocation_ids = tuple(self._allocations)
            for allocation_id in allocation_ids:
                handle = self._allocations[allocation_id]
                try:
                    self._vmm.release(handle)
                except Exception as exc:
                    raise self._latch("server allocation cleanup failed", exc) from exc
                del self._allocations[allocation_id]
            return len(allocation_ids)

    def _get(self, allocation_id: str) -> int:
        if not allocation_id:
            raise GMSError("allocation ID must not be empty")
        try:
            return self._allocations[allocation_id]
        except KeyError:
            raise GMSError("unknown allocation ID") from None

    def _check(self) -> None:
        if self._fatal is not None:
            raise self._fatal

    def _latch(self, message: str, cause: Exception) -> GMSError:
        if self._fatal is None:
            self._fatal = GMSError(f"{message}: {cause}")
        return self._fatal
