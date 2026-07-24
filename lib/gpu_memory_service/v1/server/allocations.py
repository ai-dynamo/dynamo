# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The complete server-owned state for snapshot-only GMS V1."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from uuid import uuid4

from gpu_memory_service.common.vmm import VMMDevice

from ..errors import FatalGMSError, GMSError


@dataclass
class _PhysicalAllocation:
    size: int
    handle: int
    export_fd: int


class AllocationStore:
    """One locked allocation-ID to physical-allocation store."""

    def __init__(self, gpu_uuid: str, vmm: VMMDevice, device: int):
        if not gpu_uuid:
            raise ValueError("GPU UUID must not be empty")
        self.server_nonce = str(uuid4())
        self.gpu_uuid = gpu_uuid
        self._vmm = vmm
        self._device = device
        self._vmm.ensure_initialized()
        self._granularity = int(self._vmm.get_allocation_granularity(device))
        if self._granularity <= 0:
            raise ValueError("allocation granularity must be positive")
        self._allocations: dict[str, _PhysicalAllocation] = {}
        self._fatal: FatalGMSError | None = None
        self._lock = threading.Lock()

    def hello(self) -> tuple[str, str]:
        with self._lock:
            self._check()
            return self.server_nonce, self.gpu_uuid

    def allocate(self, allocation_id: str, aligned_size: int) -> None:
        with self._lock:
            self._check()
            self._validate_id(allocation_id)
            if aligned_size <= 0 or aligned_size % self._granularity:
                raise GMSError("allocation size is not aligned for this GPU")
            existing = self._allocations.get(allocation_id)
            if existing is not None:
                if existing.size != aligned_size:
                    raise GMSError("allocation ID was replayed with another size")
                return

            allocated, handle = self._vmm.create_tolerate_oom(
                aligned_size, self._device
            )
            if not allocated:
                raise MemoryError(f"cannot allocate {aligned_size} GPU bytes")
            try:
                export_fd = int(self._vmm.export_to_shareable_handle(int(handle)))
            except Exception as cause:
                try:
                    self._vmm.release(int(handle))
                except Exception as cleanup:
                    # The fatal latch prevents use, but retain the handle as
                    # exact ownership evidence for diagnosis.
                    self._allocations[allocation_id] = _PhysicalAllocation(
                        aligned_size, int(handle), -1
                    )
                    raise self._latch(
                        "allocation export cleanup failed", cleanup
                    ) from cause
                raise
            self._allocations[allocation_id] = _PhysicalAllocation(
                aligned_size, int(handle), export_fd
            )

    def export(self, allocation_id: str) -> int:
        with self._lock:
            self._check()
            self._validate_id(allocation_id)
            allocation = self._allocations.get(allocation_id)
            if allocation is None:
                raise GMSError("unknown allocation ID")
            if allocation.export_fd < 0 or allocation.handle == 0:
                raise self._latch(
                    "server allocation ownership is incomplete",
                    GMSError(allocation_id),
                )
            return os.dup(allocation.export_fd)

    def free(self, allocation_id: str) -> None:
        """Free an allocation; replay after a lost response is a no-op."""
        with self._lock:
            self._check()
            self._validate_id(allocation_id)
            allocation = self._allocations.get(allocation_id)
            if allocation is None:
                return
            failures: list[Exception] = []
            if allocation.export_fd >= 0:
                try:
                    os.close(allocation.export_fd)
                except Exception as exc:
                    failures.append(exc)
                else:
                    allocation.export_fd = -1
            if allocation.handle:
                try:
                    self._vmm.release(allocation.handle)
                except Exception as exc:
                    failures.append(exc)
                else:
                    allocation.handle = 0
            if failures:
                raise self._latch("server allocation cleanup failed", failures[0])
            del self._allocations[allocation_id]

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
