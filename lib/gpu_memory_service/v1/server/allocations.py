# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""V1 server ownership of physical allocations by opaque ID."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from gpu_memory_service.common.vmm import VMMDevice

logger = logging.getLogger(__name__)

_ALLOCATION_RETRY_INTERVAL = 0.5


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
        self._lock = threading.Lock()

    def allocate(
        self,
        allocation_id: str,
        aligned_size: int,
        is_connected: Callable[[], bool] | None = None,
    ) -> None:
        with self._lock:
            if not allocation_id:
                raise RuntimeError("allocation ID must not be empty")
            if aligned_size <= 0 or aligned_size % self._granularity:
                raise RuntimeError("allocation size is not aligned for this GPU")
            if allocation_id in self._allocations:
                raise RuntimeError("allocation ID already exists")
            while True:
                if is_connected is not None and not is_connected():
                    raise ConnectionAbortedError(
                        "RW client disconnected during allocation retry"
                    )
                allocated, handle = self._vmm.create_tolerate_oom(
                    aligned_size, self._device
                )
                if allocated:
                    break
                if is_connected is None:
                    raise MemoryError(f"cannot allocate {aligned_size} GPU bytes")
                logger.warning(
                    "cuMemCreate OOM for aligned_size=%d; retrying in %.3fs",
                    aligned_size,
                    _ALLOCATION_RETRY_INTERVAL,
                )
                time.sleep(_ALLOCATION_RETRY_INTERVAL)
            self._allocations[allocation_id] = int(handle)

    def export(self, allocation_id: str) -> int:
        with self._lock:
            handle = self._get(allocation_id)
            return int(self._vmm.export_to_shareable_handle(handle))

    def free(self, allocation_id: str) -> None:
        with self._lock:
            handle = self._get(allocation_id)
            self._vmm.release(handle)
            del self._allocations[allocation_id]

    def clear(self) -> int:
        with self._lock:
            allocation_ids = tuple(self._allocations)
            for allocation_id in allocation_ids:
                handle = self._allocations[allocation_id]
                self._vmm.release(handle)
                del self._allocations[allocation_id]
            return len(allocation_ids)

    def _get(self, allocation_id: str) -> int:
        if not allocation_id:
            raise RuntimeError("allocation ID must not be empty")
        try:
            return self._allocations[allocation_id]
        except KeyError:
            raise RuntimeError("unknown allocation ID") from None
