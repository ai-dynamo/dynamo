# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client-local CUDA VMM memory manager.

Used for tags whose physical memory is private to one engine — primarily
kv_cache, both for shadow-failover engines (aliased=True) and plain
enable_sleep_mode workers (aliased=False). No GMS server, no IPC, no
cross-engine sharing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional

from gpu_memory_service.common.cuda_utils import (
    align_to_granularity,
    cuda_ensure_initialized,
    cuda_synchronize,
    cumem_address_free,
    cumem_address_reserve,
    cumem_create_tolerate_oom,
    cumem_get_allocation_granularity,
    cumem_map,
    cumem_release,
    cumem_set_access,
    cumem_unmap,
)
from gpu_memory_service.common.locks import GrantedLockType

logger = logging.getLogger(__name__)


@dataclass
class _Entry:
    base_va: int
    aligned_size: int
    tag: str
    handle: int = 0  # 0 = unmapped (VA preserved); nonzero = current physical


class ClientLocalMemoryManager:
    """Client-local CUDA VMM allocator. No GMS server, no IPC, no
    cross-engine sharing. Used for tags whose physical memory is private
    to one engine — notably kv_cache (shadow failover and plain
    enable_sleep_mode).

    Modes (set once at construction):
      aliased=False  cuMemCreate(aligned_size) + one map per tensor.
      aliased=True   one granularity-sized handle aliased across the full
                     VA range. Footprint ~granularity per tensor instead
                     of full size; collapses to real on first remap_all_vas
                     and never returns.

    Lifecycle (matches the GMSClientMemoryManager interface):
      create_mapping   reserve VA, install physical (aliased or real)
      unmap_all_vas    sleep — release physical, preserve VAs
      remap_all_vas    wake — cuMemCreate(aligned_size), single map; always real
      destroy_mapping  single-entry teardown

    Physical does NOT survive a sleep — the client holds the only
    cuMemCreate ref; unmap drops it, wake allocates fresh. Matches vLLM's
    sleep contract; contrast GMSClientMemoryManager (server holds the ref,
    data preserved across client sleep/wake).
    """

    def __init__(
        self,
        device: int = 0,
        *,
        aliased: bool = False,
        tag: Optional[str] = None,
        retry_interval: float = 0.5,
        retry_timeout: Optional[float] = 30.0,
    ) -> None:
        cuda_ensure_initialized()
        self.device = device
        self.tag = tag
        self.aliased = aliased
        self._retry_interval = retry_interval
        self._retry_timeout = retry_timeout
        self.granularity = cumem_get_allocation_granularity(device)
        self._mappings: Dict[int, _Entry] = {}
        self._unmapped = False

    # ==================== Properties ====================

    @property
    def is_unmapped(self) -> bool:
        return self._unmapped

    @property
    def mappings(self) -> Dict[int, _Entry]:
        return self._mappings

    @property
    def is_connected(self) -> bool:
        # Symmetry with GMSClientMemoryManager. Always True — no IPC to
        # disconnect.
        return True

    # ==================== Interface (matches GMSClientMemoryManager) ====================

    def create_mapping(self, size: int, tag: str = "kv_cache") -> int:
        """Allocate a new mapping. Behavior depends on construction-time
        `aliased` flag. Returns base_va.
        """
        if self._unmapped:
            raise RuntimeError(
                "create_mapping called while unmapped; call remap_all_vas first"
            )
        aligned_size = align_to_granularity(size, self.granularity)
        va = cumem_address_reserve(aligned_size, self.granularity)
        if self.aliased:
            handle = self._install_aliased(va, aligned_size)
        else:
            handle = self._install_real(va, aligned_size)
        self._mappings[va] = _Entry(
            base_va=va, aligned_size=aligned_size, tag=tag, handle=handle
        )
        return va

    def destroy_mapping(self, va: int) -> bool:
        """Tear down a single entry. Returns True if owned, False otherwise."""
        entry = self._mappings.pop(va, None)
        if entry is None:
            return False
        cuda_synchronize()
        if entry.handle != 0:
            cumem_unmap(entry.base_va, entry.aligned_size)
            cumem_release(entry.handle)
        cumem_address_free(entry.base_va, entry.aligned_size)
        return True

    def unmap_all_vas(self) -> None:
        """Sleep: release physical for all entries; preserve VA reservations.
        After: is_unmapped=True. Idempotent.
        """
        cuda_synchronize()
        unmapped_count = 0
        total_bytes = 0
        for entry in self._mappings.values():
            if entry.handle == 0:
                continue
            cumem_unmap(entry.base_va, entry.aligned_size)
            cumem_release(entry.handle)
            entry.handle = 0
            unmapped_count += 1
            total_bytes += entry.aligned_size
        self._unmapped = True
        logger.info(
            "[GMS] Unmapped %d client-local mappings (%.2f GiB), "
            "preserving %d VA reservations",
            unmapped_count,
            total_bytes / (1 << 30),
            len(self._mappings),
        )

    def remap_all_vas(self) -> None:
        """Wake: allocate fresh REAL per-tensor backing at every preserved VA.
        Aliased-mode entries collapse to real here; subsequent cycles look
        identical to a non-aliased manager.
        """
        if not self._unmapped:
            raise RuntimeError("remap_all_vas requires prior unmap_all_vas")
        cuda_synchronize()
        remapped_count = 0
        total_bytes = 0
        for entry in self._mappings.values():
            if entry.handle != 0:
                continue
            handle = self._cumem_create_with_retry(entry.aligned_size)
            cumem_map(entry.base_va, entry.aligned_size, handle)
            cumem_set_access(
                entry.base_va, entry.aligned_size, self.device, GrantedLockType.RW
            )
            entry.handle = handle
            remapped_count += 1
            total_bytes += entry.aligned_size
        self._unmapped = False
        logger.info(
            "[GMS] Remapped %d client-local mappings (%.2f GiB) with real backing",
            remapped_count,
            total_bytes / (1 << 30),
        )

    def abort(self) -> None:
        """No-op for the client-local manager (no IPC to drop). Provided for
        shape parity with GMSClientMemoryManager so callers can treat the
        two uniformly.
        """
        return

    def close(self) -> None:
        """Final teardown. Frees physical, releases handles, frees VAs."""
        cuda_synchronize()
        for va in list(self._mappings.keys()):
            self.destroy_mapping(va)

    def __enter__(self) -> "ClientLocalMemoryManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ==================== Private helpers ====================

    def _install_aliased(self, va: int, aligned_size: int) -> int:
        """Allocate one granularity-sized scratch chunk, alias it across the
        full VA range (N cuMemMap calls).
        """
        handle = self._cumem_create_with_retry(self.granularity)
        for offset in range(0, aligned_size, self.granularity):
            cumem_map(va + offset, self.granularity, handle)
        cumem_set_access(va, aligned_size, self.device, GrantedLockType.RW)
        return handle

    def _install_real(self, va: int, aligned_size: int) -> int:
        """Allocate full-size physical and single-map at va."""
        handle = self._cumem_create_with_retry(aligned_size)
        cumem_map(va, aligned_size, handle)
        cumem_set_access(va, aligned_size, self.device, GrantedLockType.RW)
        return handle

    def _cumem_create_with_retry(self, size: int) -> int:
        """cuMemCreate with OOM retry. Mirrors the server's policy in
        server/allocations.py:allocate(). Logs on first OOM, raises
        TimeoutError if total wait exceeds retry_timeout.
        """
        started_at = time.monotonic()
        reported = False
        while True:
            ok, handle = cumem_create_tolerate_oom(size, self.device)
            if ok:
                return handle
            if not reported:
                logger.warning(
                    "[GMS] cuMemCreate OOM (size=%d, device=%d) — retrying every %.1fs",
                    size,
                    self.device,
                    self._retry_interval,
                )
                reported = True
            if (
                self._retry_timeout is not None
                and time.monotonic() - started_at >= self._retry_timeout
            ):
                raise TimeoutError(
                    f"cuMemCreate OOM after {self._retry_timeout}s "
                    f"for size={size} on device={self.device}"
                )
            time.sleep(self._retry_interval)
