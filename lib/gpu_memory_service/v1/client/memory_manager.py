# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VA-stable client memory ownership for one GMS V1 socket domain.

The client backs allocations with server-owned slabs, mapped back to back in
one virtual-address reservation per domain, so an allocation can span
adjacent slabs and no slab tail is left unusable.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from uuid import uuid4

from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType
from gpu_memory_service.common.vmm import VMMDevice
from gpu_memory_service.v1 import device as device_identity
from gpu_memory_service.v1.client.mapping import (
    LocalMapping,
    install_mapping,
    unmap_mapping,
)
from gpu_memory_service.v1.client.session import _GMSClientSession

logger = logging.getLogger(__name__)

DEFAULT_SLAB_SIZE = 2 * 1024 * 1024 * 1024

_SessionFactory = Callable[
    [str, RequestedLockType],
    _GMSClientSession,
]


@dataclass(frozen=True)
class _InstalledMapping(LocalMapping):
    """One local reservation with its currently imported CUDA handle."""

    handle: int


class GMSClientMemoryManager:
    """Own one GMS socket session, allocation table, and local VA table."""

    def __init__(
        self,
        socket_path: str,
        vmm: VMMDevice,
        device: int,
        *,
        session_factory: _SessionFactory = _GMSClientSession,
        slab_size: int = DEFAULT_SLAB_SIZE,
    ):
        self._socket_path = socket_path
        self._vmm = vmm
        self._device = device
        self._session_factory = session_factory
        self._session: _GMSClientSession | None = None
        self._mappings: dict[int, _InstalledMapping] = {}
        self._regions: dict[int, int] = {}
        # Free (va, length) runs over mapped slabs, sorted and coalesced.
        self._free: list[tuple[int, int]] = []
        # The domain's VA reservation, (base, size), while it has slabs.
        self._arena: tuple[int, int] | None = None
        self._lock = threading.RLock()
        self._failure: str | None = None
        self._vmm.ensure_initialized()
        self._granularity = int(self._vmm.get_allocation_granularity(device))
        if self._granularity <= 0:
            raise ValueError("allocation granularity must be positive")
        self._slab_size = self._align(slab_size)
        if self._slab_size <= 0:
            raise ValueError("slab size must be positive")

    @property
    def mappings(self) -> tuple[LocalMapping, ...]:
        with self._lock:
            return self._ordered_mappings()

    def owns(self, va: int) -> bool:
        with self._lock:
            return va in self._regions

    def connect(self, lock_type: RequestedLockType) -> None:
        with self._lock:
            self._check()
            if self._session is not None:
                raise RuntimeError("GMS memory manager is already connected")
            try:
                device_identity.invalidate_device_uuid_cache()
                device_uuid = device_identity.get_device_uuid(self._device)
                session = self._session_factory(self._socket_path, lock_type)
                if session.identity[1] != device_uuid:
                    try:
                        session.close()
                    except Exception:
                        logger.exception("GMS close failed after identity mismatch")
                    raise RuntimeError("GMS sidecar is on another physical GPU")
                self._session = session
            except Exception as exc:
                raise self._latch("GMS connect failed", exc) from exc

    def create_mapping(self, size: int) -> int:
        with self._lock:
            self._check()
            self._require_rw()
            if size <= 0:
                raise ValueError("allocation size must be positive")
            if any(not mapping.handle for mapping in self._ordered_mappings()):
                raise RuntimeError(
                    "cannot create a mapping while GMS slabs are unmapped"
                )
            aligned_size = self._align(size)
            try:
                va = self._carve(size, aligned_size)
                if va is None:
                    self._add_slab(aligned_size)
                    va = self._carve(size, aligned_size)
                if va is None:
                    raise RuntimeError("new GMS slab has no room for the allocation")
                return va
            except Exception as exc:
                raise self._latch("GMS mapping creation failed", exc) from exc

    def destroy_mapping(self, va: int, size: int | None = None) -> None:
        with self._lock:
            self._check()
            try:
                requested_size = self._regions[va]
            except KeyError:
                raise RuntimeError(f"GMS does not own VA 0x{va:x}") from None
            if size is not None and size != requested_size:
                raise RuntimeError("allocator free does not match the GMS mapping")
            try:
                del self._regions[va]
                self._put_free(va, self._align(requested_size))
                for mapping in self._ordered_mappings():
                    if self._is_free(mapping.base, mapping.aligned_size):
                        self._destroy_slab(mapping)
            except Exception as exc:
                raise self._latch("GMS mapping destruction failed", exc) from exc

    def commit(self) -> None:
        """Publish current mappings and downgrade the same socket from RW to RO."""
        with self._lock:
            self._check()
            session = self._require_rw()
            if not self._mappings:
                raise RuntimeError("cannot commit an empty GMS allocation set")
            try:
                self._select_device()
                self._vmm.synchronize()
                for mapping in self._ordered_mappings():
                    self._vmm.set_access(
                        mapping.base,
                        mapping.aligned_size,
                        self._device,
                        GrantedLockType.RO,
                    )
                session.commit()
            except Exception as exc:
                raise self._latch("GMS commit failed", exc) from exc

    def unmap_all_vas(self) -> None:
        """Drop imported handles while preserving allocation records and VAs."""
        with self._lock:
            self._check()
            try:
                self._select_device()
                self._vmm.synchronize()
                for mapping in reversed(self._ordered_mappings()):
                    if mapping.handle:
                        self._unmap(mapping)
            except Exception as exc:
                raise self._latch("GMS unmap failed", exc) from exc

    def reallocate_all_handles(self) -> None:
        """Create fresh server backing under every saved allocation ID."""
        with self._lock:
            self._check()
            session = self._require_rw()
            try:
                for mapping in self._ordered_mappings():
                    session.allocate(mapping.allocation_id, mapping.aligned_size)
            except Exception as exc:
                raise self._latch("GMS backing reallocation failed", exc) from exc

    def remap_all_vas(self) -> None:
        """Install saved allocation IDs at their existing VAs."""
        with self._lock:
            self._check()
            session = self._require_session()
            try:
                self._select_device()
                for mapping in self._ordered_mappings():
                    if mapping.handle:
                        raise RuntimeError("GMS mapping is already installed")
                    handle = install_mapping(
                        self._vmm,
                        mapping,
                        session.export(mapping.allocation_id),
                        self._device,
                        session.lock_type,
                    )
                    self._mappings[mapping.base] = replace(mapping, handle=handle)
            except Exception as exc:
                raise self._latch("GMS remap failed", exc) from exc

    def disconnect(self) -> None:
        with self._lock:
            session = self._session
            self._session = None
            if session is not None:
                session.close()

    def close(self) -> None:
        """Release local mappings and VAs, then disconnect the socket lease."""
        with self._lock:
            self._check()
            for va in reversed(sorted(self._regions)):
                self.destroy_mapping(va)
            self.disconnect()

    def _ordered_mappings(self) -> tuple[_InstalledMapping, ...]:
        return tuple(self._mappings[va] for va in sorted(self._mappings))

    def _unmap(self, mapping: _InstalledMapping) -> None:
        unmap_mapping(self._vmm, mapping, mapping.handle)
        self._mappings[mapping.base] = replace(mapping, handle=0)

    def _select_device(self) -> None:
        self._vmm.runtime_set_device(self._device)

    def _require_session(self) -> _GMSClientSession:
        if self._session is None:
            raise RuntimeError("GMS memory manager is disconnected")
        return self._session

    def _require_rw(self) -> _GMSClientSession:
        session = self._require_session()
        if session.lock_type is not GrantedLockType.RW:
            raise RuntimeError("operation requires an RW session")
        return session

    def _align(self, size: int) -> int:
        return (size + self._granularity - 1) // self._granularity * self._granularity

    def _add_slab(self, aligned_size: int) -> _InstalledMapping:
        if self._arena is None:
            _, total = self._vmm.device_memory_info(self._device)
            # Room for every slab the GPU can back, plus gaps that freed
            # slabs leave between the ones still in use.
            arena_size = self._align(2 * total)
            self._arena = (
                int(self._vmm.address_reserve(arena_size, self._granularity)),
                arena_size,
            )
        base, slab_bytes = self._place_slab(aligned_size)
        session = self._require_rw()
        allocation_id = f"allocation-{uuid4()}"
        session.allocate(allocation_id, slab_bytes)
        self._select_device()
        mapping = _InstalledMapping(
            allocation_id, slab_bytes, slab_bytes, base, slab_bytes, handle=0
        )
        handle = install_mapping(
            self._vmm,
            mapping,
            session.export(allocation_id),
            self._device,
            GrantedLockType.RW,
        )
        installed = replace(mapping, handle=handle)
        self._mappings[base] = installed
        self._put_free(base, slab_bytes)
        return installed

    def _place_slab(self, aligned_size: int) -> tuple[int, int]:
        """Find VA for a new slab that, with any free run ending right before
        it, can hold ``aligned_size``. Returns the slab's base and size."""
        arena_base, arena_size = self._arena
        gaps, cursor = [], arena_base
        for mapping in self._ordered_mappings():
            gaps.append((cursor, mapping.base - cursor))
            cursor = mapping.base + mapping.aligned_size
        gaps.append((cursor, arena_base + arena_size - cursor))
        free_ends = {va + length: length for va, length in self._free}
        for gap_base, gap_length in gaps:
            needed = self._align(aligned_size - free_ends.get(gap_base, 0))
            slab_bytes = max(self._slab_size, needed)
            if slab_bytes <= gap_length:
                return gap_base, slab_bytes
        raise RuntimeError("GMS client VA reservation is exhausted")

    def _destroy_slab(self, mapping: _InstalledMapping) -> None:
        self._select_device()
        if mapping.handle:
            self._unmap(mapping)
        if self._session is not None and self._session.lock_type is GrantedLockType.RW:
            self._session.free(mapping.allocation_id)
        del self._mappings[mapping.base]
        self._take_free_range(mapping.base, mapping.aligned_size)
        if not self._mappings:
            self._vmm.address_free(*self._arena)
            self._arena = None

    def _carve(self, size: int, aligned_size: int) -> int | None:
        for index, (va, length) in enumerate(self._free):
            if length < aligned_size:
                continue
            if length > aligned_size:
                self._free[index] = (va + aligned_size, length - aligned_size)
            else:
                del self._free[index]
            self._regions[va] = size
            return va
        return None

    def _is_free(self, va: int, length: int) -> bool:
        return any(
            run_va <= va and va + length <= run_va + run_length
            for run_va, run_length in self._free
        )

    def _take_free_range(self, va: int, length: int) -> None:
        """Drop [va, va + length), all free, from the free runs."""
        runs = []
        for run_va, run_length in self._free:
            run_end = run_va + run_length
            if run_end <= va or va + length <= run_va:
                runs.append((run_va, run_length))
                continue
            if run_va < va:
                runs.append((run_va, va - run_va))
            if va + length < run_end:
                runs.append((va + length, run_end - va - length))
        self._free = runs

    def _put_free(self, va: int, length: int) -> None:
        runs = sorted([*self._free, (va, length)])
        merged: list[tuple[int, int]] = [runs[0]]
        for run_va, run_length in runs[1:]:
            prev_va, prev_length = merged[-1]
            prev_end = prev_va + prev_length
            if run_va <= prev_end:
                end = max(prev_end, run_va + run_length)
                merged[-1] = (prev_va, end - prev_va)
            else:
                merged.append((run_va, run_length))
        self._free = merged

    def _check(self) -> None:
        if self._failure is not None:
            raise RuntimeError(self._failure)

    def _latch(self, message: str, cause: Exception) -> RuntimeError:
        if self._failure is None:
            self._failure = f"{message}: {cause}"
        session = self._session
        self._session = None
        if session is not None:
            try:
                session.close()
            except Exception:
                logger.exception("GMS disconnect failed after operational failure")
        return RuntimeError(self._failure)
