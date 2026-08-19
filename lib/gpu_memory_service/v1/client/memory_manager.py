# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VA-stable client memory ownership for one GMS V1 socket domain."""

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
    reserve_and_install_mapping,
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
        self._regions: dict[int, tuple[int, int]] = {}
        self._free: dict[int, list[tuple[int, int]]] = {}
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
            aligned_size = self._align(size)
            try:
                for mapping in self._ordered_mappings():
                    offset = self._take_free(mapping.base, aligned_size)
                    if offset is None:
                        continue
                    va = mapping.base + offset
                    self._regions[va] = (mapping.base, size)
                    return va
                return self._add_slab(size, aligned_size)
            except Exception as exc:
                raise self._latch("GMS mapping creation failed", exc) from exc

    def destroy_mapping(self, va: int, size: int | None = None) -> None:
        with self._lock:
            self._check()
            try:
                slab_base, requested_size = self._regions[va]
            except KeyError:
                raise RuntimeError(f"GMS does not own VA 0x{va:x}") from None
            if size is not None and size != requested_size:
                raise RuntimeError("allocator free does not match the GMS mapping")
            try:
                mapping = self._mappings[slab_base]
                del self._regions[va]
                self._put_free(slab_base, va - slab_base, self._align(requested_size))
                if self._free[slab_base] == [(0, mapping.aligned_size)]:
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

    def _add_slab(self, size: int, aligned_size: int) -> int:
        slab_bytes = max(self._slab_size, aligned_size)
        session = self._require_rw()
        allocation_id = f"allocation-{uuid4()}"
        session.allocate(allocation_id, slab_bytes)
        self._select_device()
        mapping, handle = reserve_and_install_mapping(
            self._vmm,
            session.export(allocation_id),
            allocation_id,
            slab_bytes,
            slab_bytes,
            slab_bytes,
            self._granularity,
            self._device,
            GrantedLockType.RW,
        )
        installed = _InstalledMapping(
            mapping.allocation_id,
            mapping.requested_size,
            mapping.aligned_size,
            mapping.base,
            mapping.reservation_size,
            handle,
        )
        self._mappings[mapping.base] = installed
        leftover = slab_bytes - aligned_size
        self._free[mapping.base] = [(aligned_size, leftover)] if leftover else []
        self._regions[mapping.base] = (mapping.base, size)
        return mapping.base

    def _destroy_slab(self, mapping: _InstalledMapping) -> None:
        self._select_device()
        if mapping.handle:
            self._unmap(mapping)
        if self._session is not None and self._session.lock_type is GrantedLockType.RW:
            self._session.free(mapping.allocation_id)
        self._vmm.address_free(mapping.base, mapping.reservation_size)
        del self._mappings[mapping.base]
        del self._free[mapping.base]

    def _take_free(self, slab_base: int, aligned_size: int) -> int | None:
        holes = self._free[slab_base]
        for index, (offset, length) in enumerate(holes):
            if length < aligned_size:
                continue
            leftover = length - aligned_size
            if leftover:
                holes[index] = (offset + aligned_size, leftover)
            else:
                del holes[index]
            return offset
        return None

    def _put_free(self, slab_base: int, offset: int, length: int) -> None:
        holes = self._free[slab_base]
        index = 0
        while index < len(holes) and holes[index][0] < offset:
            index += 1
        holes.insert(index, (offset, length))
        merged: list[tuple[int, int]] = []
        for hole_offset, hole_length in holes:
            if merged and hole_offset <= merged[-1][0] + merged[-1][1]:
                prev_offset, prev_length = merged[-1]
                merged[-1] = (
                    prev_offset,
                    max(prev_offset + prev_length, hole_offset + hole_length)
                    - prev_offset,
                )
                continue
            merged.append((hole_offset, hole_length))
        self._free[slab_base] = merged

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
