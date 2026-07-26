# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""V1 policies for persistent Parameters and ephemeral KV cache."""

from __future__ import annotations

import threading
from collections.abc import Callable
from uuid import uuid4

import torch
from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType
from gpu_memory_service.common.vmm import VMMDevice
from gpu_memory_service.core.client.memory_manager import (
    LocalMapping,
    install_mapping,
    reserve_and_install_mapping,
    unmap_mapping,
)
from gpu_memory_service.core.client.session import GMSClientSession
from gpu_memory_service.core.errors import FatalGMSError, GMSError

SessionFactory = Callable[
    [str, RequestedLockType, tuple[str, str] | None],
    GMSClientSession,
]


class PersistentParameterMemoryManager:
    """Own persistent, committed Parameter mappings."""

    def __init__(
        self,
        socket_path: str,
        vmm: VMMDevice,
        device: int,
        *,
        session_factory: SessionFactory = GMSClientSession,
    ):
        self.socket_path = socket_path
        self.vmm = vmm
        self.device = device
        self._session_factory = session_factory
        self._session: GMSClientSession | None = None
        self._mappings: dict[int, LocalMapping] = {}
        self._imports: dict[int, int] = {}
        self._lock = threading.RLock()
        self._fatal: FatalGMSError | None = None
        self._retired = False
        self.vmm.ensure_initialized()
        self._granularity = int(self.vmm.get_allocation_granularity(device))
        if self._granularity <= 0:
            raise ValueError("allocation granularity must be positive")

        session = self._session_factory(socket_path, RequestedLockType.RW, None)
        self._session = session
        try:
            self._server_nonce, self._gpu_uuid = session.identity
            local_gpu = self._gpu_identity()
        except Exception:
            session.close()
            self._session = None
            raise
        if local_gpu != self._gpu_uuid:
            session.close()
            self._session = None
            raise self._latch("weights sidecar is on another physical GPU")

    @property
    def mappings(self) -> tuple[LocalMapping, ...]:
        with self._lock:
            return tuple(self._mappings[base] for base in sorted(self._mappings))

    @property
    def retained_gms_allocation_count(self) -> int:
        with self._lock:
            return len(self._mappings)

    def owns(self, base: int) -> bool:
        with self._lock:
            return base in self._mappings

    def allocate(self, size: int) -> int:
        with self._lock:
            self._check_constructing()
            if size <= 0:
                raise ValueError("allocation size must be positive")
            aligned_size = self._align(size)
            allocation_id = f"allocation-{uuid4()}"
            try:
                self._require_session().allocate(allocation_id, aligned_size)
                self._select_device()
                mapping, handle = reserve_and_install_mapping(
                    self.vmm,
                    self._require_session().export(allocation_id),
                    allocation_id,
                    size,
                    aligned_size,
                    aligned_size,
                    self._granularity,
                    self.device,
                    GrantedLockType.RW,
                )
                base = mapping.base
            except Exception as cause:
                self._close_session()
                raise self._latch("Parameter allocation failed", cause) from cause
            self._mappings[base] = mapping
            self._imports[base] = handle
            return base

    def free_from_allocator(self, base: int, size: int) -> None:
        """Release one exact local segment and its uncommitted backing, if any."""
        with self._lock:
            self._check()
            mapping = self._mappings.get(base)
            if mapping is None or mapping.requested_size != size:
                self._close_session()
                raise self._latch("allocator free does not match an exact mapping")
            session = self._session
            constructing = (
                session is not None and session.lock_type is GrantedLockType.RW
            )
            if session is not None and base not in self._imports:
                self._close_session()
                raise self._latch("allocator freed a mapping without an import")
            try:
                self._select_device()
                if base in self._imports:
                    self._drop_import(mapping)
                if constructing:
                    self._require_session().free(mapping.allocation_id)
                self.vmm.address_free(mapping.base, mapping.reservation_size)
            except Exception as cause:
                self._close_session()
                raise self._latch("allocator free failed", cause) from cause
            del self._mappings[base]
            if not constructing and not self._mappings:
                self._close_session()
                self._retired = True

    def commit(self) -> None:
        """Make every mapping RO and atomically downgrade the RW socket session."""
        with self._lock:
            self._check_constructing()
            mappings = self.mappings
            if not mappings:
                self._close_session()
                raise self._latch("weights have no Parameter allocations")

            try:
                self._select_device()
                self.vmm.synchronize()
                for mapping in mappings:
                    self.vmm.set_access(
                        mapping.base,
                        mapping.aligned_size,
                        self.device,
                        GrantedLockType.RO,
                    )
                self._require_session().commit()
            except Exception as cause:
                self._close_session()
                raise self._latch("read-only Parameter commit failed", cause) from cause

    def sleep(self) -> None:
        """Drop RO imports, then close RO while preserving IDs and VAs."""
        with self._lock:
            self._check()
            if (
                self._session is None
                or self._session.lock_type is not GrantedLockType.RO
            ):
                raise GMSError("weights have not been committed")
            try:
                self._select_device()
                self.vmm.synchronize()
                for mapping in reversed(self.mappings):
                    self._drop_import(mapping)
            except Exception as cause:
                raise self._latch("Parameter sleep failed", cause) from cause
            self._close_session()

    def wake(self) -> None:
        """Acquire RO and reinstall checkpointed IDs at their exact VAs."""
        with self._lock:
            self._check()
            if self._session is not None or self._imports:
                raise GMSError("Parameter memory manager is not fully asleep")

            try:
                session = self._session_factory(
                    self.socket_path,
                    RequestedLockType.RO,
                    (self._server_nonce, self._gpu_uuid),
                )
                self._session = session
                if self._gpu_identity() != self._gpu_uuid:
                    raise GMSError("restored process is on another physical GPU")
                self._select_device()
                for mapping in self.mappings:
                    handle = install_mapping(
                        self.vmm,
                        mapping,
                        session.export(mapping.allocation_id),
                        self.device,
                        GrantedLockType.RO,
                    )
                    self._imports[mapping.base] = handle
            except Exception as cause:
                self._close_session()
                raise self._latch(f"Parameter wake failed: {cause}") from cause

    def retire(self) -> None:
        """Release local imports/reservations and the session, not sidecar backing."""
        with self._lock:
            if self._retired:
                return
            self._check()
            try:
                self._select_device()
                if self._imports:
                    self.vmm.synchronize()
                for mapping in reversed(self.mappings):
                    if mapping.base in self._imports:
                        self._drop_import(mapping)
                    self.vmm.address_free(mapping.base, mapping.reservation_size)
            except Exception as cause:
                self._close_session()
                raise self._latch("Parameter retirement failed", cause) from cause
            self._close_session()
            self._mappings.clear()
            self._retired = True

    def abort(self, cause: Exception) -> None:
        """Fail-stop model preparation and close RW to abort its epoch."""
        with self._lock:
            self._close_session()
            raise self._latch("Parameter preparation failed", cause) from cause

    def _gpu_identity(self) -> str:
        return str(torch.cuda.get_device_properties(self.device).uuid)

    def _drop_import(self, mapping: LocalMapping) -> None:
        base = mapping.base
        unmap_mapping(self.vmm, mapping, self._imports[base])
        del self._imports[base]

    def _select_device(self) -> None:
        self.vmm.runtime_set_device(self.device)

    def _close_session(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None

    def _require_session(self) -> GMSClientSession:
        if self._session is None:
            raise GMSError("weights session is disconnected")
        return self._session

    def _align(self, size: int) -> int:
        return (size + self._granularity - 1) // self._granularity * self._granularity

    def _check_constructing(self) -> None:
        self._check()
        if self._session is None or self._session.lock_type is not GrantedLockType.RW:
            raise GMSError("weights are no longer under construction")

    def _check(self) -> None:
        if self._fatal is not None:
            raise self._fatal
        if self._retired:
            raise GMSError("Parameter memory manager is retired")

    def _latch(self, message: str, cause: Exception | None = None) -> FatalGMSError:
        if self._fatal is None:
            suffix = f": {cause}" if cause is not None else ""
            self._fatal = FatalGMSError(message + suffix)
        return self._fatal


class EphemeralKVCacheMemoryManager:
    """Own one exclusive RW KV allocation epoch at preserved VAs."""

    def __init__(
        self,
        socket_path: str,
        vmm: VMMDevice,
        device: int,
        *,
        session_factory: SessionFactory = GMSClientSession,
    ):
        self.socket_path = socket_path
        self.vmm = vmm
        self.device = device
        self._session_factory = session_factory
        self._session: GMSClientSession | None = None
        self._mappings: dict[int, LocalMapping] = {}
        self._imports: dict[int, int] = {}
        self._lock = threading.RLock()
        self._fatal: FatalGMSError | None = None
        self.vmm.ensure_initialized()
        self._granularity = int(self.vmm.get_allocation_granularity(device))
        if self._granularity <= 0:
            raise ValueError("allocation granularity must be positive")

        session = self._session_factory(socket_path, RequestedLockType.RW, None)
        self._session = session
        try:
            self._server_nonce, self._gpu_uuid = session.identity
            local_gpu = self._gpu_identity()
        except Exception:
            session.close()
            self._session = None
            raise
        if local_gpu != self._gpu_uuid:
            session.close()
            self._session = None
            raise self._latch("KV cache sidecar is on another physical GPU")

    @property
    def mappings(self) -> tuple[LocalMapping, ...]:
        with self._lock:
            return tuple(self._mappings[base] for base in sorted(self._mappings))

    def owns(self, base: int) -> bool:
        with self._lock:
            return base in self._mappings

    def allocate(self, size: int) -> int:
        with self._lock:
            self._check()
            session = self._require_active_session()
            if size <= 0:
                raise ValueError("allocation size must be positive")
            aligned_size = self._align(size)
            allocation_id = f"allocation-{uuid4()}"
            try:
                session.allocate(allocation_id, aligned_size)
                self._select_device()
                mapping, handle = reserve_and_install_mapping(
                    self.vmm,
                    session.export(allocation_id),
                    allocation_id,
                    size,
                    aligned_size,
                    aligned_size,
                    self._granularity,
                    self.device,
                    GrantedLockType.RW,
                )
                base = mapping.base
            except Exception as cause:
                raise self._latch("KV cache allocation failed", cause) from cause
            self._mappings[base] = mapping
            self._imports[base] = handle
            return base

    def free_from_allocator(self, base: int, size: int) -> None:
        """Release one exact KV allocation and its VA reservation."""
        with self._lock:
            self._check()
            mapping = self._mappings.get(base)
            if mapping is None or mapping.requested_size != size:
                raise self._latch("allocator free does not match an exact KV mapping")
            try:
                self._select_device()
                if base in self._imports:
                    self._drop_import(mapping)
                if self._session is not None:
                    self._require_active_session().free(mapping.allocation_id)
                self.vmm.address_free(mapping.base, mapping.reservation_size)
            except Exception as cause:
                raise self._latch("KV allocator free failed", cause) from cause
            del self._mappings[base]

    def sleep(self) -> None:
        """Unmap all KV imports, then close RW so the server clears the epoch."""
        with self._lock:
            self._check()
            self._require_active_session()
            try:
                self._select_device()
                self.vmm.synchronize()
                for mapping in reversed(self.mappings):
                    self._drop_import(mapping)
            except Exception as cause:
                raise self._latch("KV cache sleep failed", cause) from cause
            self._close_session()

    def wake(self) -> None:
        """Acquire RW, recreate saved IDs, and map fresh backing at saved VAs."""
        with self._lock:
            self._check()
            if self._session is not None or self._imports:
                raise GMSError("KV cache memory manager is not fully asleep")

            try:
                session = self._session_factory(
                    self.socket_path,
                    RequestedLockType.RW,
                    (self._server_nonce, self._gpu_uuid),
                )
                self._session = session
                if self._gpu_identity() != self._gpu_uuid:
                    raise GMSError("restored process is on another physical GPU")
                self._select_device()
                for mapping in self.mappings:
                    session.allocate(mapping.allocation_id, mapping.aligned_size)
                    self._imports[mapping.base] = install_mapping(
                        self.vmm,
                        mapping,
                        session.export(mapping.allocation_id),
                        self.device,
                        GrantedLockType.RW,
                    )
            except Exception as cause:
                raise self._latch(f"KV cache wake failed: {cause}") from cause

    def _gpu_identity(self) -> str:
        return str(torch.cuda.get_device_properties(self.device).uuid)

    def _drop_import(self, mapping: LocalMapping) -> None:
        base = mapping.base
        unmap_mapping(self.vmm, mapping, self._imports[base])
        del self._imports[base]

    def _select_device(self) -> None:
        self.vmm.runtime_set_device(self.device)

    def _close_session(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None

    def _require_active_session(self) -> GMSClientSession:
        if self._session is None:
            raise GMSError("KV cache session is disconnected")
        if self._session.lock_type is not GrantedLockType.RW:
            raise GMSError("KV cache session is not RW")
        return self._session

    def _align(self, size: int) -> int:
        return (size + self._granularity - 1) // self._granularity * self._granularity

    def _check(self) -> None:
        if self._fatal is not None:
            raise self._fatal

    def _latch(self, message: str, cause: Exception | None = None) -> FatalGMSError:
        if self._fatal is None:
            suffix = f": {cause}" if cause is not None else ""
            self._fatal = FatalGMSError(message + suffix)
        return self._fatal
