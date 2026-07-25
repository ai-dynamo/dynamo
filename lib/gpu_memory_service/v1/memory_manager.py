# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic client lifecycle for Snapshot-restored GMS mappings."""

from __future__ import annotations

import threading
from collections.abc import Callable
from uuid import uuid4

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


class SnapshotMemoryManager:
    """Apply the Snapshot profile lifecycle to neutral local mappings."""

    def __init__(
        self,
        socket_path: str,
        vmm: VMMDevice,
        device: int,
        *,
        session_factory: Callable[
            [str, RequestedLockType, tuple[str, str] | None],
            GMSClientSession,
        ] = GMSClientSession,
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
            raise self._latch("GMS sidecar is on another physical GPU")

    @property
    def mappings(self) -> tuple[LocalMapping, ...]:
        with self._lock:
            return tuple(self._mappings[base] for base in sorted(self._mappings))

    @property
    def retained_gms_allocation_count(self) -> int:
        with self._lock:
            return len(self._mappings)

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
                raise self._latch("snapshot allocation failed", cause) from cause
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
                    assert session is not None
                    session.free(mapping.allocation_id)
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
                raise self._latch("snapshot has no parameter allocations")

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
                raise self._latch("read-only commit failed", cause) from cause

    def sleep(self) -> None:
        """Drop RO imports, then close the RO socket while preserving IDs and VAs."""
        with self._lock:
            self._check()
            if (
                self._session is None
                or self._session.lock_type is not GrantedLockType.RO
            ):
                raise GMSError("snapshot weights have not been committed")
            try:
                self._select_device()
                self.vmm.synchronize()
                for mapping in reversed(self.mappings):
                    self._drop_import(mapping)
            except Exception as cause:
                self._close_session()
                raise self._latch("snapshot sleep failed", cause) from cause
            self._close_session()

    def wake(self) -> None:
        """Acquire RO, verify identity, and reinstall checkpointed IDs at exact VAs."""
        with self._lock:
            self._check()
            if self._session is not None or self._imports:
                raise GMSError("snapshot memory manager is not fully asleep")

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
                raise self._latch(f"snapshot wake failed: {cause}") from cause

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
                raise self._latch("snapshot retirement failed", cause) from cause
            self._close_session()
            self._mappings.clear()
            self._retired = True

    def abort(self, cause: Exception) -> None:
        """Fail-stop model preparation and close the RW socket to abort its epoch."""
        with self._lock:
            self._close_session()
            raise self._latch("snapshot model preparation failed", cause) from cause

    def _gpu_identity(self) -> str:
        import torch

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
            raise GMSError("GMS session is disconnected")
        return self._session

    def _align(self, size: int) -> int:
        return (size + self._granularity - 1) // self._granularity * self._granularity

    def _check_constructing(self) -> None:
        self._check()
        if self._session is None or self._session.lock_type is not GrantedLockType.RW:
            raise GMSError("snapshot weights are no longer under construction")

    def _check(self) -> None:
        if self._fatal is not None:
            raise self._fatal
        if self._retired:
            raise GMSError("Snapshot memory manager is retired")

    def _latch(self, message: str, cause: Exception | None = None) -> FatalGMSError:
        if self._fatal is None:
            suffix = f": {cause}" if cause is not None else ""
            self._fatal = FatalGMSError(message + suffix)
        return self._fatal
