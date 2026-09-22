# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""V1 allocation admission, memory ownership, and Unix RPC composition."""

from __future__ import annotations

import logging
import os
import select
import socket
import socketserver
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING
from uuid import uuid4

from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType
from gpu_memory_service.common.vmm import VMMDevice
from gpu_memory_service.v1.protocol import (
    CHECKPOINT_CONTROL_TYPES,
    REQUEST_TYPES,
    AbortRequest,
    AllocateRequest,
    AllocationRecord,
    CommitRequest,
    ErrorResponse,
    ExportRequest,
    ExportResponse,
    FreeRequest,
    HandshakeRequest,
    HandshakeResponse,
    ListAllocationsRequest,
    ListAllocationsResponse,
    Message,
    Request,
    Response,
    SuccessResponse,
    receive_message,
    send_message,
)
from gpu_memory_service.v1.server.allocations import GMSAllocationManager

if TYPE_CHECKING:
    from gpu_memory_service.v1.checkpoint import GMSCheckpointLifecycle

logger = logging.getLogger(__name__)

_CANCELLATION_POLL_SECONDS = 0.01


def _socket_is_alive(sock: socket.socket) -> bool:
    """Return whether a connected V1 lease has not reached EOF or reset."""
    try:
        fd = sock.fileno()
    except OSError:
        return False
    if fd < 0:
        return False

    flags = select.POLLERR | select.POLLHUP | select.POLLNVAL
    if hasattr(select, "POLLRDHUP"):
        flags |= select.POLLRDHUP
    poller = select.poll()
    poller.register(fd, flags)
    return not poller.poll(0)


@dataclass(eq=False)
class ServerSession:
    """Opaque token for one admitted socket session."""

    mode: GrantedLockType


@dataclass(frozen=True)
class SessionSnapshot:
    committed: bool
    rw_sessions: int
    ro_sessions: int
    waiting_writers: int
    writer_reserved: bool


class GMSSessionManager:
    """Own lock admission, writer priority, publication, and crash cleanup."""

    def __init__(
        self,
        clear_epoch: Callable[[], object],
        *,
        condition: threading.Condition | None = None,
        admission_allowed: Callable[[], bool] | None = None,
    ):
        self._clear_epoch = clear_epoch
        if (condition is None) != (admission_allowed is None):
            raise ValueError(
                "checkpoint condition and admission callback must be configured together"
            )
        self._condition = condition if condition is not None else threading.Condition()
        self._admission_allowed = (
            admission_allowed if admission_allowed is not None else lambda: True
        )
        self._rw_session: ServerSession | None = None
        self._ro_sessions: set[ServerSession] = set()
        self._writer_reserved = False
        self._waiting_writers = 0
        self._committed = False

    def acquire(
        self,
        requested: RequestedLockType,
        timeout: float | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> ServerSession | None:
        deadline = monotonic() + timeout if timeout is not None else None
        if requested is RequestedLockType.RW:
            with self._condition:
                self._require_admission()
                self._waiting_writers += 1
                try:
                    if not self._wait_for(
                        lambda: self._admission_allowed() and self._can_grant_rw(),
                        deadline,
                        is_cancelled,
                    ):
                        return None
                    self._require_admission()
                    if is_cancelled is not None and is_cancelled():
                        return None
                    self._reserve_writer()
                finally:
                    self._waiting_writers -= 1
                    self._condition.notify_all()
            return self._start_writer()

        with self._condition:
            self._require_admission()
            if requested is RequestedLockType.RO:
                if not self._wait_for(
                    lambda: self._admission_allowed() and self._can_grant_ro(),
                    deadline,
                    is_cancelled,
                ):
                    return None
                self._require_admission()
                return self._start_reader()
            if requested is not RequestedLockType.RW_OR_RO:
                raise RuntimeError(f"unsupported GMS lock type {requested.value}")
            if not self._wait_for(
                lambda: self._admission_allowed() and self._can_grant_rw_or_ro(),
                deadline,
                is_cancelled,
            ):
                return None
            self._require_admission()
            if self._can_grant_ro():
                return self._start_reader()
            if is_cancelled is not None and is_cancelled():
                return None
            self._reserve_writer()
        return self._start_writer()

    def commit(self, session: ServerSession) -> None:
        with self._condition:
            if session is not self._rw_session:
                raise RuntimeError("operation requires an RW session")
            self._rw_session = None
            session.mode = GrantedLockType.RO
            self._ro_sessions.add(session)
            self._committed = True
            self._condition.notify_all()

    def close(self, session: ServerSession) -> None:
        with self._condition:
            if session is self._rw_session:
                self._rw_session = None
                self._writer_reserved = True
                self._committed = False
            elif session in self._ro_sessions:
                self._ro_sessions.remove(session)
                self._condition.notify_all()
                return
            else:
                return

        try:
            self._clear_epoch()
        finally:
            with self._condition:
                self._writer_reserved = False
                self._condition.notify_all()

    def is_writer(self, session: ServerSession) -> bool:
        with self._condition:
            return session is self._rw_session

    def is_active(self, session: ServerSession) -> bool:
        with self._condition:
            return session is self._rw_session or session in self._ro_sessions

    def snapshot(self) -> SessionSnapshot:
        with self._condition:
            return SessionSnapshot(
                committed=self._committed,
                rw_sessions=int(self._rw_session is not None),
                ro_sessions=len(self._ro_sessions),
                waiting_writers=self._waiting_writers,
                writer_reserved=self._writer_reserved,
            )

    def _require_admission(self) -> None:
        if not self._admission_allowed():
            raise RuntimeError("GMS admission is fenced for checkpoint")

    def _can_grant_rw(self) -> bool:
        return (
            not self._writer_reserved
            and self._rw_session is None
            and not self._ro_sessions
        )

    def _can_grant_ro(self) -> bool:
        return (
            self._committed
            and not self._writer_reserved
            and self._rw_session is None
            and self._waiting_writers == 0
        )

    def _can_grant_rw_or_ro(self) -> bool:
        if self._can_grant_ro():
            return True
        return (
            not self._committed and self._waiting_writers == 0 and self._can_grant_rw()
        )

    def _reserve_writer(self) -> None:
        self._writer_reserved = True
        self._committed = False

    def _start_writer(self) -> ServerSession:
        try:
            self._clear_epoch()
        except BaseException:
            with self._condition:
                self._writer_reserved = False
                self._condition.notify_all()
            raise

        with self._condition:
            if not self._writer_reserved or self._rw_session is not None:
                raise AssertionError("GMS writer reservation was lost")
            session = ServerSession(GrantedLockType.RW)
            self._rw_session = session
            self._writer_reserved = False
            self._condition.notify_all()
            return session

    def _start_reader(self) -> ServerSession:
        session = ServerSession(GrantedLockType.RO)
        self._ro_sessions.add(session)
        return session

    def _wait_for(
        self,
        predicate: Callable[[], bool],
        deadline: float | None,
        is_cancelled: Callable[[], bool] | None,
    ) -> bool:
        while True:
            self._require_admission()
            if is_cancelled is not None and is_cancelled():
                return False
            if predicate():
                return True
            wait = None if deadline is None else deadline - monotonic()
            if wait is not None and wait <= 0:
                return False
            if is_cancelled is not None:
                wait = (
                    _CANCELLATION_POLL_SECONDS
                    if wait is None
                    else min(wait, _CANCELLATION_POLL_SECONDS)
                )
            self._condition.wait(wait)


class GMSServerMemoryManager:
    """Own identity, lock admission, and physical allocations for one socket."""

    def __init__(
        self,
        gpu_uuid: str,
        vmm: VMMDevice,
        device: int,
        *,
        checkpoint_lifecycle: GMSCheckpointLifecycle | None = None,
    ):
        if not gpu_uuid:
            raise ValueError("GPU UUID must not be empty")
        self._identity = (str(uuid4()), gpu_uuid)
        self._allocations = GMSAllocationManager(vmm, device)
        self._allocation_sizes: dict[str, int] = {}
        self._checkpoint_lifecycle = checkpoint_lifecycle
        self._sessions = GMSSessionManager(
            self._clear_allocations,
            condition=(
                checkpoint_lifecycle.condition
                if checkpoint_lifecycle is not None
                else None
            ),
            admission_allowed=(
                checkpoint_lifecycle.admission_allowed
                if checkpoint_lifecycle is not None
                else None
            ),
        )

    @property
    def identity(self) -> tuple[str, str]:
        return self._identity

    @property
    def checkpoint_lifecycle(self) -> GMSCheckpointLifecycle | None:
        return self._checkpoint_lifecycle

    def acquire(
        self,
        requested: RequestedLockType,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> ServerSession | None:
        return self._sessions.acquire(requested, is_cancelled=is_cancelled)

    def handle_request(
        self,
        session: ServerSession,
        request: Request,
        is_connected: Callable[[], bool] | None = None,
    ) -> tuple[Response, int]:
        if isinstance(request, AllocateRequest):
            self._require_rw(session)
            self._allocations.allocate(
                request.allocation_id,
                request.aligned_size,
                is_connected,
            )
            self._allocation_sizes[request.allocation_id] = request.aligned_size
            return SuccessResponse(), -1
        if isinstance(request, ExportRequest):
            self._require_active(session)
            return ExportResponse(), self._allocations.export(request.allocation_id)
        if isinstance(request, FreeRequest):
            self._require_rw(session)
            self._allocations.free(request.allocation_id)
            del self._allocation_sizes[request.allocation_id]
            return SuccessResponse(), -1
        if isinstance(request, ListAllocationsRequest):
            self._require_active(session)
            allocations = tuple(
                AllocationRecord(allocation_id, aligned_size)
                for allocation_id, aligned_size in self._allocation_sizes.items()
            )
            return ListAllocationsResponse(allocations), -1
        if isinstance(request, CommitRequest):
            self._require_rw(session)
            self._sessions.commit(session)
            return SuccessResponse(), -1
        if isinstance(request, AbortRequest):
            self._require_rw(session)
            self._sessions.close(session)
            return SuccessResponse(), -1
        raise RuntimeError(f"unsupported GMS request {type(request).__name__}")

    def close(self, session: ServerSession) -> None:
        self._sessions.close(session)

    def session_snapshot(self) -> SessionSnapshot:
        return self._sessions.snapshot()

    def allocation_snapshot(self) -> tuple[tuple[str, int], ...]:
        return tuple(sorted(self._allocation_sizes.items()))

    def _clear_allocations(self) -> int:
        cleared = self._allocations.clear()
        self._allocation_sizes.clear()
        return cleared

    def _require_rw(self, session: ServerSession) -> None:
        if not self._sessions.is_writer(session):
            raise RuntimeError("operation requires an RW session")

    def _require_active(self, session: ServerSession) -> None:
        if not self._sessions.is_active(session):
            raise RuntimeError("operation requires an active GMS session")


class _GMSRequestHandler(socketserver.BaseRequestHandler):
    server: GMSRPCServer

    def handle(self) -> None:
        session: ServerSession | None = None
        try:
            request = self._receive()
            if isinstance(request, CHECKPOINT_CONTROL_TYPES):
                lifecycle = self.server.checkpoint_lifecycle
                if lifecycle is None:
                    response = ErrorResponse("GMS checkpoint lifecycle is unavailable")
                else:
                    try:
                        response = lifecycle.handle(request)
                    except RuntimeError as exc:
                        response = ErrorResponse(str(exc))
                send_message(self.request, response)
                return
            if not isinstance(request, HandshakeRequest):
                raise RuntimeError("expected GMS handshake")  # noqa: TRY004
            manager = self.server.manager
            if (
                request.expected_identity is not None
                and request.expected_identity != manager.identity
            ):
                send_message(
                    self.request,
                    ErrorResponse("GMS server incarnation or physical GPU changed"),
                )
                return
            try:
                session = manager.acquire(
                    request.lock_type,
                    lambda: not _socket_is_alive(self.request),
                )
            except RuntimeError as exc:
                send_message(self.request, ErrorResponse(str(exc)))
                return
            if session is None:
                return
            nonce, gpu_uuid = manager.identity
            send_message(
                self.request,
                HandshakeResponse(session.mode, nonce, gpu_uuid),
            )

            while True:
                try:
                    request = self._receive()
                except EOFError as exc:
                    logger.debug("GMS client disconnected: %s", exc)
                    return
                export_fd = -1
                try:
                    if not isinstance(request, REQUEST_TYPES):
                        raise RuntimeError(  # noqa: TRY004
                            "handshake is valid only as the first message"
                        )
                    response, export_fd = manager.handle_request(
                        session,
                        request,
                        lambda: _socket_is_alive(self.request),
                    )
                except ConnectionAbortedError:
                    return
                except Exception as exc:
                    if isinstance(exc, (MemoryError, RuntimeError)):
                        logger.log(
                            logging.WARNING
                            if isinstance(exc, MemoryError)
                            else logging.DEBUG,
                            "GMS request failed: %s",
                            exc,
                        )
                    else:
                        logger.exception("Unexpected GMS request failure")
                    response = ErrorResponse(
                        str(exc),
                        out_of_memory=isinstance(exc, MemoryError),
                    )
                try:
                    send_message(self.request, response, export_fd)
                except OSError as exc:
                    logger.debug("GMS client disconnected: %s", exc)
                    return
                except Exception:
                    logger.exception("Failed to send GMS response")
                    return
                finally:
                    if export_fd >= 0:
                        os.close(export_fd)
        except (EOFError, OSError) as exc:
            logger.debug("GMS client disconnected: %s", exc)
        except Exception:
            logger.exception("Unexpected GMS connection failure")
        finally:
            if session is not None:
                self.server.manager.close(session)

    def _receive(self) -> Message:
        request, received_fd = receive_message(self.request)
        if received_fd >= 0:
            os.close(received_fd)
            raise RuntimeError("GMS clients must not send file descriptors")
        return request


class GMSRPCServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def __init__(self, path: str, manager: GMSServerMemoryManager):
        self.path = path
        self.manager = manager
        self.checkpoint_lifecycle = manager.checkpoint_lifecycle
        self._prepare_socket_path()
        previous_umask = os.umask(0o177)
        try:
            super().__init__(path, _GMSRequestHandler)
        finally:
            os.umask(previous_umask)
        try:
            os.chmod(path, 0o600)
        except BaseException:
            self.server_close()
            raise

    def _prepare_socket_path(self) -> None:
        if not os.path.exists(self.path):
            return

        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.connect(self.path)
        except OSError:
            if os.path.exists(self.path):
                os.unlink(self.path)
            return
        finally:
            probe.close()

        raise RuntimeError(f"GMS already running at {self.path}")

    def server_close(self) -> None:
        super().server_close()
        Path(self.path).unlink(missing_ok=True)
