# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Socket-session admission and allocation epoch state machine.

Connected socket is the lease:

    EMPTY ----- acquire(RW), clear epoch -----> RW
    RW -------- commit(same socket) ----------> RO
    RW -------- close/abort, clear epoch -----> EMPTY
    RO -------- acquire(RO) ------------------> RO
    RO -------- close(last reader) -----------> COMMITTED
    COMMITTED - acquire(RO) ------------------> RO
    COMMITTED - acquire(RW), clear old epoch -> RW

RO remains RO while readers remain. A waiting writer blocks late readers. Writer
reservation is a transient admission state while epoch clear occurs.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic

from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType

_CANCELLATION_POLL_SECONDS = 0.01


@dataclass(eq=False)
class ServerSession:
    """Opaque token for one admitted socket session."""

    mode: GrantedLockType


class GMSSessionManager:
    """Own lock admission, writer priority, publication, and crash cleanup."""

    def __init__(self, clear_epoch: Callable[[], object]):
        self._clear_epoch = clear_epoch
        self._condition = threading.Condition()
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
                self._waiting_writers += 1
                try:
                    if not self._wait_for(self._can_grant_rw, deadline, is_cancelled):
                        return None
                    if is_cancelled is not None and is_cancelled():
                        return None
                    self._reserve_writer()
                finally:
                    self._waiting_writers -= 1
                    self._condition.notify_all()
            return self._start_writer()

        with self._condition:
            if requested is RequestedLockType.RO:
                if not self._wait_for(self._can_grant_ro, deadline, is_cancelled):
                    return None
                return self._start_reader()
            if requested is not RequestedLockType.RW_OR_RO:
                raise RuntimeError(f"unsupported GMS lock type {requested.value}")
            if not self._wait_for(self._can_grant_rw_or_ro, deadline, is_cancelled):
                return None
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
