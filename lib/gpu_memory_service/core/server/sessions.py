# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Socket-session admission and allocation epoch state machine."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from time import monotonic

from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType

from ..errors import GMSError

_CANCELLATION_POLL_SECONDS = 0.01


class ServerState(str, Enum):
    EMPTY = "EMPTY"
    RW = "RW"
    COMMITTED = "COMMITTED"
    RO = "RO"


class StateEvent(Enum):
    RW_CONNECT = auto()
    RW_COMMIT = auto()
    RW_ABORT = auto()
    RO_CONNECT = auto()
    RO_DISCONNECT = auto()


class EpochClearReason(Enum):
    START = auto()
    ABORT = auto()


@dataclass(eq=False)
class ServerSession:
    mode: GrantedLockType


@dataclass(frozen=True)
class SessionSnapshot:
    state: ServerState
    has_rw_session: bool
    ro_session_count: int
    waiting_writers: int
    committed: bool
    is_ready: bool


class GMSSessionManager:
    """Own lock admission, writer priority, publication, and crash cleanup."""

    def __init__(
        self,
        clear_epoch: Callable[[EpochClearReason, bool], None],
    ):
        self._clear_epoch = clear_epoch
        self._condition = threading.Condition()
        self._rw_session: ServerSession | None = None
        self._ro_sessions: set[ServerSession] = set()
        self._writer_reserved = False
        self._waiting_writers = 0
        self._committed = False

    @property
    def state(self) -> ServerState:
        with self._condition:
            return self._state()

    @property
    def rw_session(self) -> ServerSession | None:
        with self._condition:
            return self._rw_session

    @property
    def ro_sessions(self) -> set[ServerSession]:
        with self._condition:
            return set(self._ro_sessions)

    def snapshot(self) -> SessionSnapshot:
        with self._condition:
            has_rw_session = self._rw_session is not None or self._writer_reserved
            return SessionSnapshot(
                state=self._state(),
                has_rw_session=has_rw_session,
                ro_session_count=len(self._ro_sessions),
                waiting_writers=self._waiting_writers,
                committed=self._committed,
                is_ready=self._committed and not has_rw_session,
            )

    def acquire(
        self,
        requested: RequestedLockType,
        timeout: float | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> ServerSession | None:
        deadline = monotonic() + timeout if timeout is not None else None
        if requested == RequestedLockType.RW:
            with self._condition:
                self._waiting_writers += 1
                try:
                    if not self._wait_for(
                        self._can_grant_rw,
                        deadline,
                        is_cancelled,
                    ):
                        return None
                    if is_cancelled is not None and is_cancelled():
                        return None
                    replacing_committed = self._reserve_writer()
                finally:
                    self._waiting_writers -= 1
                    self._condition.notify_all()
            return self._start_writer(replacing_committed)

        with self._condition:
            if requested == RequestedLockType.RO:
                if not self._wait_for(
                    self._can_grant_ro,
                    deadline,
                    is_cancelled,
                ):
                    return None
                return self._start_reader()

            if requested == RequestedLockType.RW_OR_RO:
                if not self._wait_for(
                    self._can_grant_rw_or_ro,
                    deadline,
                    is_cancelled,
                ):
                    return None
                if self._can_grant_ro():
                    return self._start_reader()
                if is_cancelled is not None and is_cancelled():
                    return None
                replacing_committed = self._reserve_writer()
            else:
                raise GMSError(f"unsupported GMS lock type {requested.value}")

        return self._start_writer(replacing_committed)

    def wake_waiters(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def commit(self, session: ServerSession) -> None:
        with self._condition:
            if session is not self._rw_session or session.mode != GrantedLockType.RW:
                raise GMSError("session no longer owns the RW lock")
            self._rw_session = None
            session.mode = GrantedLockType.RO
            self._ro_sessions.add(session)
            self._committed = True
            self._condition.notify_all()

    def close(self, session: ServerSession) -> StateEvent | None:
        with self._condition:
            if session is self._rw_session:
                self._clear_epoch(EpochClearReason.ABORT, False)
                self._rw_session = None
                self._committed = False
                event = StateEvent.RW_ABORT
            elif session in self._ro_sessions:
                self._ro_sessions.remove(session)
                event = StateEvent.RO_DISCONNECT
            else:
                return None
            self._condition.notify_all()
            return event

    def _state(self) -> ServerState:
        if self._rw_session is not None or self._writer_reserved:
            return ServerState.RW
        if self._ro_sessions:
            return ServerState.RO
        if self._committed:
            return ServerState.COMMITTED
        return ServerState.EMPTY

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

    def _reserve_writer(self) -> bool:
        replacing_committed = self._committed
        self._writer_reserved = True
        self._committed = False
        return replacing_committed

    def _start_writer(self, replacing_committed: bool) -> ServerSession:
        try:
            self._clear_epoch(EpochClearReason.START, replacing_committed)
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
            timeout = None if deadline is None else deadline - monotonic()
            if timeout is not None and timeout <= 0:
                return False
            if is_cancelled is not None:
                timeout = (
                    _CANCELLATION_POLL_SECONDS
                    if timeout is None
                    else min(timeout, _CANCELLATION_POLL_SECONDS)
                )
            self._condition.wait(timeout)
