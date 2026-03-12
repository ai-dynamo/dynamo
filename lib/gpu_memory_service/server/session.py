# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Server-side connection, FSM, and waiter state."""

from __future__ import annotations

import asyncio
import socket
from dataclasses import dataclass, field
from typing import Optional, Set

from gpu_memory_service.common.types import (
    RO_ALLOWED,
    RW_ALLOWED,
    RW_REQUIRED,
    GrantedLockType,
    RequestedLockType,
    ServerState,
    StateEvent,
)


@dataclass(eq=False)
class Connection:
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    mode: GrantedLockType
    session_id: str
    recv_buffer: bytearray = field(default_factory=bytearray)

    def __hash__(self) -> int:
        return hash(self.session_id)

    @property
    def raw_socket(self) -> socket.socket:
        return self.writer.get_extra_info("socket")

    async def close(self) -> None:
        self.writer.close()
        try:
            await self.writer.wait_closed()
        except Exception:
            pass


class InvalidTransition(Exception):
    """Raised when an invalid state transition is attempted."""


class OperationNotAllowed(Exception):
    """Raised when an operation is not allowed in the current state/mode."""


@dataclass(frozen=True)
class Transition:
    from_states: frozenset[ServerState]
    event: StateEvent
    to_state: Optional[ServerState]
    condition: Optional[str] = None


TRANSITIONS: list[Transition] = [
    Transition(
        from_states=frozenset({ServerState.EMPTY, ServerState.COMMITTED}),
        event=StateEvent.RW_CONNECT,
        to_state=ServerState.RW,
    ),
    Transition(
        from_states=frozenset({ServerState.RW}),
        event=StateEvent.RW_COMMIT,
        to_state=ServerState.COMMITTED,
    ),
    Transition(
        from_states=frozenset({ServerState.RW}),
        event=StateEvent.RW_ABORT,
        to_state=ServerState.EMPTY,
    ),
    Transition(
        from_states=frozenset({ServerState.COMMITTED, ServerState.RO}),
        event=StateEvent.RO_CONNECT,
        to_state=ServerState.RO,
    ),
    Transition(
        from_states=frozenset({ServerState.RO}),
        event=StateEvent.RO_DISCONNECT,
        to_state=ServerState.RO,
        condition="has_remaining_readers",
    ),
    Transition(
        from_states=frozenset({ServerState.RO}),
        event=StateEvent.RO_DISCONNECT,
        to_state=ServerState.COMMITTED,
        condition="is_last_reader",
    ),
]


class GMSLocalFSM:
    """Explicit connection/lock state machine."""

    def __init__(self):
        self._rw_conn: Optional[Connection] = None
        self._ro_conns: Set[Connection] = set()
        self._committed = False

    @property
    def state(self) -> ServerState:
        if self._rw_conn is not None:
            return ServerState.RW
        if self._ro_conns:
            return ServerState.RO
        if self._committed:
            return ServerState.COMMITTED
        return ServerState.EMPTY

    @property
    def rw_conn(self) -> Optional[Connection]:
        return self._rw_conn

    @property
    def ro_conns(self) -> Set[Connection]:
        return self._ro_conns

    @property
    def ro_count(self) -> int:
        return len(self._ro_conns)

    @property
    def committed(self) -> bool:
        return self._committed

    def _has_remaining_readers(self, conn: Connection) -> bool:
        return len(self._ro_conns) > 1 or conn not in self._ro_conns

    def _is_last_reader(self, conn: Connection) -> bool:
        return len(self._ro_conns) == 1 and conn in self._ro_conns

    def _check_condition(self, condition: Optional[str], conn: Connection) -> bool:
        if condition is None:
            return True
        if condition == "has_remaining_readers":
            return self._has_remaining_readers(conn)
        if condition == "is_last_reader":
            return self._is_last_reader(conn)
        raise ValueError(f"Unknown condition: {condition}")

    def _find_transition(
        self,
        from_state: ServerState,
        event: StateEvent,
        conn: Connection,
    ) -> Optional[Transition]:
        for transition in TRANSITIONS:
            if from_state not in transition.from_states:
                continue
            if transition.event != event:
                continue
            if not self._check_condition(transition.condition, conn):
                continue
            return transition
        return None

    def _apply_event(self, event: StateEvent, conn: Connection) -> None:
        if event == StateEvent.RW_CONNECT:
            self._rw_conn = conn
            self._committed = False
        elif event == StateEvent.RW_COMMIT:
            self._committed = True
            self._rw_conn = None
        elif event == StateEvent.RW_ABORT:
            self._rw_conn = None
        elif event == StateEvent.RO_CONNECT:
            self._ro_conns.add(conn)
        elif event == StateEvent.RO_DISCONNECT:
            self._ro_conns.discard(conn)

    def transition(self, event: StateEvent, conn: Connection) -> ServerState:
        transition = self._find_transition(self.state, event, conn)
        if transition is None:
            raise InvalidTransition(
                f"No transition for {event.name} from state {self.state.name} "
                f"(session={conn.session_id})"
            )
        self._apply_event(event, conn)
        return self.state

    def check_operation(self, msg_type: type, conn: Connection) -> None:
        if conn.mode == GrantedLockType.RW and msg_type not in RW_ALLOWED:
            raise OperationNotAllowed(
                f"{msg_type.__name__} not allowed for RW session in state {self.state.name}"
            )
        if conn.mode == GrantedLockType.RO and msg_type not in RO_ALLOWED:
            raise OperationNotAllowed(
                f"{msg_type.__name__} not allowed for RO session in state {self.state.name}"
            )
        if msg_type in RW_REQUIRED and conn.mode != GrantedLockType.RW:
            raise OperationNotAllowed(
                f"{msg_type.__name__} requires RW session, got {conn.mode.value}"
            )

    def can_acquire_rw(self) -> bool:
        return self._rw_conn is None and not self._ro_conns

    def can_acquire_ro(self, waiting_writers: int) -> bool:
        return self._committed and self._rw_conn is None and waiting_writers == 0


@dataclass(frozen=True)
class SessionSnapshot:
    state: ServerState
    has_rw_session: bool
    ro_session_count: int
    waiting_writers: int
    committed: bool
    is_ready: bool


class GMSSessionManager:
    """Owns lock transitions, waiter coordination, and cleanup."""

    def __init__(self):
        self._locking = GMSLocalFSM()
        self._waiting_writers = 0
        self._reserved_rw_session_id: Optional[str] = None
        self._condition = asyncio.Condition()
        self._next_session_id = 0

    @property
    def state(self) -> ServerState:
        return self._locking.state

    def next_session_id(self) -> str:
        self._next_session_id += 1
        return f"session_{self._next_session_id}"

    def snapshot(self) -> SessionSnapshot:
        has_rw_session = self._locking.rw_conn is not None
        return SessionSnapshot(
            state=self._locking.state,
            has_rw_session=has_rw_session,
            ro_session_count=self._locking.ro_count,
            waiting_writers=self._waiting_writers,
            committed=self._locking.committed,
            is_ready=self._locking.committed and not has_rw_session,
        )

    def _can_grant_rw(self) -> bool:
        return self._reserved_rw_session_id is None and self._locking.can_acquire_rw()

    def _can_grant_ro(self) -> bool:
        return self._reserved_rw_session_id is None and self._locking.can_acquire_ro(
            self._waiting_writers
        )

    def _can_grant_rw_or_ro(self) -> bool:
        if self._can_grant_ro():
            return True
        return self._can_grant_rw() and not self._locking.committed

    async def acquire_lock(
        self,
        mode: RequestedLockType,
        timeout_ms: Optional[int],
        session_id: str,
    ) -> Optional[GrantedLockType]:
        timeout = timeout_ms / 1000 if timeout_ms is not None else None

        if mode == RequestedLockType.RW:
            try:
                async with self._condition:
                    self._waiting_writers += 1
                    try:
                        await asyncio.wait_for(
                            self._condition.wait_for(self._can_grant_rw),
                            timeout=timeout,
                        )
                    except asyncio.TimeoutError:
                        return None
                    self._reserved_rw_session_id = session_id
                    return GrantedLockType.RW
            finally:
                async with self._condition:
                    self._waiting_writers -= 1
                    self._condition.notify_all()

        if mode == RequestedLockType.RO:
            async with self._condition:
                try:
                    await asyncio.wait_for(
                        self._condition.wait_for(self._can_grant_ro),
                        timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    return None
            return GrantedLockType.RO

        async with self._condition:
            if self._can_grant_rw() and not self._locking.committed:
                self._reserved_rw_session_id = session_id
                return GrantedLockType.RW
            try:
                await asyncio.wait_for(
                    self._condition.wait_for(self._can_grant_rw_or_ro),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                return None
            if self._can_grant_rw() and not self._locking.committed:
                self._reserved_rw_session_id = session_id
                return GrantedLockType.RW
        return GrantedLockType.RO

    async def cancel_connect(
        self,
        session_id: str,
        mode: Optional[GrantedLockType],
    ) -> None:
        if mode != GrantedLockType.RW:
            return
        async with self._condition:
            if self._reserved_rw_session_id == session_id:
                self._reserved_rw_session_id = None
                self._condition.notify_all()

    def on_connect(self, conn: Connection) -> None:
        if conn.mode == GrantedLockType.RW:
            if self._reserved_rw_session_id != conn.session_id:
                raise AssertionError(
                    f"RW session {conn.session_id} was not reserved before connect"
                )
            self._reserved_rw_session_id = None
        event = (
            StateEvent.RW_CONNECT
            if conn.mode == GrantedLockType.RW
            else StateEvent.RO_CONNECT
        )
        self._locking.transition(event, conn)

    def on_commit(self, conn: Connection) -> None:
        self._locking.transition(StateEvent.RW_COMMIT, conn)

    def check_operation(self, msg_type: type, conn: Connection) -> None:
        self._locking.check_operation(msg_type, conn)

    def begin_cleanup(self, conn: Optional[Connection]) -> StateEvent | None:
        if conn is None:
            return None

        event = None
        if conn.mode == GrantedLockType.RW:
            if self._locking.rw_conn is conn and not self._locking.committed:
                self._locking.transition(StateEvent.RW_ABORT, conn)
                event = StateEvent.RW_ABORT
        elif conn in self._locking.ro_conns:
            self._locking.transition(StateEvent.RO_DISCONNECT, conn)
            event = StateEvent.RO_DISCONNECT
        return event

    async def finish_cleanup(self, conn: Optional[Connection]) -> None:
        if conn is not None:
            await conn.close()
        async with self._condition:
            self._condition.notify_all()
