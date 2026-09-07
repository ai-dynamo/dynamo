# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Set

from gpu_memory_service.common.locks import GrantedLockType


class ServerState(str, Enum):
    EMPTY = "EMPTY"
    RW = "RW"
    # Pages are durable and reattachable; nothing is promised about their contents.
    # COMMITTED additionally asserts the contents are stable, which is what admits readers.
    LAYOUT_COMMITTED = "LAYOUT_COMMITTED"
    COMMITTED = "COMMITTED"
    RO = "RO"


class StateEvent(Enum):
    RW_CONNECT = auto()
    RW_COMMIT = auto()
    RW_ABORT = auto()
    RW_DISCONNECT = auto()
    LAYOUT_COMMIT = auto()
    RO_CONNECT = auto()
    RO_DISCONNECT = auto()


@dataclass(eq=False)
class Connection:
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    mode: GrantedLockType
    session_id: str
    recv_buffer: bytearray = field(default_factory=bytearray)

    def __hash__(self) -> int:
        return hash(self.session_id)

    async def close(self) -> None:
        self.writer.close()
        try:
            await self.writer.wait_closed()
        except Exception:
            pass


class InvalidTransition(Exception):
    pass


@dataclass(frozen=True)
class Transition:
    from_states: frozenset[ServerState]
    event: StateEvent
    to_state: Optional[ServerState]
    condition: Optional[str] = None


TRANSITIONS: list[Transition] = [
    Transition(
        from_states=frozenset(
            {ServerState.EMPTY, ServerState.LAYOUT_COMMITTED, ServerState.COMMITTED}
        ),
        event=StateEvent.RW_CONNECT,
        to_state=ServerState.RW,
    ),
    Transition(
        from_states=frozenset({ServerState.RW}),
        event=StateEvent.RW_COMMIT,
        to_state=ServerState.COMMITTED,
    ),
    # Seals the shape without ending the session, so the state stays RW until the
    # writer disconnects.
    Transition(
        from_states=frozenset({ServerState.RW}),
        event=StateEvent.LAYOUT_COMMIT,
        to_state=ServerState.RW,
    ),
    # The writer left without sealing, so its work is discarded.
    Transition(
        from_states=frozenset({ServerState.RW}),
        event=StateEvent.RW_ABORT,
        to_state=ServerState.EMPTY,
    ),
    # The writer left a sealed layout behind, so the pages survive it.
    Transition(
        from_states=frozenset({ServerState.RW}),
        event=StateEvent.RW_DISCONNECT,
        to_state=ServerState.LAYOUT_COMMITTED,
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


class GMSFSM:
    def __init__(self):
        self._rw_conn: Optional[Connection] = None
        self._ro_conns: Set[Connection] = set()
        self._committed = False
        self._layout_committed = False

    @property
    def state(self) -> ServerState:
        if self._rw_conn is not None:
            return ServerState.RW
        if self._ro_conns:
            return ServerState.RO
        # COMMITTED implies a durable layout, so it wins when both hold.
        if self._committed:
            return ServerState.COMMITTED
        if self._layout_committed:
            return ServerState.LAYOUT_COMMITTED
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

    @property
    def layout_committed(self) -> bool:
        """The allocation set is sealed, so it outlives the session that built it.

        Implied by ``committed``. Separate because ``RW_CONNECT`` clears ``_committed``
        -- a new writer invalidates the contents -- but must not clear this.
        """
        return self._layout_committed or self._committed

    def _check_condition(self, condition: Optional[str], conn: Connection) -> bool:
        if condition is None:
            return True
        if condition == "has_remaining_readers":
            return len(self._ro_conns) > 1 or conn not in self._ro_conns
        if condition == "is_last_reader":
            return len(self._ro_conns) == 1 and conn in self._ro_conns
        raise ValueError(f"Unknown condition: {condition}")

    def transition(self, event: StateEvent, conn: Connection) -> ServerState:
        from_state = self.state
        for transition in TRANSITIONS:
            if from_state not in transition.from_states:
                continue
            if transition.event != event:
                continue
            if not self._check_condition(transition.condition, conn):
                continue
            break
        else:
            raise InvalidTransition(
                f"No transition for {event.name} from state {from_state.name} "
                f"(session={conn.session_id})"
            )

        if event == StateEvent.RW_CONNECT:
            self._rw_conn = conn
            # A new writer invalidates any published contents but not the pages,
            # unless it asked for full RW, which means "replace this layout".
            self._committed = False
            if conn.mode == GrantedLockType.RW:
                self._layout_committed = False
        elif event == StateEvent.RW_COMMIT:
            self._committed = True
            self._rw_conn = None
        elif event == StateEvent.LAYOUT_COMMIT:
            self._layout_committed = True
        elif event in (StateEvent.RW_ABORT, StateEvent.RW_DISCONNECT):
            self._rw_conn = None
        elif event == StateEvent.RO_CONNECT:
            self._ro_conns.add(conn)
        elif event == StateEvent.RO_DISCONNECT:
            self._ro_conns.discard(conn)
        return self.state

    def can_acquire_rw(self) -> bool:
        return self._rw_conn is None and not self._ro_conns

    def can_acquire_ro(self, waiting_writers: int) -> bool:
        return self._committed and self._rw_conn is None and waiting_writers == 0
