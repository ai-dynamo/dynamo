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
    # A committed layout with no session: the pages are durable and reattachable, but
    # nothing is promised about their contents. Distinct from COMMITTED, which also
    # asserts the contents are published and stable (and so admits RO readers).
    ALLOCATED = "ALLOCATED"
    COMMITTED = "COMMITTED"
    RO = "RO"


class StateEvent(Enum):
    RW_CONNECT = auto()
    RW_COMMIT = auto()
    RW_ABORT = auto()
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
            {ServerState.EMPTY, ServerState.ALLOCATED, ServerState.COMMITTED}
        ),
        event=StateEvent.RW_CONNECT,
        to_state=ServerState.RW,
    ),
    Transition(
        from_states=frozenset({ServerState.RW}),
        event=StateEvent.RW_COMMIT,
        to_state=ServerState.COMMITTED,
    ),
    # Seal the shape. The writer keeps its session and its mappings -- only the
    # allocation set is frozen -- so the derived state stays RW until it disconnects,
    # at which point RW_ABORT resolves to ALLOCATED instead of EMPTY.
    Transition(
        from_states=frozenset({ServerState.RW}),
        event=StateEvent.LAYOUT_COMMIT,
        to_state=ServerState.RW,
    ),
    # Resolves to ALLOCATED when the layout was committed, EMPTY otherwise. The
    # derived `state` property picks between them; death never demotes the layout.
    Transition(
        from_states=frozenset({ServerState.RW}),
        event=StateEvent.RW_ABORT,
        to_state=None,
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
        # Content-committed is the stronger claim (it entails a durable layout), so it
        # wins when both hold.
        if self._committed:
            return ServerState.COMMITTED
        if self._layout_committed:
            return ServerState.ALLOCATED
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

        Implied by ``committed``: publishing contents necessarily means the pages
        persist. Kept as a separate field only because a *new* writer invalidates the
        contents (``RW_CONNECT`` clears ``_committed``) without making the pages vanish.
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
            # A writer can mutate the bytes, so any published contents are no longer
            # trustworthy. The pages are untouched, so a sealed layout stays sealed --
            # unless the caller asked for full RW, which means "replace it".
            self._committed = False
            if conn.mode == GrantedLockType.RW:
                self._layout_committed = False
        elif event == StateEvent.RW_COMMIT:
            self._committed = True
            self._rw_conn = None
        elif event == StateEvent.LAYOUT_COMMIT:
            self._layout_committed = True
        elif event == StateEvent.RW_ABORT:
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
