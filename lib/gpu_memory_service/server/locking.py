# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Connection and state machine for GPU Memory Service.

This module handles:
- Connection: Represents an active client connection
- GlobalLockFSM: Explicit state transitions with validated permissions

State Diagram:

    EMPTY ──RW_CONNECT──► RW ──RW_COMMIT──► COMMITTED
      ▲                    │                   │
      │                    │                   │
      └───RW_ABORT─────────┘                   │
                                               ▼
    COMMITTED ◄──RO_DISCONNECT (last)── RO ◄──RO_CONNECT
                      │                  ▲
                      │                  │
                      └──RO_CONNECT──────┘
                      └──RO_DISCONNECT───┘ (not last)
"""

from __future__ import annotations

import asyncio
import logging
import socket
from dataclasses import dataclass, field
from typing import Callable, Optional, Set

from gpu_memory_service.common.types import (
    RO_ALLOWED,
    RW_ALLOWED,
    RW_REQUIRED,
    GrantedLockType,
    ServerState,
    StateEvent,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Connection
# =============================================================================


@dataclass(eq=False)
class Connection:
    """Represents an active connection.

    The existence of Connection objects IS the state - we don't track
    sessions separately. When a Connection is removed, the lock is released.

    Note: eq=False disables auto-generated __eq__ so we can use default
    object identity for equality and add __hash__ for use in sets.
    """

    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    mode: GrantedLockType
    session_id: str
    recv_buffer: bytearray = field(default_factory=bytearray)

    def __hash__(self) -> int:
        """Hash based on session_id (immutable identifier)."""
        return hash(self.session_id)

    @property
    def raw_socket(self) -> socket.socket:
        """Get underlying socket for FD passing."""
        return self.writer.get_extra_info("socket")

    async def close(self) -> None:
        """Close the connection."""
        self.writer.close()
        try:
            await self.writer.wait_closed()
        except Exception:
            pass


# =============================================================================
# State Machine
# =============================================================================


class InvalidTransition(Exception):
    """Raised when an invalid state transition is attempted."""

    pass


class OperationNotAllowed(Exception):
    """Raised when an operation is not allowed in the current state/mode."""

    pass


@dataclass(frozen=True)
class Transition:
    """A valid state transition.

    Attributes:
        from_states: Set of states this transition can originate from
        event: The event that triggers this transition
        to_state: The resulting state (or None if conditional)
        condition: Optional condition function for conditional transitions
    """

    from_states: frozenset[ServerState]
    event: StateEvent
    to_state: Optional[ServerState]
    condition: Optional[str] = None  # Name of condition method


# Transition table - the single source of truth for valid state transitions
TRANSITIONS: list[Transition] = [
    # From EMPTY or COMMITTED: RW can connect
    # Writer acquires exclusive lock
    Transition(
        from_states=frozenset({ServerState.EMPTY, ServerState.COMMITTED}),
        event=StateEvent.RW_CONNECT,
        to_state=ServerState.RW,
    ),
    # From RW: commit publishes and transitions to COMMITTED
    # Writer publishes and releases lock
    Transition(
        from_states=frozenset({ServerState.RW}),
        event=StateEvent.RW_COMMIT,
        to_state=ServerState.COMMITTED,
    ),
    # From RW: abort (disconnect without commit) transitions to EMPTY
    # Writer aborts, state invalidated
    Transition(
        from_states=frozenset({ServerState.RW}),
        event=StateEvent.RW_ABORT,
        to_state=ServerState.EMPTY,
    ),
    # From COMMITTED or RO: RO can connect
    # Reader acquires shared lock
    Transition(
        from_states=frozenset({ServerState.COMMITTED, ServerState.RO}),
        event=StateEvent.RO_CONNECT,
        to_state=ServerState.RO,
    ),
    # From RO: reader disconnect (not last) stays in RO
    # Reader leaves, others remain
    Transition(
        from_states=frozenset({ServerState.RO}),
        event=StateEvent.RO_DISCONNECT,
        to_state=ServerState.RO,
        condition="has_remaining_readers",
    ),
    # From RO: last reader disconnect transitions to COMMITTED
    # Last reader leaves
    Transition(
        from_states=frozenset({ServerState.RO}),
        event=StateEvent.RO_DISCONNECT,
        to_state=ServerState.COMMITTED,
        condition="is_last_reader",
    ),
]


@dataclass
class TransitionRecord:
    """Record of a state transition for debugging/auditing."""

    from_state: ServerState
    event: StateEvent
    to_state: ServerState
    session_id: Optional[str] = None


class GlobalLockFSM:
    """Explicit state machine for GPU Memory Service.

    State is DERIVED from actual connection objects:
    - _rw_conn: The active RW connection (or None)
    - _ro_conns: Set of active RO connections
    - _committed: Whether allocations have been committed

    All state mutations happen through explicit transitions.
    """

    def __init__(self, on_rw_abort: Optional[Callable[[], None]] = None):
        """Initialize the state machine.

        Args:
            on_rw_abort: Callback invoked when RW aborts (for cleanup)
        """
        # Connection state - THIS IS THE SOURCE OF TRUTH
        self._rw_conn: Optional[Connection] = None
        self._ro_conns: Set[Connection] = set()
        self._committed: bool = False

        # Callback for RW abort cleanup
        self._on_rw_abort = on_rw_abort

        # Transition history for debugging
        self._transition_log: list[TransitionRecord] = []

    # ==================== State Properties ====================

    @property
    def state(self) -> ServerState:
        """Derive current state from connection objects."""
        if self._rw_conn is not None:
            return ServerState.RW
        if len(self._ro_conns) > 0:
            return ServerState.RO
        if self._committed:
            return ServerState.COMMITTED
        return ServerState.EMPTY

    @property
    def rw_conn(self) -> Optional[Connection]:
        """The active RW connection, if any."""
        return self._rw_conn

    @property
    def ro_conns(self) -> Set[Connection]:
        """Set of active RO connections."""
        return self._ro_conns

    @property
    def ro_count(self) -> int:
        """Number of active RO connections."""
        return len(self._ro_conns)

    @property
    def committed(self) -> bool:
        """Whether allocations have been committed."""
        return self._committed

    @property
    def transition_log(self) -> list[TransitionRecord]:
        """History of state transitions."""
        return self._transition_log

    # ==================== Transition Conditions ====================

    def _has_remaining_readers(self, conn: Connection) -> bool:
        """Check if there are readers remaining after removing conn."""
        return len(self._ro_conns) > 1 or conn not in self._ro_conns

    def _is_last_reader(self, conn: Connection) -> bool:
        """Check if conn is the last reader."""
        return len(self._ro_conns) == 1 and conn in self._ro_conns

    def _check_condition(self, condition: Optional[str], conn: Connection) -> bool:
        """Evaluate a named condition."""
        if condition is None:
            return True
        if condition == "has_remaining_readers":
            return self._has_remaining_readers(conn)
        if condition == "is_last_reader":
            return self._is_last_reader(conn)
        raise ValueError(f"Unknown condition: {condition}")

    # ==================== State Transitions ====================

    def _find_transition(
        self, from_state: ServerState, event: StateEvent, conn: Connection
    ) -> Optional[Transition]:
        """Find the applicable transition for the given event."""
        for t in TRANSITIONS:
            if from_state not in t.from_states:
                continue
            if t.event != event:
                continue
            if not self._check_condition(t.condition, conn):
                continue
            return t
        return None

    def _apply_event(self, event: StateEvent, conn: Connection) -> None:
        """Mutate internal state based on event."""
        match event:
            case StateEvent.RW_CONNECT:
                self._rw_conn = conn
                self._committed = False  # Invalidate on RW connect
            case StateEvent.RW_COMMIT:
                self._committed = True
                self._rw_conn = None
            case StateEvent.RW_ABORT:
                self._rw_conn = None
                if self._on_rw_abort:
                    self._on_rw_abort()
            case StateEvent.RO_CONNECT:
                self._ro_conns.add(conn)
            case StateEvent.RO_DISCONNECT:
                self._ro_conns.discard(conn)

    def transition(self, event: StateEvent, conn: Connection) -> ServerState:
        """Execute a state transition.

        Args:
            event: The triggering event
            conn: The connection involved in the transition

        Returns:
            The new state after the transition

        Raises:
            InvalidTransition: If the transition is not valid from current state
        """
        from_state = self.state
        session_id = conn.session_id if conn else None

        # Find valid transition
        trans = self._find_transition(from_state, event, conn)
        if trans is None:
            raise InvalidTransition(
                f"No transition for {event.name} from state {from_state.name} "
                f"(session={session_id})"
            )

        # Apply the transition
        self._apply_event(event, conn)
        to_state = self.state

        # Validate we ended up in expected state
        if trans.to_state is not None and to_state != trans.to_state:
            raise InvalidTransition(
                f"Transition mismatch: expected {trans.to_state.name}, "
                f"got {to_state.name}"
            )

        # Record transition
        record = TransitionRecord(from_state, event, to_state, session_id)
        self._transition_log.append(record)

        logger.info(
            f"State transition: {from_state.name} --{event.name}--> {to_state.name} "
            f"(session={session_id})"
        )

        return to_state

    # ==================== Operation Permissions ====================

    def check_operation(self, msg_type: type, conn: Connection) -> None:
        """Check if a request type is allowed for the given connection.

        Args:
            msg_type: The request message type (e.g., AllocateRequest)
            conn: The connection attempting the operation

        Raises:
            OperationNotAllowed: If the operation is not permitted
        """
        current_state = self.state

        # Determine allowed operations based on state
        if current_state == ServerState.RW:
            allowed = RW_ALLOWED
        elif current_state == ServerState.RO:
            allowed = RO_ALLOWED
        else:
            allowed = frozenset()  # EMPTY and COMMITTED have no connections

        if msg_type not in allowed:
            raise OperationNotAllowed(
                f"{msg_type.__name__} not allowed in state {current_state.name}"
            )

        # Check connection mode
        if msg_type in RW_REQUIRED and conn.mode != GrantedLockType.RW:
            raise OperationNotAllowed(
                f"{msg_type.__name__} requires RW connection, "
                f"but connection is {conn.mode.value}"
            )

    # ==================== Lock Acquisition Predicates ====================

    def can_acquire_rw(self) -> bool:
        """Check if RW lock can be acquired now.

        RW can only be acquired if:
        - No current RW holder
        - No RO holders

        Note: This allows RW from COMMITTED state (for explicit reload).
        For rw_or_ro mode, callers should also check `committed` to prefer RO.
        """
        return self._rw_conn is None and len(self._ro_conns) == 0

    def can_acquire_ro(self, waiting_writers: int) -> bool:
        """Check if RO lock can be acquired now.

        Args:
            waiting_writers: Number of writers waiting for the lock
        """
        return self._rw_conn is None and waiting_writers == 0 and self._committed
