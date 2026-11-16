# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Server-side lock acquisition and cleanup."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType
from gpu_memory_service.common.protocol.messages import (
    AllocateRequest,
    CommitLayoutRequest,
    CommitRequest,
    ExportAllocationRequest,
    FreeAllocationRequest,
    GetAllocationRequest,
    GetAllocationStateRequest,
    GetLockStateRequest,
    GetStateHashRequest,
    ListAllocationsRequest,
    MetadataDeleteRequest,
    MetadataGetRequest,
    MetadataListRequest,
    MetadataPutRequest,
)

from .fsm import GMSFSM, Connection, ServerState, StateEvent

logger = logging.getLogger(__name__)


class OperationNotAllowed(Exception):
    pass


RW_REQUIRED: frozenset[type] = frozenset(
    {
        AllocateRequest,
        FreeAllocationRequest,
        MetadataPutRequest,
        MetadataDeleteRequest,
        CommitRequest,
        CommitLayoutRequest,
    }
)

RO_ALLOWED: frozenset[type] = frozenset(
    {
        ExportAllocationRequest,
        GetAllocationRequest,
        ListAllocationsRequest,
        MetadataGetRequest,
        MetadataListRequest,
        GetLockStateRequest,
        GetAllocationStateRequest,
        GetStateHashRequest,
    }
)

# The operations that change a layout's shape. Sealing is exactly the act of giving
# these up, so they are the difference between RW and RW_DATA.
LAYOUT_MUTATING: frozenset[type] = frozenset(
    {
        AllocateRequest,
        FreeAllocationRequest,
        MetadataPutRequest,
        MetadataDeleteRequest,
    }
)

# Equal to RO_ALLOWED because an RW_DATA session's real work, writing bytes into its own
# mappings, never reaches the server. It still holds the exclusive writer slot.
RW_DATA_ALLOWED: frozenset[type] = RO_ALLOWED

RW_ALLOWED: frozenset[type] = RW_REQUIRED | RO_ALLOWED

# Permission is a pure function of the granted mode. State reaches it only by deciding
# which mode is granted; see resolve_writer_mode.
_ALLOWED_BY_MODE: dict[GrantedLockType, frozenset[type]] = {
    GrantedLockType.RO: RO_ALLOWED,
    GrantedLockType.RW_DATA: RW_DATA_ALLOWED,
    GrantedLockType.RW: RW_ALLOWED,
}


@dataclass(frozen=True)
class SessionSnapshot:
    state: ServerState
    has_rw_session: bool
    ro_session_count: int
    waiting_writers: int
    committed: bool
    is_ready: bool
    # Implied by `committed`; reported separately so an operator can tell "held by a
    # live writer" from "held for reattach".
    layout_committed: bool = False


class GMSSessionManager:
    """Owns lock transitions, waiter coordination, and cleanup."""

    def __init__(self):
        self._locking = GMSFSM()
        self._waiting_writers = 0
        self._reserved_rw_session_id: Optional[str] = None
        self._condition = asyncio.Condition()
        self._next_session_id = 0

    @property
    def state(self) -> ServerState:
        return self._locking.state

    @property
    def layout_committed(self) -> bool:
        return self._locking.layout_committed

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
            layout_committed=self._locking.layout_committed,
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

    def resolve_writer_mode(self, requested: RequestedLockType) -> GrantedLockType:
        """Which writer mode a request earns against the current layout.

        ``RW`` means "replace it"; ``RW_DATA_OR_RW`` means "adopt it if it exists".
        ``RW_DATA`` is only granted, never requested, and only from LAYOUT_COMMITTED: from
        COMMITTED the contents are published, so a writer arriving means replace.
        """
        if (
            requested is RequestedLockType.RW_DATA_OR_RW
            and self._locking.state is ServerState.LAYOUT_COMMITTED
        ):
            return GrantedLockType.RW_DATA
        return GrantedLockType.RW

    async def acquire_lock(
        self,
        mode: RequestedLockType,
        timeout_ms: Optional[int],
        session_id: str,
    ) -> Optional[GrantedLockType]:
        timeout = timeout_ms / 1000 if timeout_ms is not None else None

        # All writer requests are exclusive, whichever mode they resolve to.
        if mode in (RequestedLockType.RW, RequestedLockType.RW_DATA_OR_RW):
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
                    return self.resolve_writer_mode(mode)
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
        if mode not in (GrantedLockType.RW, GrantedLockType.RW_DATA):
            return
        async with self._condition:
            if self._reserved_rw_session_id == session_id:
                self._reserved_rw_session_id = None
                self._condition.notify_all()

    def on_connect(self, conn: Connection) -> None:
        is_writer = conn.mode in (GrantedLockType.RW, GrantedLockType.RW_DATA)
        if is_writer:
            if self._reserved_rw_session_id != conn.session_id:
                raise AssertionError(
                    f"{conn.mode.name} session {conn.session_id} "
                    "was not reserved before connect"
                )
            self._reserved_rw_session_id = None
        event = StateEvent.RW_CONNECT if is_writer else StateEvent.RO_CONNECT
        self._locking.transition(event, conn)

    def on_commit(self, conn: Connection) -> None:
        self._locking.transition(StateEvent.RW_COMMIT, conn)

    def on_layout_commit(self, conn: Connection) -> None:
        """Seal the shape and narrow the caller to RW_DATA.

        Permission is a function of the granted mode, so sealing is expressed by
        narrowing it. The caller keeps its session and its mappings.
        """
        self._locking.transition(StateEvent.LAYOUT_COMMIT, conn)
        self._regrant(conn, GrantedLockType.RW_DATA)

    def _regrant(self, conn: Connection, mode: GrantedLockType) -> None:
        """The only place a live session's grant changes.

        Logged: a capability changing underneath a client is worth having in the log.
        """
        if conn.mode is mode:
            return
        logger.info(
            "session %s regranted %s -> %s",
            conn.session_id,
            conn.mode.name,
            mode.name,
        )
        conn.mode = mode

    def check_operation(self, msg_type: type, conn: Connection) -> None:
        allowed = _ALLOWED_BY_MODE.get(conn.mode, frozenset())
        if msg_type not in allowed:
            if conn.mode == GrantedLockType.RW_DATA and msg_type in LAYOUT_MUTATING:
                raise OperationNotAllowed(
                    f"{msg_type.__name__} not allowed: the layout is committed. "
                    f"Reconnect with RW to replace it."
                )
            raise OperationNotAllowed(
                f"{msg_type.__name__} not allowed for {conn.mode.name} session "
                f"in state {self.state.name}"
            )

    def begin_cleanup(self, conn: Optional[Connection]) -> StateEvent | None:
        if conn is None:
            return None

        event = None
        # RW_DATA is still the writer. Missing it here would leave _rw_conn set forever
        # and wedge every later connect.
        if conn.mode in (GrantedLockType.RW, GrantedLockType.RW_DATA):
            if self._locking.rw_conn is conn and not self._locking.committed:
                # A sealed layout survives its writer; an unsealed one, including a
                # pool the writer died part-way through building, is discarded.
                event = (
                    StateEvent.RW_DISCONNECT
                    if self._locking.layout_committed
                    else StateEvent.RW_ABORT
                )
                self._locking.transition(event, conn)
        elif conn in self._locking.ro_conns:
            self._locking.transition(StateEvent.RO_DISCONNECT, conn)
            event = StateEvent.RO_DISCONNECT
        return event

    async def finish_cleanup(self, conn: Optional[Connection]) -> None:
        if conn is not None:
            await conn.close()
        async with self._condition:
            self._condition.notify_all()
