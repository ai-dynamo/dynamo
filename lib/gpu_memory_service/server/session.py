# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Server-side lock acquisition and cleanup."""

from __future__ import annotations

import asyncio
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
    ReleaseLayoutRequest,
)

from .fsm import GMSFSM, Connection, ServerState, StateEvent


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
        ReleaseLayoutRequest,
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

# Operations that change the shape of the layout -- which allocations exist, how big
# they are, and the metadata folded into the layout hash. Sealing the layout is exactly
# the act of giving these up, so they are the difference between RW and RW_DATA.
LAYOUT_MUTATING: frozenset[type] = frozenset(
    {
        AllocateRequest,
        FreeAllocationRequest,
        MetadataPutRequest,
        MetadataDeleteRequest,
    }
)

# A sealed-layout writer: everything a reader may do, plus writing bytes (which never
# reaches the server) and abandoning the layout wholesale. Notably excludes CommitRequest
# -- publishing contents that were never intended as a publication is deliberately not
# reachable from here.
RW_DATA_ALLOWED: frozenset[type] = RO_ALLOWED | {ReleaseLayoutRequest}

RW_ALLOWED: frozenset[type] = RW_REQUIRED | RO_ALLOWED

# Permission stays a pure function of the granted mode; the durability state reaches it
# only by choosing which mode is granted (see GMSSessionManager.resolve_writer_mode).
_ALLOWED_BY_MODE: dict[GrantedLockType, frozenset[type]] = {
    GrantedLockType.RO: RO_ALLOWED,
    GrantedLockType.RW_DATA: RW_DATA_ALLOWED,
    GrantedLockType.RW: RW_ALLOWED,
}

_WRITER_REQUESTS: frozenset[RequestedLockType] = frozenset(
    {
        RequestedLockType.RW,
        RequestedLockType.RW_DATA_OR_RW,
    }
)


@dataclass(frozen=True)
class SessionSnapshot:
    state: ServerState
    has_rw_session: bool
    ro_session_count: int
    waiting_writers: int
    committed: bool
    is_ready: bool
    # The allocation set is sealed and outlives its session. Implied by `committed`;
    # reported separately so an operator can tell "pages held because a writer is live"
    # from "pages held deliberately for reattach".
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

        This is where adopt-vs-replace is decided, and the client decides it by what it
        asks for -- a caller that cannot reshape a layout has no use for a wiped one:

          * ``RW``            -- "replace it": full control, the layout is cleared
          * ``RW_DATA_OR_RW`` -- "adopt it if it's there, otherwise build one"

        ``RW_DATA`` is only ever *granted*, never requested: it means "you joined a
        layout somebody else sealed". Adopting is offered only from ALLOCATED -- from
        COMMITTED the contents are published, so a writer showing up means replace, which
        is exactly what a writer connecting to a committed layout does today.
        """
        if (
            requested is RequestedLockType.RW_DATA_OR_RW
            and self._locking.state is ServerState.ALLOCATED
        ):
            return GrantedLockType.RW_DATA
        return GrantedLockType.RW

    def check_admissible(self, granted: GrantedLockType) -> None:
        """Reject a connect whose intent cannot be honoured, rather than guessing.

        Deliberately not a silent fallback: a reader asking for a pool whose contents are
        unspecified has misunderstood the server, and handing it a live mutating buffer
        that it will treat as published data is the one failure this state exists to
        prevent.
        """
        if (
            granted is GrantedLockType.RO
            and self._locking.state is ServerState.ALLOCATED
        ):
            raise OperationNotAllowed(
                "RO not available: the layout is committed but its contents are not. "
                "Nothing may attach read-only to a live, mutating pool."
            )

    async def acquire_lock(
        self,
        mode: RequestedLockType,
        timeout_ms: Optional[int],
        session_id: str,
    ) -> Optional[GrantedLockType]:
        timeout = timeout_ms / 1000 if timeout_ms is not None else None

        # Writer requests are exclusive regardless of which mode they resolve to --
        # RW_DATA is narrower in what it may *do*, not in how many may hold it -- so they
        # queue on the same predicate and differ only in the mode granted.
        if mode in _WRITER_REQUESTS:
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
                    granted = self.resolve_writer_mode(mode)
                    self.check_admissible(granted)
                    self._reserved_rw_session_id = session_id
                    return granted
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
                self.check_admissible(GrantedLockType.RO)
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
                    f"{conn.mode.value} session {conn.session_id} "
                    "was not reserved before connect"
                )
            self._reserved_rw_session_id = None
        event = StateEvent.RW_CONNECT if is_writer else StateEvent.RO_CONNECT
        self._locking.transition(event, conn)

    def on_commit(self, conn: Connection) -> None:
        self._locking.transition(StateEvent.RW_COMMIT, conn)

    def on_layout_commit(self, conn: Connection) -> None:
        """Seal the shape and narrow the caller to RW_DATA.

        Permission is a pure function of the granted mode, so sealing is *expressed* by
        narrowing the mode rather than by a second check against the durability state.
        The caller keeps its session and its mappings; it simply may no longer reshape
        what it just froze.
        """
        self._locking.transition(StateEvent.LAYOUT_COMMIT, conn)
        conn.mode = GrantedLockType.RW_DATA

    def on_layout_release(self, conn: Connection) -> None:
        """Unseal, and widen the caller back to RW.

        The state drops (there is no longer a durable layout) while the caller's
        capability rises -- RW_DATA exists only to protect a sealed layout, so once
        there is none the restriction is meaningless. That is what lets a standby which
        adopted an incompatible layout recover in-session, without ever dropping the lock.
        """
        self._locking.transition(StateEvent.LAYOUT_RELEASE, conn)
        conn.mode = GrantedLockType.RW

    def check_operation(self, msg_type: type, conn: Connection) -> None:
        allowed = _ALLOWED_BY_MODE.get(conn.mode, frozenset())
        if msg_type not in allowed:
            # Name the reason rather than the mode: a caller holding RW_DATA asked for
            # RW and was downgraded because the layout is sealed, so "requires RW" alone
            # would read as a lock-acquisition problem rather than a sealed layout.
            if conn.mode == GrantedLockType.RW_DATA and msg_type in LAYOUT_MUTATING:
                raise OperationNotAllowed(
                    f"{msg_type.__name__} not allowed: the layout is committed. "
                    f"Call release_layout() first to reshape it."
                )
            raise OperationNotAllowed(
                f"{msg_type.__name__} not allowed for {conn.mode.name} session "
                f"in state {self.state.name}"
            )

    def begin_cleanup(self, conn: Optional[Connection]) -> StateEvent | None:
        if conn is None:
            return None

        event = None
        # RW_DATA is still the writer -- a session that sealed its layout, or adopted
        # one, holds the same exclusive slot. Missing it here would leave _rw_conn set
        # forever and wedge every later connect.
        if conn.mode in (GrantedLockType.RW, GrantedLockType.RW_DATA):
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
