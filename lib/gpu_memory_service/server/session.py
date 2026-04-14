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


class OperationNotAllowed(Exception):
    pass


RW_REQUIRED: frozenset[type] = frozenset(
    {
        AllocateRequest,
        FreeAllocationRequest,
        MetadataPutRequest,
        MetadataDeleteRequest,
        CommitRequest,
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

RW_ALLOWED: frozenset[type] = RW_REQUIRED | RO_ALLOWED


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
        self._locking = GMSFSM()
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
