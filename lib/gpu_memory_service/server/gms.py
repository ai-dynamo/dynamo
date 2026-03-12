# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Top-level server-side GMS service."""

from __future__ import annotations

import logging
from typing import Callable, Optional

from gpu_memory_service.common.protocol.messages import (
    AllocateRequest,
    AllocateResponse,
    CommitRequest,
    CommitResponse,
    ExportAllocationRequest,
    ExportAllocationResponse,
    FreeAllocationRequest,
    FreeAllocationResponse,
    GetAllocationRequest,
    GetAllocationResponse,
    GetAllocationStateRequest,
    GetAllocationStateResponse,
    GetLockStateRequest,
    GetLockStateResponse,
    GetStateHashRequest,
    GetStateHashResponse,
    ListAllocationsRequest,
    ListAllocationsResponse,
    MetadataDeleteRequest,
    MetadataDeleteResponse,
    MetadataGetRequest,
    MetadataGetResponse,
    MetadataListRequest,
    MetadataListResponse,
    MetadataPutRequest,
    MetadataPutResponse,
)
from gpu_memory_service.common.types import (
    GrantedLockType,
    RequestedLockType,
    ServerState,
)

from .allocations import GMSAllocationManager
from .epochs import GMSEpochManager
from .session import Connection, GMSSessionManager

logger = logging.getLogger(__name__)


class GMS:
    """Owns all non-transport server state."""

    def __init__(
        self,
        device: int = 0,
        *,
        allocation_retry_interval: float = 0.5,
        allocation_retry_timeout: Optional[float] = None,
    ):
        allocations = GMSAllocationManager(
            device,
            allocation_retry_interval=allocation_retry_interval,
            allocation_retry_timeout=allocation_retry_timeout,
        )
        self._epochs = GMSEpochManager(allocations)
        self._sessions = GMSSessionManager(on_rw_abort=self._epochs.on_rw_abort)
        logger.info("GMS initialized: device=%d", device)

    @property
    def state(self) -> ServerState:
        return self._sessions.state

    @property
    def committed(self) -> bool:
        return self._sessions.snapshot().committed

    @property
    def committed_epoch_id(self) -> int | None:
        return self._epochs.committed_epoch_id

    @property
    def active_rw_epoch_id(self) -> int | None:
        return self._epochs.active_rw_epoch_id

    @property
    def allocation_count(self) -> int:
        return self._epochs.allocation_count

    def is_ready(self) -> bool:
        return self._sessions.snapshot().is_ready

    def next_session_id(self) -> str:
        return self._sessions.next_session_id()

    async def acquire_lock(
        self,
        mode: RequestedLockType,
        timeout_ms: int | None,
        session_id: str,
    ) -> GrantedLockType | None:
        return await self._sessions.acquire_lock(mode, timeout_ms, session_id)

    async def cancel_connect(
        self,
        session_id: str,
        mode: GrantedLockType | None,
    ) -> None:
        await self._sessions.cancel_connect(session_id, mode)

    def on_connect(self, conn: Connection) -> None:
        rw_epoch_initialized = False
        try:
            if conn.mode == GrantedLockType.RW:
                self._epochs.on_rw_connect()
                rw_epoch_initialized = True
            self._sessions.on_connect(conn)
        except Exception:
            if rw_epoch_initialized:
                self._epochs.on_rw_abort()
            raise

    async def cleanup_connection(self, conn: Connection | None) -> None:
        await self._sessions.cleanup_connection(conn)

    async def handle_request(
        self,
        conn: Connection,
        msg,
        is_connected: Callable[[], bool],
    ) -> tuple[object, int, bool]:
        msg_type = type(msg)
        self._sessions.check_operation(msg_type, conn)

        if msg_type is CommitRequest:
            self._epochs.on_commit()
            self._sessions.on_commit(conn)
            return CommitResponse(success=True), -1, True

        if msg_type is AllocateRequest:
            info = await self._epochs.allocate(msg.size, msg.tag, is_connected)
            return (
                AllocateResponse(
                    allocation_id=info.allocation_id,
                    size=info.size,
                    aligned_size=info.aligned_size,
                    epoch_id=info.epoch_id,
                ),
                -1,
                False,
            )

        if msg_type is GetLockStateRequest:
            snapshot = self._sessions.snapshot()
            return (
                GetLockStateResponse(
                    state=snapshot.state.name,
                    has_rw_session=snapshot.has_rw_session,
                    ro_session_count=snapshot.ro_session_count,
                    waiting_writers=snapshot.waiting_writers,
                    committed=snapshot.committed,
                    is_ready=snapshot.is_ready,
                ),
                -1,
                False,
            )

        if msg_type is GetAllocationStateRequest:
            return (
                GetAllocationStateResponse(
                    allocation_count=self._epochs.allocation_count
                ),
                -1,
                False,
            )

        if msg_type is ExportAllocationRequest:
            info, fd = self._epochs.export_allocation(conn.mode, msg.allocation_id)
            return (
                ExportAllocationResponse(
                    allocation_id=info.allocation_id,
                    size=info.size,
                    aligned_size=info.aligned_size,
                    tag=info.tag,
                    epoch_id=info.epoch_id,
                ),
                fd,
                False,
            )

        if msg_type is GetStateHashRequest:
            return (
                GetStateHashResponse(
                    memory_layout_hash=self._epochs.memory_layout_hash
                ),
                -1,
                False,
            )

        if msg_type is GetAllocationRequest:
            info = self._epochs.get_allocation(conn.mode, msg.allocation_id)
            return (
                GetAllocationResponse(
                    allocation_id=info.allocation_id,
                    size=info.size,
                    aligned_size=info.aligned_size,
                    tag=info.tag,
                    epoch_id=info.epoch_id,
                ),
                -1,
                False,
            )

        if msg_type is ListAllocationsRequest:
            return (
                ListAllocationsResponse(
                    allocations=[
                        GetAllocationResponse(
                            allocation_id=info.allocation_id,
                            size=info.size,
                            aligned_size=info.aligned_size,
                            tag=info.tag,
                            epoch_id=info.epoch_id,
                        )
                        for info in self._epochs.list_allocations(conn.mode, msg.tag)
                    ]
                ),
                -1,
                False,
            )

        if msg_type is FreeAllocationRequest:
            return (
                FreeAllocationResponse(
                    success=self._epochs.free_allocation(msg.allocation_id)
                ),
                -1,
                False,
            )

        if msg_type is MetadataPutRequest:
            self._epochs.put_metadata(
                msg.key,
                msg.allocation_id,
                msg.offset_bytes,
                msg.value,
            )
            return MetadataPutResponse(success=True), -1, False

        if msg_type is MetadataGetRequest:
            entry = self._epochs.get_metadata(conn.mode, msg.key)
            if entry is None:
                return MetadataGetResponse(found=False), -1, False
            return (
                MetadataGetResponse(
                    found=True,
                    allocation_id=entry.allocation_id,
                    offset_bytes=entry.offset_bytes,
                    value=entry.value,
                ),
                -1,
                False,
            )

        if msg_type is MetadataDeleteRequest:
            return (
                MetadataDeleteResponse(deleted=self._epochs.delete_metadata(msg.key)),
                -1,
                False,
            )

        if msg_type is MetadataListRequest:
            return (
                MetadataListResponse(
                    keys=self._epochs.list_metadata(conn.mode, msg.prefix)
                ),
                -1,
                False,
            )

        raise ValueError(f"Unknown request: {msg_type.__name__}")
