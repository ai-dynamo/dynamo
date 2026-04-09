# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Top-level server-side GMS service."""

from __future__ import annotations

import hashlib
import logging
from collections import deque
from dataclasses import dataclass
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
    GetEventHistoryResponse,
    GetLockStateRequest,
    GetLockStateResponse,
    GetRuntimeStateResponse,
    GetStateHashRequest,
    GetStateHashResponse,
    GMSRuntimeEvent,
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
    StateEvent,
)

from .allocations import AllocationInfo, GMSAllocationManager
from .session import Connection, GMSSessionManager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetadataEntry:
    allocation_id: str
    offset_bytes: int
    value: bytes


class GMS:
    """Owns all non-transport server state."""

    _MAX_EVENTS = 256

    def __init__(
        self,
        device: int = 0,
        *,
        allocation_retry_interval: float = 0.5,
        allocation_retry_timeout: Optional[float] = None,
    ):
        self._allocations = GMSAllocationManager(
            device,
            allocation_retry_interval=allocation_retry_interval,
            allocation_retry_timeout=allocation_retry_timeout,
        )
        self._sessions = GMSSessionManager()
        self._events: deque[GMSRuntimeEvent] = deque(maxlen=self._MAX_EVENTS)
        self._metadata: dict[str, MetadataEntry] = {}
        self._memory_layout_hash = ""
        logger.info("GMS initialized: device=%d", device)

    @property
    def state(self) -> ServerState:
        return self._sessions.state

    @property
    def committed(self) -> bool:
        return self._sessions.snapshot().committed

    @property
    def allocation_count(self) -> int:
        return self._allocations.allocation_count

    def is_ready(self) -> bool:
        return self._sessions.snapshot().is_ready

    def get_runtime_state(self) -> GetRuntimeStateResponse:
        session = self._sessions.snapshot()
        return GetRuntimeStateResponse(
            state=session.state.name,
            has_rw_session=session.has_rw_session,
            ro_session_count=session.ro_session_count,
            waiting_writers=session.waiting_writers,
            committed=session.committed,
            is_ready=session.is_ready,
            allocation_count=self._allocations.allocation_count,
            memory_layout_hash=self._memory_layout_hash,
        )

    def get_event_history(self) -> GetEventHistoryResponse:
        return GetEventHistoryResponse(events=list(self._events))

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

    def _validate_metadata_target(
        self,
        allocation: AllocationInfo,
        offset_bytes: int,
    ) -> None:
        if offset_bytes < 0:
            raise ValueError(f"offset_bytes must be >= 0, got {offset_bytes}")
        if offset_bytes >= allocation.aligned_size:
            raise ValueError(
                f"offset_bytes {offset_bytes} out of range for allocation {allocation.allocation_id} "
                f"(aligned_size={allocation.aligned_size})"
            )

    def _drop_metadata_for_allocation(self, allocation_id: str) -> int:
        keys_to_remove = [
            key
            for key, entry in self._metadata.items()
            if entry.allocation_id == allocation_id
        ]
        for key in keys_to_remove:
            self._metadata.pop(key, None)
        return len(keys_to_remove)

    def _validate_metadata_integrity(
        self,
        allocations_by_id: dict[str, AllocationInfo],
    ) -> None:
        for key, entry in self._metadata.items():
            info = allocations_by_id.get(entry.allocation_id)
            if info is None:
                raise AssertionError(
                    f"Metadata key {key!r} references missing allocation "
                    f"{entry.allocation_id!r}"
                )

            if entry.offset_bytes < 0 or entry.offset_bytes >= info.aligned_size:
                raise AssertionError(
                    f"Metadata key {key!r} has invalid offset {entry.offset_bytes} "
                    f"for allocation {entry.allocation_id!r} "
                    f"(aligned_size={info.aligned_size})"
                )

    def _compute_memory_layout_hash(self, allocations: list[AllocationInfo]) -> str:
        h = hashlib.sha256()
        allocation_slots_by_id: dict[str, int] = {}
        for info in sorted(allocations, key=lambda info: info.layout_slot):
            allocation_slots_by_id[info.allocation_id] = info.layout_slot
            h.update(
                f"{info.layout_slot}:{info.size}:{info.aligned_size}:{info.tag}".encode()
            )

        for key in sorted(self._metadata):
            entry = self._metadata[key]
            layout_slot = allocation_slots_by_id[entry.allocation_id]
            h.update(f"{key}:{layout_slot}:{entry.offset_bytes}:".encode())
            h.update(entry.value)
        return h.hexdigest()

    def _clear_layout_state(self) -> int:
        self._metadata.clear()
        self._memory_layout_hash = ""
        return self._allocations.clear_all()

    def on_connect(self, conn: Connection) -> None:
        if conn.mode == GrantedLockType.RW:
            had_committed_layout = self._sessions.snapshot().committed
            cleared = self._clear_layout_state()
            if had_committed_layout:
                self._events.append(
                    GMSRuntimeEvent(
                        kind="allocations_cleared",
                        allocation_count=cleared,
                    )
                )

        self._sessions.on_connect(conn)
        if conn.mode == GrantedLockType.RW:
            self._events.append(GMSRuntimeEvent(kind="rw_connected"))

    async def cleanup_connection(self, conn: Connection | None) -> None:
        event = self._sessions.begin_cleanup(conn)
        if event == StateEvent.RW_ABORT:
            logger.warning("RW aborted; clearing active layout")
            cleared = self._clear_layout_state()
            self._events.append(GMSRuntimeEvent(kind="rw_aborted"))
            self._events.append(
                GMSRuntimeEvent(
                    kind="allocations_cleared",
                    allocation_count=cleared,
                )
            )
        await self._sessions.finish_cleanup(conn)

    async def handle_request(
        self,
        conn: Connection,
        msg,
        is_connected: Callable[[], bool],
    ) -> tuple[object, int, bool]:
        msg_type = type(msg)
        self._sessions.check_operation(msg_type, conn)

        if msg_type is CommitRequest:
            if self.state != ServerState.RW:
                raise AssertionError("RW state is not active")

            allocations = self._allocations.list_allocations()
            allocations_by_id = {info.allocation_id: info for info in allocations}
            self._validate_metadata_integrity(allocations_by_id)
            self._memory_layout_hash = self._compute_memory_layout_hash(allocations)

            logger.info(
                "Committed layout with state hash: %s...",
                self._memory_layout_hash[:16],
            )
            self._sessions.on_commit(conn)
            self._events.append(GMSRuntimeEvent(kind="committed"))
            return CommitResponse(success=True), -1, True

        if msg_type is AllocateRequest:
            if self.state != ServerState.RW:
                raise AssertionError("RW state is not active")

            info = await self._allocations.allocate(
                size=msg.size,
                tag=msg.tag,
                is_connected=is_connected,
                on_oom=lambda: self._events.append(
                    GMSRuntimeEvent(
                        kind="allocation_oom",
                        allocation_count=self._allocations.allocation_count,
                    )
                ),
            )
            return (
                AllocateResponse(
                    allocation_id=info.allocation_id,
                    size=info.size,
                    aligned_size=info.aligned_size,
                    layout_slot=info.layout_slot,
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
                    allocation_count=self._allocations.allocation_count
                ),
                -1,
                False,
            )

        if msg_type is ExportAllocationRequest:
            info = self._allocations.get_allocation(msg.allocation_id)
            fd = self._allocations.export_allocation(info.allocation_id)
            return (
                ExportAllocationResponse(
                    allocation_id=info.allocation_id,
                    size=info.size,
                    aligned_size=info.aligned_size,
                    tag=info.tag,
                    layout_slot=info.layout_slot,
                ),
                fd,
                False,
            )

        if msg_type is GetStateHashRequest:
            return (
                GetStateHashResponse(memory_layout_hash=self._memory_layout_hash),
                -1,
                False,
            )

        if msg_type is GetAllocationRequest:
            info = self._allocations.get_allocation(msg.allocation_id)
            return (
                GetAllocationResponse(
                    allocation_id=info.allocation_id,
                    size=info.size,
                    aligned_size=info.aligned_size,
                    tag=info.tag,
                    layout_slot=info.layout_slot,
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
                            layout_slot=info.layout_slot,
                        )
                        for info in self._allocations.list_allocations(msg.tag)
                    ]
                ),
                -1,
                False,
            )

        if msg_type is FreeAllocationRequest:
            success = self._allocations.free_allocation(msg.allocation_id)
            if success:
                self._drop_metadata_for_allocation(msg.allocation_id)
            return (
                FreeAllocationResponse(success=success),
                -1,
                False,
            )

        if msg_type is MetadataPutRequest:
            allocation = self._allocations.get_allocation(msg.allocation_id)
            self._validate_metadata_target(allocation, msg.offset_bytes)
            self._metadata[msg.key] = MetadataEntry(
                allocation_id=allocation.allocation_id,
                offset_bytes=msg.offset_bytes,
                value=msg.value,
            )
            return MetadataPutResponse(success=True), -1, False

        if msg_type is MetadataGetRequest:
            entry = self._metadata.get(msg.key)
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
                MetadataDeleteResponse(
                    deleted=self._metadata.pop(msg.key, None) is not None
                ),
                -1,
                False,
            )

        if msg_type is MetadataListRequest:
            if not msg.prefix:
                keys = sorted(self._metadata)
            else:
                keys = sorted(
                    key for key in self._metadata if key.startswith(msg.prefix)
                )
            return MetadataListResponse(keys=keys), -1, False

        raise ValueError(f"Unknown request: {msg_type.__name__}")
