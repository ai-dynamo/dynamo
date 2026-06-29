# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Top-level server-side GMS service."""

from __future__ import annotations

import hashlib
import logging
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType
from gpu_memory_service.common.protocol.messages import (
    AllocateRequest,
    AllocateResponse,
    ClaimPersistentAllocationRequest,
    ClaimPersistentAllocationResponse,
    CommitRequest,
    CommitResponse,
    ErrorResponse,
    ExportAllocationRequest,
    ExportAllocationResponse,
    ExportPersistentAllocationRequest,
    ExportPersistentAllocationResponse,
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
    ListPersistentAllocationsRequest,
    ListPersistentAllocationsResponse,
    MetadataDeleteRequest,
    MetadataDeleteResponse,
    MetadataGetRequest,
    MetadataGetResponse,
    MetadataListRequest,
    MetadataListResponse,
    MetadataPutRequest,
    MetadataPutResponse,
    PersistentAllocationInfo,
    ReleasePersistentAllocationRequest,
    ReleasePersistentAllocationResponse,
)

from .allocations import AllocationInfo, GMSAllocationManager
from .fsm import Connection, ServerState, StateEvent
from .persistent_allocations import (
    PersistentAllocationManager,
    PersistentClaimConflictError,
    PersistentNotFoundError,
)
from .session import GMSSessionManager

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
        self._persistent = PersistentAllocationManager(device)
        # Per-session set of (engine_id, tag) keys claimed via the
        # persistent namespace, so we know what to unclaim on disconnect.
        # The unclaim releases the CLAIM (lets another client attach
        # next), not the underlying allocation — that requires explicit
        # release_persistent.
        self._persistent_claims_by_session: dict[str, set[tuple[str, str]]] = {}
        self._sessions = GMSSessionManager()
        self._events: deque[GMSRuntimeEvent] = deque(maxlen=self._MAX_EVENTS)
        self._metadata: dict[str, MetadataEntry] = {}
        self._memory_layout_hash = ""
        # Last-observed active persistent-claim count, used to project the
        # persistent KV layout onto the FSM event history (rw_connected when
        # the first claim establishes a KV layout; rw_aborted +
        # allocations_cleared when the last claim is released on pause/crash).
        self._persistent_layout_count = 0
        logger.info("GMS initialized: device=%d", device)

    @property
    def persistent(self) -> PersistentAllocationManager:
        return self._persistent

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

    def _sync_persistent_layout_events(self) -> None:
        """Project persistent-claim transitions onto the FSM event history.

        Persistent KV bypasses the single-writer FSM (RW_PERSISTENT grants
        return immediately and never transition the lock state), so its layout
        is otherwise invisible to layout assertions. Mirror the FSM vocabulary:
        the first active claim opens an RW KV layout (``rw_connected``); losing
        the last claim — on engine pause (``abort()``) or crash-cleanup — tears
        it down (``rw_aborted`` + ``allocations_cleared``). Call after every
        claim/release/disconnect so the projection stays edge-accurate.
        """
        now = self._persistent.active_claim_count
        prev = self._persistent_layout_count
        if prev == 0 and now > 0:
            self._events.append(GMSRuntimeEvent(kind="rw_connected"))
        elif prev > 0 and now == 0:
            self._events.append(GMSRuntimeEvent(kind="rw_aborted"))
            self._events.append(
                GMSRuntimeEvent(kind="allocations_cleared", allocation_count=prev)
            )
        self._persistent_layout_count = now

    def get_runtime_state(self) -> GetRuntimeStateResponse:
        session = self._sessions.snapshot()
        state = session.state
        allocation_count = self._allocations.allocation_count
        # Project the persistent KV layout onto the reported state when the
        # FSM itself is idle (the kv_cache daemon never drives the weights FSM).
        persistent_claims = self._persistent.active_claim_count
        if persistent_claims > 0 and state == ServerState.EMPTY:
            state = ServerState.RW
            allocation_count = persistent_claims
        return GetRuntimeStateResponse(
            state=state.name,
            has_rw_session=session.has_rw_session,
            ro_session_count=session.ro_session_count,
            waiting_writers=session.waiting_writers,
            committed=session.committed,
            is_ready=session.is_ready,
            allocation_count=allocation_count,
            memory_layout_hash=self._memory_layout_hash,
        )

    def get_event_history(self) -> GetEventHistoryResponse:
        return GetEventHistoryResponse(events=list(self._events))

    def next_session_id(self) -> str:
        return self._sessions.next_session_id()

    def _has_persistent_claim(
        self,
        conn: Connection,
        engine_id: str,
        tag: str,
    ) -> bool:
        claims = self._persistent_claims_by_session.get(conn.session_id, set())
        return (engine_id, tag) in claims

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
        allocation_ranks_by_id: dict[str, int] = {}
        for rank, info in enumerate(
            sorted(allocations, key=lambda info: info.layout_slot)
        ):
            allocation_ranks_by_id[info.allocation_id] = rank
            h.update(f"{rank}:{info.size}:{info.aligned_size}:{info.tag}".encode())

        for key in sorted(self._metadata):
            entry = self._metadata[key]
            rank = allocation_ranks_by_id[entry.allocation_id]
            h.update(f"{key}:{rank}:{entry.offset_bytes}:".encode())
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
        # Release any persistent claims held by this session. The
        # allocations themselves persist; this only releases the
        # exclusive claim so a new client can attach.
        if conn is not None:
            claims = self._persistent_claims_by_session.pop(
                conn.session_id,
                None,
            )
            if claims:
                for engine_id, tag in claims:
                    self._persistent.unclaim(engine_id, tag)
                logger.info(
                    "Released %d persistent claims on session=%s disconnect",
                    len(claims),
                    conn.session_id,
                )
                self._sync_persistent_layout_events()
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

        # ----------------------------------------------------------------
        # Persistent allocations (KV-pool namespace; lock-state-independent)
        # ----------------------------------------------------------------

        if msg_type is ClaimPersistentAllocationRequest:
            claims = self._persistent_claims_by_session.setdefault(
                conn.session_id, set()
            )
            key = (msg.engine_id, msg.tag)
            try:
                if key in claims and getattr(msg, "shared", False):
                    alloc, reattached = self._persistent.get(*key), True
                else:
                    alloc, reattached = self._persistent.claim(
                        engine_id=msg.engine_id,
                        tag=msg.tag,
                        size=msg.size,
                        shared=getattr(msg, "shared", False),
                    )
            except PersistentClaimConflictError as exc:
                return ErrorResponse(error=str(exc), code=1), -1, False
            except (ValueError, MemoryError) as exc:
                return ErrorResponse(error=str(exc), code=2), -1, False
            # Track for cleanup-on-disconnect: when this session goes
            # away, we'll unclaim every key it owns.
            claims.add(key)
            self._sync_persistent_layout_events()
            return (
                ClaimPersistentAllocationResponse(
                    allocation_id=alloc.allocation_id,
                    size=alloc.size,
                    aligned_size=alloc.aligned_size,
                    reattached=reattached,
                ),
                -1,
                False,
            )

        if msg_type is ReleasePersistentAllocationRequest:
            if not self._has_persistent_claim(conn, msg.engine_id, msg.tag):
                return (
                    ErrorResponse(
                        error="persistent allocation not claimed by session",
                        code=4,
                    ),
                    -1,
                    False,
                )
            try:
                released = self._persistent.release(
                    engine_id=msg.engine_id,
                    tag=msg.tag,
                )
            except PersistentClaimConflictError as exc:
                return ErrorResponse(error=str(exc), code=1), -1, False
            # Drop the claim record too.
            claims = self._persistent_claims_by_session.get(conn.session_id)
            if claims is not None:
                claims.discard((msg.engine_id, msg.tag))
                if not claims:
                    self._persistent_claims_by_session.pop(conn.session_id, None)
            self._sync_persistent_layout_events()
            return ReleasePersistentAllocationResponse(released=released), -1, False

        if msg_type is ExportPersistentAllocationRequest:
            if not self._has_persistent_claim(conn, msg.engine_id, msg.tag):
                return (
                    ErrorResponse(
                        error="persistent allocation not claimed by session",
                        code=4,
                    ),
                    -1,
                    False,
                )
            try:
                alloc, fd = self._persistent.export(
                    engine_id=msg.engine_id,
                    tag=msg.tag,
                )
            except PersistentNotFoundError as exc:
                return ErrorResponse(error=str(exc), code=3), -1, False
            return (
                ExportPersistentAllocationResponse(
                    allocation_id=alloc.allocation_id,
                    size=alloc.size,
                    aligned_size=alloc.aligned_size,
                ),
                fd,
                False,
            )

        if msg_type is ListPersistentAllocationsRequest:
            session_claims = self._persistent_claims_by_session.get(
                conn.session_id, set()
            )
            allocations = [
                a
                for a in self._persistent.list(engine_id=msg.engine_id)
                if (a.engine_id, a.tag) in session_claims
            ]
            return (
                ListPersistentAllocationsResponse(
                    allocations=[
                        PersistentAllocationInfo(
                            allocation_id=a.allocation_id,
                            engine_id=a.engine_id,
                            tag=a.tag,
                            size=a.size,
                            aligned_size=a.aligned_size,
                            claimed=True,
                        )
                        for a in allocations
                    ]
                ),
                -1,
                False,
            )

        raise ValueError(f"Unknown request: {msg_type.__name__}")
