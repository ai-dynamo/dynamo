# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request handlers for GPU Memory Service."""

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional

from gpu_memory_service.common.protocol.messages import (
    AllocateRequest,
    AllocateResponse,
    ExportAllocationRequest,
    ExportAllocationResponse,
    FreeAllocationRequest,
    FreeAllocationResponse,
    GetAllocationRequest,
    GetAllocationResponse,
    GetAllocationStateResponse,
    GetLockStateResponse,
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
from gpu_memory_service.common.types import GrantedLockType, derive_state

from .memory_manager import AllocationNotFoundError, GMSServerMemoryManager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetadataEntry:
    allocation_id: str
    offset_bytes: int
    value: bytes
    epoch_id: int


@dataclass
class Epoch:
    id: int
    metadata: dict[str, MetadataEntry] = field(default_factory=dict)


@dataclass
class EpochState:
    next_id: int = 1
    active_rw: Optional[Epoch] = None
    committed: Optional[Epoch] = None
    memory_layout_hash: str = ""


class RequestHandler:
    """Handles allocation and metadata requests."""

    def __init__(
        self,
        device: int = 0,
        *,
        allocation_retry_interval: float = 0.5,
        allocation_retry_timeout: Optional[float] = None,
    ):
        self._memory_manager = GMSServerMemoryManager(
            device,
            allocation_retry_interval=allocation_retry_interval,
            allocation_retry_timeout=allocation_retry_timeout,
        )
        self._epochs = EpochState()
        logger.info(f"RequestHandler initialized: device={device}")

    @property
    def committed_epoch_id(self) -> Optional[int]:
        if self._epochs.committed is None:
            return None
        return self._epochs.committed.id

    @property
    def active_rw_epoch_id(self) -> Optional[int]:
        if self._epochs.active_rw is None:
            return None
        return self._epochs.active_rw.id

    def _require_epoch(self, mode: GrantedLockType) -> Epoch:
        if mode == GrantedLockType.RW:
            if self._epochs.active_rw is None:
                raise AssertionError("RW epoch is not active")
            return self._epochs.active_rw
        if self._epochs.committed is None:
            raise AssertionError("Committed epoch is not available")
        return self._epochs.committed

    def _validate_metadata_target(
        self, epoch: Epoch, allocation_id: str, offset_bytes: int
    ) -> None:
        try:
            info = self._memory_manager.get_allocation(allocation_id, epoch.id)
        except AllocationNotFoundError:
            raise ValueError(
                f"Metadata target allocation does not exist in epoch {epoch.id}: {allocation_id}"
            ) from None

        if offset_bytes < 0:
            raise ValueError(f"offset_bytes must be >= 0, got {offset_bytes}")
        if offset_bytes >= info.aligned_size:
            raise ValueError(
                f"offset_bytes {offset_bytes} out of range for allocation {allocation_id} "
                f"(aligned_size={info.aligned_size})"
            )

    def _drop_metadata_for_allocation(self, epoch: Epoch, allocation_id: str) -> int:
        keys_to_remove = [
            key
            for key, entry in epoch.metadata.items()
            if entry.allocation_id == allocation_id
        ]
        for key in keys_to_remove:
            epoch.metadata.pop(key, None)
        return len(keys_to_remove)

    def _validate_epoch_integrity(self, epoch: Epoch) -> None:
        for key, entry in epoch.metadata.items():
            try:
                info = self._memory_manager.get_allocation(
                    entry.allocation_id, epoch.id
                )
            except AllocationNotFoundError:
                raise AssertionError(
                    f"Metadata key {key!r} references missing allocation "
                    f"{entry.allocation_id!r} in epoch {epoch.id}"
                ) from None

            if entry.offset_bytes < 0 or entry.offset_bytes >= info.aligned_size:
                raise AssertionError(
                    f"Metadata key {key!r} has invalid offset {entry.offset_bytes} "
                    f"for allocation {entry.allocation_id!r} (aligned_size={info.aligned_size})"
                )

    def _compute_memory_layout_hash(self, epoch: Epoch) -> str:
        """Compute hash of allocations + metadata for a single epoch."""
        h = hashlib.sha256()

        allocations = sorted(
            self._memory_manager.list_allocations(epoch.id),
            key=lambda x: x.allocation_id,
        )
        for info in allocations:
            h.update(
                f"{info.allocation_id}:{info.size}:{info.aligned_size}:{info.tag}:{info.epoch_id}".encode()
            )

        for key in sorted(epoch.metadata):
            entry = epoch.metadata[key]
            h.update(
                f"{key}:{entry.allocation_id}:{entry.offset_bytes}:{entry.epoch_id}:".encode()
            )
            h.update(entry.value)

        return h.hexdigest()

    def on_rw_connect(self) -> None:
        """Called after RW lock acquisition."""
        if self._epochs.active_rw is not None:
            raise AssertionError("RW epoch is already active")

        # Any RW epoch invalidates committed visibility.
        if self._epochs.committed is not None:
            old_epoch = self._epochs.committed
            self._memory_manager.clear_all_allocations(old_epoch.id)
            self._epochs.committed = None
            self._epochs.memory_layout_hash = ""
            logger.info(f"RW connected; invalidated committed epoch {old_epoch.id}")

        epoch = Epoch(id=self._epochs.next_id)
        self._epochs.next_id += 1
        self._epochs.active_rw = epoch
        logger.info(f"RW connected; opened active epoch {epoch.id}")

    def on_rw_abort(self) -> None:
        """Called when RW connection closes without commit."""
        epoch = self._epochs.active_rw
        if epoch is None:
            return

        logger.warning(f"RW aborted; clearing active epoch {epoch.id}")
        self._memory_manager.clear_all_allocations(epoch.id)
        self._epochs.active_rw = None

        if self._epochs.committed is None:
            self._epochs.memory_layout_hash = ""

    def on_commit(self) -> None:
        """Called when RW connection commits."""
        epoch = self._require_epoch(GrantedLockType.RW)

        # Commit is the last chance to reject dangling metadata references.
        self._validate_epoch_integrity(epoch)
        self._epochs.memory_layout_hash = self._compute_memory_layout_hash(epoch)

        old_committed = self._epochs.committed
        self._epochs.committed = epoch
        self._epochs.active_rw = None

        # Drop old committed allocations and metadata immediately once replaced.
        if old_committed is not None and old_committed.id != epoch.id:
            self._memory_manager.clear_all_allocations(old_committed.id)

        logger.info(
            f"Committed epoch {epoch.id} with state hash: {self._epochs.memory_layout_hash[:16]}..."
        )

    # ==================== State Queries ====================

    def handle_get_lock_state(
        self,
        has_rw: bool,
        ro_count: int,
        waiting_writers: int,
        committed: bool,
    ) -> GetLockStateResponse:
        """Get lock/session state."""
        state = derive_state(has_rw, ro_count, committed)
        return GetLockStateResponse(
            state=state.value,
            has_rw_session=has_rw,
            ro_session_count=ro_count,
            waiting_writers=waiting_writers,
            committed=committed,
            is_ready=committed and not has_rw,
        )

    def handle_get_allocation_state(self) -> GetAllocationStateResponse:
        """Get allocation state."""
        return GetAllocationStateResponse(
            allocation_count=self._memory_manager.allocation_count
        )

    # ==================== Allocation Operations ====================

    async def handle_allocate(
        self, req: AllocateRequest, is_connected
    ) -> AllocateResponse:
        """Create physical memory allocation in active RW epoch."""
        epoch = self._require_epoch(GrantedLockType.RW)
        info = await self._memory_manager.allocate(
            req.size,
            req.tag,
            epoch_id=epoch.id,
            is_connected=is_connected,
        )
        return AllocateResponse(
            allocation_id=info.allocation_id,
            size=info.size,
            aligned_size=info.aligned_size,
            epoch_id=info.epoch_id,
        )

    def handle_export(
        self, req: ExportAllocationRequest, mode: GrantedLockType
    ) -> tuple[ExportAllocationResponse, int]:
        """Export allocation as POSIX FD.

        Returns (response, fd). Caller must close fd after sending.
        """
        epoch = self._require_epoch(mode)
        allocation_id = req.allocation_id
        try:
            fd = self._memory_manager.export_allocation(allocation_id, epoch.id)
            info = self._memory_manager.get_allocation(allocation_id, epoch.id)
        except AllocationNotFoundError:
            raise ValueError(f"Unknown allocation: {allocation_id}") from None
        response = ExportAllocationResponse(
            allocation_id=info.allocation_id,
            size=info.size,
            aligned_size=info.aligned_size,
            tag=info.tag,
            epoch_id=info.epoch_id,
        )
        return response, fd

    def handle_get_allocation(
        self, req: GetAllocationRequest, mode: GrantedLockType
    ) -> GetAllocationResponse:
        """Get allocation info from the current read epoch."""
        epoch = self._require_epoch(mode)
        try:
            info = self._memory_manager.get_allocation(req.allocation_id, epoch.id)
            return GetAllocationResponse(
                allocation_id=info.allocation_id,
                size=info.size,
                aligned_size=info.aligned_size,
                tag=info.tag,
                epoch_id=info.epoch_id,
            )
        except AllocationNotFoundError:
            raise ValueError(f"Unknown allocation: {req.allocation_id}") from None

    def handle_list_allocations(
        self, req: ListAllocationsRequest, mode: GrantedLockType
    ) -> ListAllocationsResponse:
        """List allocations from the current read epoch."""
        epoch = self._require_epoch(mode)
        allocations = self._memory_manager.list_allocations(epoch.id, req.tag)
        return ListAllocationsResponse(
            allocations=[
                GetAllocationResponse(
                    allocation_id=info.allocation_id,
                    size=info.size,
                    aligned_size=info.aligned_size,
                    tag=info.tag,
                    epoch_id=info.epoch_id,
                )
                for info in allocations
            ]
        )

    def handle_free(self, req: FreeAllocationRequest) -> FreeAllocationResponse:
        """Free single allocation from active RW epoch."""
        epoch = self._require_epoch(GrantedLockType.RW)
        success = self._memory_manager.free_allocation(req.allocation_id, epoch.id)
        if success:
            self._drop_metadata_for_allocation(epoch, req.allocation_id)
        return FreeAllocationResponse(success=success)

    # ==================== Metadata Operations ====================

    def handle_metadata_put(self, req: MetadataPutRequest) -> MetadataPutResponse:
        epoch = self._require_epoch(GrantedLockType.RW)
        self._validate_metadata_target(epoch, req.allocation_id, req.offset_bytes)
        epoch.metadata[req.key] = MetadataEntry(
            req.allocation_id,
            req.offset_bytes,
            req.value,
            epoch.id,
        )
        return MetadataPutResponse(success=True)

    def handle_metadata_get(
        self, req: MetadataGetRequest, mode: GrantedLockType
    ) -> MetadataGetResponse:
        epoch = self._require_epoch(mode)
        entry = epoch.metadata.get(req.key)
        if entry is None:
            return MetadataGetResponse(found=False)
        return MetadataGetResponse(
            found=True,
            allocation_id=entry.allocation_id,
            offset_bytes=entry.offset_bytes,
            value=entry.value,
        )

    def handle_metadata_delete(
        self, req: MetadataDeleteRequest
    ) -> MetadataDeleteResponse:
        epoch = self._require_epoch(GrantedLockType.RW)
        return MetadataDeleteResponse(
            deleted=epoch.metadata.pop(req.key, None) is not None
        )

    def handle_metadata_list(
        self, req: MetadataListRequest, mode: GrantedLockType
    ) -> MetadataListResponse:
        epoch = self._require_epoch(mode)
        epoch_metadata = epoch.metadata
        keys = (
            [k for k in epoch_metadata if k.startswith(req.prefix)]
            if req.prefix
            else list(epoch_metadata)
        )
        return MetadataListResponse(keys=sorted(keys))

    def handle_get_memory_layout_hash(self) -> GetStateHashResponse:
        return GetStateHashResponse(memory_layout_hash=self._epochs.memory_layout_hash)
