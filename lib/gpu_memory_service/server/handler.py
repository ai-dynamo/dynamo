# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request handlers for GPU Memory Service."""

import hashlib
import logging
from dataclasses import dataclass
from typing import Optional

from gpu_memory_service.common.protocol.messages import (
    AllocateRequest,
    AllocateResponse,
    ClearAllResponse,
    FreeRequest,
    FreeResponse,
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
from gpu_memory_service.common.types import derive_state

from .memory_manager import AllocationNotFoundError, GMSServerMemoryManager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetadataEntry:
    allocation_id: str
    offset_bytes: int
    value: bytes
    epoch_id: str


class RequestHandler:
    """Handles allocation and metadata requests."""

    def __init__(self, device: int = 0):
        self._memory_manager = GMSServerMemoryManager(device)
        self._metadata_by_epoch: dict[str, dict[str, MetadataEntry]] = {}
        self._committed_epoch_id: Optional[str] = None
        self._active_rw_epoch_id: Optional[str] = None
        self._next_epoch_index: int = 0
        self._memory_layout_hash: str = ""
        logger.info(f"RequestHandler initialized: device={device}")

    @property
    def granularity(self) -> int:
        return self._memory_manager.granularity

    @property
    def committed_epoch_id(self) -> Optional[str]:
        return self._committed_epoch_id

    @property
    def active_rw_epoch_id(self) -> Optional[str]:
        return self._active_rw_epoch_id

    def _new_epoch_id(self) -> str:
        self._next_epoch_index += 1
        return f"epoch_{self._next_epoch_index}"

    def _require_active_rw_epoch(self) -> str:
        if self._active_rw_epoch_id is None:
            raise RuntimeError("RW epoch is not active")
        return self._active_rw_epoch_id

    def _validate_metadata_target(
        self, epoch_id: str, allocation_id: str, offset_bytes: int
    ) -> None:
        try:
            info = self._memory_manager.get_allocation(allocation_id, epoch_id=epoch_id)
        except AllocationNotFoundError:
            raise ValueError(
                f"Metadata target allocation does not exist in epoch {epoch_id}: {allocation_id}"
            ) from None

        if offset_bytes < 0:
            raise ValueError(f"offset_bytes must be >= 0, got {offset_bytes}")
        if offset_bytes >= info.aligned_size:
            raise ValueError(
                f"offset_bytes {offset_bytes} out of range for allocation {allocation_id} "
                f"(aligned_size={info.aligned_size})"
            )

    def _drop_metadata_for_allocation(self, epoch_id: str, allocation_id: str) -> int:
        epoch_metadata = self._metadata_by_epoch.get(epoch_id)
        if not epoch_metadata:
            return 0

        keys_to_remove = [
            key
            for key, entry in epoch_metadata.items()
            if entry.allocation_id == allocation_id
        ]
        for key in keys_to_remove:
            epoch_metadata.pop(key, None)
        return len(keys_to_remove)

    def _validate_epoch_integrity(self, epoch_id: str) -> None:
        epoch_metadata = self._metadata_by_epoch.get(epoch_id, {})
        for key, entry in epoch_metadata.items():
            try:
                info = self._memory_manager.get_allocation(
                    entry.allocation_id, epoch_id=epoch_id
                )
            except AllocationNotFoundError:
                raise RuntimeError(
                    f"Metadata key {key!r} references missing allocation "
                    f"{entry.allocation_id!r} in epoch {epoch_id}"
                ) from None

            if entry.offset_bytes < 0 or entry.offset_bytes >= info.aligned_size:
                raise RuntimeError(
                    f"Metadata key {key!r} has invalid offset {entry.offset_bytes} "
                    f"for allocation {entry.allocation_id!r} (aligned_size={info.aligned_size})"
                )

    def _compute_memory_layout_hash(self, epoch_id: str) -> str:
        """Compute hash of allocations + metadata for a single epoch."""
        h = hashlib.sha256()

        allocations = sorted(
            self._memory_manager.list_allocations(epoch_id=epoch_id),
            key=lambda x: x.allocation_id,
        )
        for info in allocations:
            h.update(
                f"{info.allocation_id}:{info.size}:{info.aligned_size}:{info.tag}:{info.epoch_id}".encode()
            )

        metadata = self._metadata_by_epoch.get(epoch_id, {})
        for key in sorted(metadata.keys()):
            entry = metadata[key]
            h.update(
                f"{key}:{entry.allocation_id}:{entry.offset_bytes}:{entry.epoch_id}:".encode()
            )
            h.update(entry.value)

        return h.hexdigest()

    def on_rw_connect(self) -> None:
        """Called after RW lock acquisition."""
        if self._active_rw_epoch_id is not None:
            raise RuntimeError("RW epoch is already active")

        # Any RW epoch invalidates committed visibility.
        if self._committed_epoch_id is not None:
            old_epoch = self._committed_epoch_id
            self._memory_manager.clear_epoch(old_epoch)
            self._metadata_by_epoch.pop(old_epoch, None)
            self._committed_epoch_id = None
            self._memory_layout_hash = ""
            logger.info(f"RW connected; invalidated committed epoch {old_epoch}")

        epoch_id = self._new_epoch_id()
        self._active_rw_epoch_id = epoch_id
        self._metadata_by_epoch[epoch_id] = {}
        logger.info(f"RW connected; opened active epoch {epoch_id}")

    def on_rw_abort(self) -> None:
        """Called when RW connection closes without commit."""
        epoch_id = self._active_rw_epoch_id
        if epoch_id is None:
            return

        logger.warning(f"RW aborted; clearing active epoch {epoch_id}")
        self._memory_manager.clear_epoch(epoch_id)
        self._metadata_by_epoch.pop(epoch_id, None)
        self._active_rw_epoch_id = None

        if self._committed_epoch_id is None:
            self._memory_layout_hash = ""

    def on_commit(self) -> None:
        """Called when RW connection commits."""
        epoch_id = self._require_active_rw_epoch()

        # Commit is the last chance to reject dangling metadata references.
        self._validate_epoch_integrity(epoch_id)
        self._memory_layout_hash = self._compute_memory_layout_hash(epoch_id)

        old_committed = self._committed_epoch_id
        self._committed_epoch_id = epoch_id
        self._active_rw_epoch_id = None

        # Drop old committed allocations and metadata immediately once replaced.
        if old_committed is not None and old_committed != epoch_id:
            self._memory_manager.clear_epoch(old_committed)
            self._metadata_by_epoch.pop(old_committed, None)

        logger.info(
            f"Committed epoch {epoch_id} with state hash: {self._memory_layout_hash[:16]}..."
        )

    def on_shutdown(self) -> None:
        """Called on server shutdown."""
        if self._memory_manager.allocation_count > 0:
            count = self._memory_manager.clear_all()
            logger.info(f"Released {count} GPU allocations during shutdown")

        self._metadata_by_epoch.clear()
        self._committed_epoch_id = None
        self._active_rw_epoch_id = None
        self._memory_layout_hash = ""

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
            allocation_count=self._memory_manager.allocation_count,
            total_bytes=self._memory_manager.total_bytes,
        )

    # ==================== Allocation Operations ====================

    def handle_allocate(self, req: AllocateRequest) -> AllocateResponse:
        """Create physical memory allocation in active RW epoch."""
        epoch_id = self._require_active_rw_epoch()
        info = self._memory_manager.allocate(req.size, req.tag, epoch_id=epoch_id)
        return AllocateResponse(
            allocation_id=info.allocation_id,
            size=info.size,
            aligned_size=info.aligned_size,
            epoch_id=info.epoch_id,
        )

    def handle_export(
        self, allocation_id: str, epoch_id: str
    ) -> tuple[GetAllocationResponse, int]:
        """Export allocation as POSIX FD.

        Returns (response, fd). Caller must close fd after sending.
        """
        fd = self._memory_manager.export_fd(allocation_id, epoch_id=epoch_id)
        info = self._memory_manager.get_allocation(allocation_id, epoch_id=epoch_id)
        response = GetAllocationResponse(
            allocation_id=info.allocation_id,
            size=info.size,
            aligned_size=info.aligned_size,
            tag=info.tag,
            epoch_id=info.epoch_id,
        )
        return response, fd

    def handle_get_allocation(
        self, req: GetAllocationRequest, epoch_id: str
    ) -> GetAllocationResponse:
        """Get allocation info for a specific epoch."""
        try:
            info = self._memory_manager.get_allocation(
                req.allocation_id, epoch_id=epoch_id
            )
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
        self, req: ListAllocationsRequest, epoch_id: str
    ) -> ListAllocationsResponse:
        """List allocations for a specific epoch."""
        allocations = self._memory_manager.list_allocations(req.tag, epoch_id=epoch_id)
        result = [
            {
                "allocation_id": info.allocation_id,
                "size": info.size,
                "aligned_size": info.aligned_size,
                "tag": info.tag,
                "epoch_id": info.epoch_id,
            }
            for info in allocations
        ]
        return ListAllocationsResponse(allocations=result)

    def handle_free(self, req: FreeRequest) -> FreeResponse:
        """Free single allocation from active RW epoch."""
        epoch_id = self._require_active_rw_epoch()
        success = self._memory_manager.free(req.allocation_id, epoch_id=epoch_id)
        if success:
            self._drop_metadata_for_allocation(epoch_id, req.allocation_id)
        return FreeResponse(success=success)

    def handle_clear_all(self) -> ClearAllResponse:
        """Clear allocations and metadata in active RW epoch."""
        epoch_id = self._require_active_rw_epoch()
        count = self._memory_manager.clear_epoch(epoch_id)
        self._metadata_by_epoch[epoch_id] = {}
        return ClearAllResponse(cleared_count=count)

    # ==================== Metadata Operations ====================

    def handle_metadata_put(self, req: MetadataPutRequest) -> MetadataPutResponse:
        epoch_id = self._require_active_rw_epoch()
        self._validate_metadata_target(epoch_id, req.allocation_id, req.offset_bytes)
        epoch_metadata = self._metadata_by_epoch.setdefault(epoch_id, {})
        epoch_metadata[req.key] = MetadataEntry(
            req.allocation_id,
            req.offset_bytes,
            req.value,
            epoch_id,
        )
        return MetadataPutResponse(success=True)

    def handle_metadata_get(
        self, req: MetadataGetRequest, epoch_id: str
    ) -> MetadataGetResponse:
        epoch_metadata = self._metadata_by_epoch.get(epoch_id, {})
        entry = epoch_metadata.get(req.key)
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
        epoch_id = self._require_active_rw_epoch()
        epoch_metadata = self._metadata_by_epoch.setdefault(epoch_id, {})
        return MetadataDeleteResponse(
            deleted=epoch_metadata.pop(req.key, None) is not None
        )

    def handle_metadata_list(
        self, req: MetadataListRequest, epoch_id: str
    ) -> MetadataListResponse:
        epoch_metadata = self._metadata_by_epoch.get(epoch_id, {})
        keys = (
            [k for k in epoch_metadata if k.startswith(req.prefix)]
            if req.prefix
            else list(epoch_metadata)
        )
        return MetadataListResponse(keys=sorted(keys))

    def handle_get_memory_layout_hash(self) -> GetStateHashResponse:
        return GetStateHashResponse(memory_layout_hash=self._memory_layout_hash)
