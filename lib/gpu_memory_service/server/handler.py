# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request handlers for GPU Memory Service."""

import hashlib
import logging
from dataclasses import dataclass

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


class RequestHandler:
    """Handles allocation and metadata requests."""

    def __init__(self, device: int = 0):
        self._memory_manager = GMSServerMemoryManager(device)
        self._metadata: dict[str, MetadataEntry] = {}
        self._memory_layout_hash: str = (
            ""  # Hash of allocations + metadata, computed on commit
        )
        logger.info(f"RequestHandler initialized: device={device}")

    @property
    def granularity(self) -> int:
        return self._memory_manager.granularity

    def on_rw_abort(self) -> None:
        """Called when RW connection closes without commit."""
        logger.warning("RW aborted; clearing allocations and metadata")
        self._memory_manager.clear_all()
        self._metadata.clear()
        self._memory_layout_hash = ""

    def on_commit(self) -> None:
        """Called when RW connection commits. Computes state hash."""
        self._memory_layout_hash = self._compute_memory_layout_hash()
        logger.info(f"Committed with state hash: {self._memory_layout_hash[:16]}...")

    def _compute_memory_layout_hash(self) -> str:
        """Compute hash of current allocations + metadata."""
        h = hashlib.sha256()
        # Hash allocations (sorted by ID for determinism)
        for info in sorted(
            self._memory_manager.list_allocations(), key=lambda x: x.allocation_id
        ):
            h.update(
                f"{info.allocation_id}:{info.size}:{info.aligned_size}:{info.tag}".encode()
            )
        # Hash metadata (sorted by key for determinism)
        for key in sorted(self._metadata.keys()):
            entry = self._metadata[key]
            h.update(f"{key}:{entry.allocation_id}:{entry.offset_bytes}:".encode())
            h.update(entry.value)
        return h.hexdigest()

    def on_shutdown(self) -> None:
        """Called on server shutdown."""
        if self._memory_manager.allocation_count > 0:
            count = self._memory_manager.clear_all()
            self._metadata.clear()
            logger.info(f"Released {count} GPU allocations during shutdown")

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
        """Create physical memory allocation.

        Requires RW connection (enforced by server).
        """
        info = self._memory_manager.allocate(req.size, req.tag)
        return AllocateResponse(
            allocation_id=info.allocation_id,
            size=info.size,
            aligned_size=info.aligned_size,
        )

    def handle_export(self, allocation_id: str) -> tuple[GetAllocationResponse, int]:
        """Export allocation as POSIX FD.

        Returns (response, fd). Caller must close fd after sending.
        """
        fd = self._memory_manager.export_fd(allocation_id)
        info = self._memory_manager.get_allocation(allocation_id)
        response = GetAllocationResponse(
            allocation_id=info.allocation_id,
            size=info.size,
            aligned_size=info.aligned_size,
            tag=info.tag,
        )
        return response, fd

    def handle_get_allocation(self, req: GetAllocationRequest) -> GetAllocationResponse:
        """Get allocation info."""
        try:
            info = self._memory_manager.get_allocation(req.allocation_id)
            return GetAllocationResponse(
                allocation_id=info.allocation_id,
                size=info.size,
                aligned_size=info.aligned_size,
                tag=info.tag,
            )
        except AllocationNotFoundError:
            raise ValueError(f"Unknown allocation: {req.allocation_id}") from None

    def handle_list_allocations(
        self, req: ListAllocationsRequest
    ) -> ListAllocationsResponse:
        """List all allocations."""
        allocations = self._memory_manager.list_allocations(req.tag)
        result = [
            {
                "allocation_id": info.allocation_id,
                "size": info.size,
                "aligned_size": info.aligned_size,
                "tag": info.tag,
            }
            for info in allocations
        ]
        return ListAllocationsResponse(allocations=result)

    def handle_free(self, req: FreeRequest) -> FreeResponse:
        """Free single allocation.

        Requires RW connection (enforced by server).
        """
        success = self._memory_manager.free(req.allocation_id)
        return FreeResponse(success=success)

    def handle_clear_all(self) -> ClearAllResponse:
        """Clear all allocations and metadata.

        Requires RW connection (enforced by server).
        """
        count = self._memory_manager.clear_all()
        self._metadata.clear()
        return ClearAllResponse(cleared_count=count)

    # ==================== Metadata Operations ====================

    def handle_metadata_put(self, req: MetadataPutRequest) -> MetadataPutResponse:
        self._metadata[req.key] = MetadataEntry(
            req.allocation_id, req.offset_bytes, req.value
        )
        return MetadataPutResponse(success=True)

    def handle_metadata_get(self, req: MetadataGetRequest) -> MetadataGetResponse:
        entry = self._metadata.get(req.key)
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
        return MetadataDeleteResponse(
            deleted=self._metadata.pop(req.key, None) is not None
        )

    def handle_metadata_list(self, req: MetadataListRequest) -> MetadataListResponse:
        keys = (
            [k for k in self._metadata if k.startswith(req.prefix)]
            if req.prefix
            else list(self._metadata)
        )
        return MetadataListResponse(keys=sorted(keys))

    def handle_get_memory_layout_hash(self) -> GetStateHashResponse:
        return GetStateHashResponse(memory_layout_hash=self._memory_layout_hash)
