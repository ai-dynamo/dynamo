"""Request handlers for GPU Memory Service.

Stateless handlers for allocation operations, metadata operations, and state
queries. Separated from server to keep lifecycle/connection management distinct.
"""

import logging

from gpu_memory_service.common.protocol import (
    AllocateRequest,
    AllocateResponse,
    ClearAllResponse,
    FreeRequest,
    FreeResponse,
    GetAllocationRequest,
    GetAllocationResponse,
    GetAllocationStateResponse,
    GetLockStateResponse,
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
from .metadata_store import GMSMetadataStore

logger = logging.getLogger(__name__)


class RequestHandler:
    """Handles allocation and metadata requests.

    This class contains all the business logic handlers, separate from
    connection/lifecycle management. It is NOT thread-safe - the async
    server ensures single-threaded access.
    """

    def __init__(self, device: int = 0):
        """Initialize handler with memory manager and metadata store.

        Args:
            device: CUDA device ID for allocations
        """
        self._memory_manager = GMSServerMemoryManager(device)
        self._metadata_store = GMSMetadataStore()

        logger.info(
            f"RequestHandler initialized: device={device}, "
            f"granularity={self._memory_manager.granularity}"
        )

    @property
    def granularity(self) -> int:
        """VMM allocation granularity."""
        return self._memory_manager.granularity

    @property
    def metadata_store(self) -> GMSMetadataStore:
        """Metadata store."""
        return self._metadata_store

    # ==================== Lifecycle Callbacks ====================

    def on_rw_abort(self) -> None:
        """Called when RW connection closes without commit.

        Clears all allocations and metadata.
        """
        logger.warning("RW aborted; clearing allocations and metadata")
        self._memory_manager.clear_all()
        self._metadata_store.clear()

    def on_shutdown(self) -> None:
        """Called on server shutdown.

        Releases all GPU memory.
        """
        if self._memory_manager.allocation_count > 0:
            count = self._memory_manager.clear_all()
            self._metadata_store.clear()
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
            raise ValueError(f"Unknown allocation: {req.allocation_id}")

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
        """Clear all allocations.

        Requires RW connection (enforced by server).
        """
        count = self._memory_manager.clear_all()
        return ClearAllResponse(cleared_count=count)

    # ==================== Metadata Operations ====================

    def handle_metadata_put(self, req: MetadataPutRequest) -> MetadataPutResponse:
        """Put metadata entry.

        Requires RW connection (enforced by server).
        """
        self._metadata_store.put(
            key=req.key,
            allocation_id=req.allocation_id,
            offset_bytes=req.offset_bytes,
            value=req.value,
        )
        return MetadataPutResponse(success=True)

    def handle_metadata_get(self, req: MetadataGetRequest) -> MetadataGetResponse:
        """Get metadata entry."""
        entry = self._metadata_store.get(req.key)
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
        """Delete metadata entry.

        Requires RW connection (enforced by server).
        """
        deleted = self._metadata_store.delete(req.key)
        return MetadataDeleteResponse(deleted=deleted)

    def handle_metadata_list(self, req: MetadataListRequest) -> MetadataListResponse:
        """List metadata keys."""
        keys = self._metadata_store.list_keys(req.prefix)
        return MetadataListResponse(keys=keys)
