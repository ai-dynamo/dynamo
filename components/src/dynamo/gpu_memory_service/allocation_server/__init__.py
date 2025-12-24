"""Allocation Server - Durable GPU memory allocation service with RW/RO locking.

The Allocation Server provides:
1. Physical GPU memory allocation via CUDA VMM (no VA mappings)
2. Connection-based RW/RO locking (socket connection IS the lock)
3. Commit state tracking separate from lock state
4. FD export for SCM_RIGHTS transfer to workers

Key Design Principle: Socket connection IS the lock.
- RW connection: Single writer has exclusive access
- RO connection: Multiple readers can access committed allocations
- Readiness = (no RW connection) AND (committed)

Usage:
    # Start server
    from dynamo.gpu_memory_service.allocation_server import AllocationServer
    server = AllocationServer(socket_path="/run/alloc_server.sock", device=0)
    server.serve_forever()

    # Writer client
    from dynamo.gpu_memory_service.allocation_server import AllocationServerClient
    with AllocationServerClient(socket_path, lock_type="rw") as client:
        alloc_id = client.allocate(size=1024*1024)
        fd = client.export(alloc_id)
        # ... write weights ...
        client.commit()
    # Lock released on exit

    # Reader client
    client = AllocationServerClient(socket_path, lock_type="ro")
    if client.committed:
        alloc_id_to_va = client.import_all()
        # ... use weights ...
    # Keep connection open during inference
    client.close()  # Only when done
"""

from .client import AllocationServerClient
from .rpc_protocol import (
    AllocateRequest,
    AllocateResponse,
    ClearAllRequest,
    ClearAllResponse,
    CommitRequest,
    CommitResponse,
    ExportRequest,
    FreeRequest,
    FreeResponse,
    GetAllocationRequest,
    GetAllocationResponse,
    GetStateRequest,
    GetStateResponse,
    HandshakeRequest,
    HandshakeResponse,
    ListAllocationsRequest,
    ListAllocationsResponse,
)
from .server import AllocationServer

__all__ = [
    "AllocationServer",
    "AllocationServerClient",
    "HandshakeRequest",
    "HandshakeResponse",
    "CommitRequest",
    "CommitResponse",
    "AllocateRequest",
    "AllocateResponse",
    "ExportRequest",
    "GetAllocationRequest",
    "GetAllocationResponse",
    "ListAllocationsRequest",
    "ListAllocationsResponse",
    "FreeRequest",
    "FreeResponse",
    "ClearAllRequest",
    "ClearAllResponse",
    "GetStateRequest",
    "GetStateResponse",
]
