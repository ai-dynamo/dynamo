"""Core GPU Memory Service client components."""

from dynamo.gpu_memory_service.core.allocator_lifecycle import (
    clear_allocator,
    get_allocator,
    get_mem_pool,
    get_or_create_allocator,
    register_allocator,
)
from dynamo.gpu_memory_service.core.rpc_cumem_allocator import RPCCumemAllocator

__all__ = [
    "RPCCumemAllocator",
    "get_or_create_allocator",
    "get_allocator",
    "get_mem_pool",
    "register_allocator",
    "clear_allocator",
]
