"""GPU Memory Service allocator lifecycle management.

This module manages the singleton GPU Memory Service allocator and its integrated PyTorch
MemPool for routing weight allocations through the GPU Memory Service.

Key principles:
- Only one allocator per process
- Write mode creates MemPool for PyTorch integration
- Read mode (import-only) doesn't need MemPool
- Mode transitions: write → read via switch_to_read() only

Usage:
    # Write mode (cold start):
    allocator, pool = get_or_create_allocator(socket_path, device, mode="write")
    with use_mem_pool(pool, device=device):
        # ... load model weights ...
    allocator.switch_to_read()

    # Read mode (import-only):
    allocator, _ = get_or_create_allocator(socket_path, device, mode="read")
    # ... materialize weights from registry ...

    # Consumers (e.g., sleep/wake):
    allocator = get_allocator()
    if allocator:
        allocator.sleep()
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, Optional, Tuple

if TYPE_CHECKING:
    from gpu_memory_service.allocator import RPCCumemAllocator
    from torch.cuda.memory import MemPool

logger = logging.getLogger(__name__)

# Global state - one allocator per process with associated PyTorch components
_allocator: Optional["RPCCumemAllocator"] = None
_mem_pool: Optional["MemPool"] = None
_pluggable_alloc: Optional[Any] = None  # CUDAPluggableAllocator


def get_or_create_allocator(
    socket_path: str,
    device: int,
    mode: Literal["write", "read"],
    *,
    timeout_ms: Optional[int] = None,
    tag: str = "weights",
) -> Tuple["RPCCumemAllocator", Optional["MemPool"]]:
    """Get the existing allocator or create a new one.

    This is the primary way to obtain an allocator. The module enforces
    that only one allocator exists per process.

    For write mode:
    - Creates allocator in write mode
    - Sets up CUDAPluggableAllocator and MemPool for PyTorch integration
    - Initializes malloc/free callbacks

    For read mode:
    - Creates allocator in read mode (for import-only)
    - No MemPool needed (weights already on GPU)

    Args:
        socket_path: Unix socket path for the allocation server.
        device: CUDA device index.
        mode: Desired mode - "write" for cold start, "read" for import-only.
        timeout_ms: Connection timeout (default: None = wait forever).
        tag: Allocation tag for write mode (default: "weights").

    Returns:
        Tuple of (allocator, pool). Pool is None for read mode.

    Raises:
        RuntimeError: If an allocator exists but is in incompatible mode:
            - Requesting "write" when allocator is in read mode (can't go back)
            - Requesting "read" when allocator is in write mode (call switch_to_read first)
    """
    global _allocator, _mem_pool, _pluggable_alloc

    from gpu_memory_service.allocator import RPCCumemAllocator

    if _allocator is not None:
        # Allocator already exists - check mode compatibility
        current_mode = _allocator.mode

        if mode == "write":
            if current_mode == "write":
                logger.debug(
                    "[GPU Memory Service] Returning existing write allocator (device=%d)",
                    device,
                )
                return _allocator, _mem_pool
            else:
                raise RuntimeError(
                    f"Cannot create write allocator: allocator already exists in '{current_mode}' mode. "
                    "Only one allocator per process is allowed."
                )
        else:  # mode == "read"
            if current_mode == "read":
                logger.debug(
                    "[GPU Memory Service] Returning existing read allocator (device=%d)",
                    device,
                )
                return _allocator, None
            else:
                raise RuntimeError(
                    f"Cannot get read allocator: allocator is in '{current_mode}' mode. "
                    "Call allocator.switch_to_read() first to transition to read mode."
                )

    # No allocator exists - create a new one
    allocator = RPCCumemAllocator(
        socket_path, mode=mode, device=device, timeout_ms=timeout_ms
    )
    _allocator = allocator

    if mode == "write":
        # Set up PyTorch integration for write mode
        pool = _setup_pytorch_integration(allocator, tag=tag)
        _mem_pool = pool
        logger.debug(
            "[GPU Memory Service] Created write allocator with MemPool (device=%d, socket=%s)",
            device,
            socket_path,
        )
        return allocator, pool
    else:
        # Read mode doesn't need MemPool
        logger.debug(
            "[GPU Memory Service] Created read allocator (device=%d, socket=%s)",
            device,
            socket_path,
        )
        return allocator, None


def _setup_pytorch_integration(
    allocator: "RPCCumemAllocator",
    tag: str = "weights",
) -> "MemPool":
    """Set up PyTorch CUDAPluggableAllocator and MemPool for the allocator.

    This creates the MemPool that routes allocations through GPU Memory Service and sets up
    the malloc/free callbacks.
    """
    global _pluggable_alloc

    from gpu_memory_service.extensions import _rpc_cumem_ext as cumem
    from torch.cuda import CUDAPluggableAllocator
    from torch.cuda.memory import MemPool

    # Configure pluggable allocator pool
    so_path = cumem.__file__
    pluggable_alloc = CUDAPluggableAllocator(so_path, "my_malloc", "my_free")
    pool = MemPool(allocator=pluggable_alloc.allocator())
    _pluggable_alloc = pluggable_alloc

    # Create callbacks that route through the allocator
    _malloc_count = [0]
    _free_count = [0]

    def malloc_cb(
        va: int, size: int, aligned_size: int, device: int, stream: int
    ) -> None:
        """Callback for PyTorch allocations - routes to GPU Memory Service allocator."""
        allocator.allocate_to_va(
            int(size), int(va), int(aligned_size), int(device), tag=tag
        )
        _malloc_count[0] += 1
        logger.debug(
            "[GPU Memory Service] malloc_cb #%d: va=0x%x size=%d aligned=%d device=%d",
            _malloc_count[0],
            va,
            size,
            aligned_size,
            device,
        )

    def free_cb(va: int) -> None:
        """Callback for PyTorch frees - handles during write phase."""
        _free_count[0] += 1
        logger.warning(
            "[GPU Memory Service] free_cb #%d called for va=0x%x - memory being unmapped",
            _free_count[0],
            va,
        )

        # Handle frees during write phase (rare but possible)
        va = int(va)
        mapping = allocator._mappings.pop(va, None)
        if mapping is None:
            return
        allocator._allocation_id_to_va.pop(mapping.allocation_id, None)
        try:
            allocator._client.free(mapping.allocation_id)
        except Exception:
            pass  # Ignore errors during cleanup

    # Initialize the cumem module with our callbacks
    cumem.init_module(malloc_cb, free_cb)

    return pool


def register_allocator(allocator: "RPCCumemAllocator") -> None:
    """Register an externally-created allocator.

    This is used for the import-only (read mode) path where the allocator
    is created directly to handle timeout gracefully. On success, it should
    be registered here.

    If an allocator is already registered, this is a no-op if it's the same
    allocator, otherwise raises an error.

    Args:
        allocator: The allocator to register.

    Raises:
        RuntimeError: If a different allocator is already registered.
    """
    global _allocator

    if _allocator is allocator:
        return  # Already registered

    if _allocator is not None:
        raise RuntimeError(
            "Cannot register allocator: another allocator is already registered. "
            "Only one allocator per process is allowed."
        )

    _allocator = allocator
    logger.debug(
        "[GPU Memory Service] Registered allocator (device=%s)", allocator.device
    )


def get_allocator() -> Optional["RPCCumemAllocator"]:
    """Get the active GPU Memory Service allocator without creating one.

    Returns:
        The allocator, or None if no allocator exists.
    """
    return _allocator


def get_mem_pool() -> Optional["MemPool"]:
    """Get the MemPool for PyTorch integration.

    Returns:
        The MemPool, or None if not created (read mode or no allocator).
    """
    return _mem_pool


def clear_allocator() -> None:
    """Clear the allocator and all associated PyTorch components.

    This should be called during cleanup.
    """
    global _allocator, _mem_pool, _pluggable_alloc

    if _allocator is not None:
        logger.debug("[GPU Memory Service] Cleared allocator and PyTorch components")
    _allocator = None
    _mem_pool = None
    _pluggable_alloc = None
