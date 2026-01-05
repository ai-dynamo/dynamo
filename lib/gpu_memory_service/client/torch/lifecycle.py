# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU Memory Service lifecycle management.

This module manages the singleton GPU Memory Service memory manager and its integrated
PyTorch MemPool for routing weight allocations through the GPU Memory Service.

Key principles:
- Only one memory manager per process
- Write mode creates MemPool for PyTorch integration
- Read mode (import-only) doesn't need MemPool
- Mode transitions: write → read via switch_to_read() only

Usage:
    # Write mode (cold start):
    manager, pool = get_or_create_allocator(socket_path, device, mode="write")
    with use_mem_pool(pool, device=device):
        # ... load model weights ...
    manager.switch_to_read()

    # Read mode (import-only):
    manager, _ = get_or_create_allocator(socket_path, device, mode="read")
    # ... materialize weights from metadata store ...

    # Consumers (e.g., sleep/wake):
    manager = get_allocator()
    if manager:
        manager.sleep()
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, Optional, Tuple

if TYPE_CHECKING:
    from gpu_memory_service.client.memory_manager import GMSClientMemoryManager
    from torch.cuda.memory import MemPool

logger = logging.getLogger(__name__)

# Global state - one memory manager per process with associated PyTorch components
_allocator: Optional["GMSClientMemoryManager"] = None
_mem_pool: Optional["MemPool"] = None
_pluggable_alloc: Optional[Any] = None  # CUDAPluggableAllocator


def get_or_create_allocator(
    socket_path: str,
    device: int,
    mode: Literal["write", "read"],
    *,
    tag: str = "weights",
    timeout_ms: Optional[int] = None,
) -> Tuple["GMSClientMemoryManager", Optional["MemPool"]]:
    """Get the existing memory manager or create a new one.

    This is the primary way to obtain a memory manager. The module enforces
    that only one memory manager exists per process.

    For write mode:
    - Creates memory manager in write mode
    - Sets up CUDAPluggableAllocator and MemPool for PyTorch integration
    - Initializes malloc/free callbacks

    For read mode:
    - Creates memory manager in read mode (for import-only)
    - No MemPool needed (weights already on GPU)

    Args:
        socket_path: Unix socket path for the allocation server.
        device: CUDA device index.
        mode: Desired mode - "write" for cold start, "read" for import-only.
        tag: Allocation tag for write mode (default: "weights").
        timeout_ms: Timeout in milliseconds for lock acquisition.
                    None means wait indefinitely.

    Returns:
        Tuple of (memory_manager, pool). Pool is None for read mode.

    Raises:
        RuntimeError: If a memory manager exists but is in incompatible mode:
            - Requesting "write" when manager is in read mode (can't go back)
            - Requesting "read" when manager is in write mode (call switch_to_read first)
    """
    global _allocator, _mem_pool, _pluggable_alloc

    from gpu_memory_service.client.memory_manager import GMSClientMemoryManager

    if _allocator is not None:
        # Memory manager already exists - check mode compatibility
        current_mode = _allocator.mode

        if mode == "write":
            if current_mode == "write":
                logger.debug(
                    "[GPU Memory Service] Returning existing write memory manager (device=%d)",
                    device,
                )
                return _allocator, _mem_pool
            else:
                raise RuntimeError(
                    f"Cannot create write memory manager: one already exists in '{current_mode}' mode. "
                    "Only one memory manager per process is allowed."
                )
        else:  # mode == "read"
            if current_mode == "read":
                logger.debug(
                    "[GPU Memory Service] Returning existing read memory manager (device=%d)",
                    device,
                )
                return _allocator, None
            else:
                raise RuntimeError(
                    f"Cannot get read memory manager: current one is in '{current_mode}' mode. "
                    "Call manager.switch_to_read() first to transition to read mode."
                )

    # No memory manager exists - create a new one
    manager = GMSClientMemoryManager(
        socket_path, mode=mode, device=device, timeout_ms=timeout_ms
    )
    _allocator = manager

    if mode == "write":
        # Set up PyTorch integration for write mode
        pool = _setup_pytorch_integration(manager, tag=tag)
        _mem_pool = pool
        logger.debug(
            "[GPU Memory Service] Created write memory manager with MemPool (device=%d, socket=%s)",
            device,
            socket_path,
        )
        return manager, pool
    else:
        # Read mode doesn't need MemPool
        logger.debug(
            "[GPU Memory Service] Created read memory manager (device=%d, socket=%s)",
            device,
            socket_path,
        )
        return manager, None


def _setup_pytorch_integration(
    manager: "GMSClientMemoryManager",
    tag: str = "weights",
) -> "MemPool":
    """Set up PyTorch CUDAPluggableAllocator and MemPool for the memory manager.

    This creates the MemPool that routes allocations through GPU Memory Service and sets up
    the malloc/free callbacks.
    """
    global _pluggable_alloc

    from gpu_memory_service.client.torch.extensions import _allocator_ext as cumem
    from torch.cuda import CUDAPluggableAllocator
    from torch.cuda.memory import MemPool

    # Configure pluggable allocator pool
    so_path = cumem.__file__
    pluggable_alloc = CUDAPluggableAllocator(so_path, "my_malloc", "my_free")
    pool = MemPool(allocator=pluggable_alloc.allocator())
    _pluggable_alloc = pluggable_alloc

    # Create callbacks that route through the memory manager
    _malloc_count = [0]
    _free_count = [0]

    def malloc_cb(
        va: int, size: int, aligned_size: int, device: int, stream: int
    ) -> None:
        """Callback for PyTorch allocations - routes to GPU Memory Service memory manager."""
        manager.allocate_to_va(
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
        mapping = manager._mappings.pop(va, None)
        if mapping is None:
            return
        manager._allocation_id_to_va.pop(mapping.allocation_id, None)
        try:
            manager._client.free(mapping.allocation_id)
        except Exception:
            pass  # Ignore errors during cleanup

    # Initialize the cumem module with our callbacks
    cumem.init_module(malloc_cb, free_cb)

    return pool


def register_allocator(manager: "GMSClientMemoryManager") -> None:
    """Register an externally-created memory manager.

    This is used for the import-only (read mode) path where the memory manager
    is created directly to handle timeout gracefully. On success, it should
    be registered here.

    If a memory manager is already registered, this is a no-op if it's the same
    manager, otherwise raises an error.

    Args:
        manager: The memory manager to register.

    Raises:
        RuntimeError: If a different memory manager is already registered.
    """
    global _allocator

    if _allocator is manager:
        return  # Already registered

    if _allocator is not None:
        raise RuntimeError(
            "Cannot register memory manager: another one is already registered. "
            "Only one memory manager per process is allowed."
        )

    _allocator = manager
    logger.debug(
        "[GPU Memory Service] Registered memory manager (device=%s)", manager.device
    )


def get_allocator() -> Optional["GMSClientMemoryManager"]:
    """Get the active GPU Memory Service memory manager without creating one.

    Returns:
        The memory manager, or None if none exists.
    """
    return _allocator


def get_mem_pool() -> Optional["MemPool"]:
    """Get the MemPool for PyTorch integration.

    Returns:
        The MemPool, or None if not created (read mode or no memory manager).
    """
    return _mem_pool


def clear_allocator() -> None:
    """Clear the memory manager and all associated PyTorch components.

    This should be called during cleanup.
    """
    global _allocator, _mem_pool, _pluggable_alloc

    if _allocator is not None:
        logger.debug(
            "[GPU Memory Service] Cleared memory manager and PyTorch components"
        )
    _allocator = None
    _mem_pool = None
    _pluggable_alloc = None
