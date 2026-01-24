# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU Memory Service allocator singleton management.

Manages the singleton memory manager and PyTorch MemPool integration.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional, Tuple

from gpu_memory_service.common.types import GrantedLockType, RequestedLockType

if TYPE_CHECKING:
    from gpu_memory_service.client.memory_manager import GMSClientMemoryManager
    from torch.cuda.memory import MemPool

logger = logging.getLogger(__name__)

# Global singleton state
_gms_client_memory_manager: Optional["GMSClientMemoryManager"] = None
_mem_pool: Optional["MemPool"] = None
_pluggable_alloc: Optional[Any] = None


def get_or_create_gms_client_memory_manager(
    socket_path: str,
    device: int,
    mode: RequestedLockType,
    *,
    tag: str = "weights",
    timeout_ms: Optional[int] = None,
) -> Tuple["GMSClientMemoryManager", Optional["MemPool"]]:
    """Get existing memory manager or create a new one.

    Args:
        socket_path: Unix socket path for the allocation server.
        device: CUDA device index.
        mode: RW for cold start, RO for import-only, RW_OR_RO for auto.
        tag: Allocation tag for RW mode.
        timeout_ms: Lock acquisition timeout (None = wait indefinitely).

    Returns:
        (gms_client_memory_manager, pool) - pool is None for RO mode.
    """
    global _gms_client_memory_manager, _mem_pool

    from gpu_memory_service.client.memory_manager import GMSClientMemoryManager

    if _gms_client_memory_manager is not None:
        return _get_existing(mode)

    # Create new manager
    gms_client_memory_manager = GMSClientMemoryManager(
        socket_path, mode=mode, device=device, timeout_ms=timeout_ms
    )
    _gms_client_memory_manager = gms_client_memory_manager

    if gms_client_memory_manager.mode == GrantedLockType.RW:
        _mem_pool = _setup_mempool(gms_client_memory_manager, tag)
        logger.info("[GMS] Created RW allocator (device=%d)", device)
        return gms_client_memory_manager, _mem_pool
    else:
        logger.info("[GMS] Created RO allocator (device=%d)", device)
        return gms_client_memory_manager, None


def _get_existing(
    mode: RequestedLockType,
) -> Tuple["GMSClientMemoryManager", Optional["MemPool"]]:
    """Return existing allocator if mode-compatible."""
    current = _gms_client_memory_manager.mode

    if mode == RequestedLockType.RW:
        if current == GrantedLockType.RW:
            return _gms_client_memory_manager, _mem_pool
        raise RuntimeError(f"Cannot get RW allocator: existing is in {current} mode")

    if mode == RequestedLockType.RO:
        if current == GrantedLockType.RO:
            return _gms_client_memory_manager, None
        raise RuntimeError(
            f"Cannot get RO allocator: existing is in {current} mode. "
            "Call manager.switch_to_read() first."
        )

    # RW_OR_RO: return whatever exists
    pool = _mem_pool if current == GrantedLockType.RW else None
    return _gms_client_memory_manager, pool


def _setup_mempool(
    gms_client_memory_manager: "GMSClientMemoryManager",
    tag: str,
) -> "MemPool":
    """Set up PyTorch CUDAPluggableAllocator and MemPool."""
    global _pluggable_alloc

    from gpu_memory_service.client.torch.extensions import _allocator_ext as cumem
    from torch.cuda import CUDAPluggableAllocator
    from torch.cuda.memory import MemPool

    pluggable_alloc = CUDAPluggableAllocator(cumem.__file__, "my_malloc", "my_free")
    pool = MemPool(allocator=pluggable_alloc.allocator())
    _pluggable_alloc = pluggable_alloc

    def malloc_cb(size: int, device: int, stream: int) -> int:
        va = gms_client_memory_manager.allocate_and_map(int(size), tag=tag)
        logger.debug("[GMS] malloc: va=0x%x size=%d", va, size)
        return va

    def free_cb(ptr: int, size: int, device: int, stream: int) -> None:
        logger.debug("[GMS] free: va=0x%x size=%d", ptr, size)
        gms_client_memory_manager.free_mapping(int(ptr))

    cumem.init_module(malloc_cb, free_cb)
    return pool


def get_gms_client_memory_manager() -> Optional["GMSClientMemoryManager"]:
    """Get the active GMS client memory manager, or None if not initialized."""
    return _gms_client_memory_manager
