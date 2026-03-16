# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU Memory Service allocator management (singleton).

Manages a single weights memory manager and PyTorch MemPool integration.
Only one GMS scope is needed: weights. KV cache is handled by CuMemAllocator.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional, Tuple

from gpu_memory_service.common.types import GrantedLockType, RequestedLockType

if TYPE_CHECKING:
    from gpu_memory_service.client.memory_manager import GMSClientMemoryManager
    from torch.cuda.memory import MemPool

logger = logging.getLogger(__name__)

# Singleton state
_manager: Optional["GMSClientMemoryManager"] = None
_mem_pool: Optional["MemPool"] = None
_tag: str = "weights"
_callbacks_initialized: bool = False
_pluggable_alloc: Optional[Any] = None


def _gms_malloc(size: int, device: int, stream: int) -> int:
    """Route malloc to the singleton weights manager."""
    if _manager is None:
        raise RuntimeError("No GMS manager initialized")
    va = _manager.create_mapping(size=int(size), tag=_tag)
    logger.debug("[GMS] malloc: va=0x%x size=%d", va, size)
    return va


def _gms_free(ptr: int, size: int, device: int, stream: int) -> None:
    """Route free to the singleton weights manager."""
    if _manager is None:
        logger.warning("[GMS] free: no manager, ignoring va=0x%x", ptr)
        return
    if int(ptr) in _manager.mappings:
        logger.debug("[GMS] free: va=0x%x size=%d", ptr, size)
        _manager.destroy_mapping(int(ptr))
    else:
        logger.warning("[GMS] free: manager does not own va=0x%x, ignoring", ptr)


def _ensure_callbacks_initialized() -> "MemPool":
    """Initialize C-level callbacks exactly once, return a new MemPool."""
    global _callbacks_initialized, _pluggable_alloc

    from gpu_memory_service.client.torch.extensions import _allocator_ext as cumem
    from torch.cuda import CUDAPluggableAllocator
    from torch.cuda.memory import MemPool

    if not _callbacks_initialized:
        _pluggable_alloc = CUDAPluggableAllocator(
            cumem.__file__, "my_malloc", "my_free"
        )
        cumem.init_module(_gms_malloc, _gms_free)
        _callbacks_initialized = True

    return MemPool(allocator=_pluggable_alloc.allocator())


def get_or_create_gms_client_memory_manager(
    socket_path: str,
    device: int,
    mode: RequestedLockType,
    *,
    tag: str = "weights",
    timeout_ms: Optional[int] = None,
) -> Tuple["GMSClientMemoryManager", Optional["MemPool"]]:
    """Get existing memory manager, or create a new one.

    Args:
        socket_path: Unix socket path for the allocation server.
        device: CUDA device index.
        mode: RW for cold start, RO for import-only, RW_OR_RO for auto.
        tag: Allocation tag for RW mode.
        timeout_ms: Lock acquisition timeout (None = wait indefinitely).

    Returns:
        (gms_client_memory_manager, pool) - pool is None for RO mode.
    """
    global _manager, _mem_pool, _tag

    from gpu_memory_service.client.memory_manager import GMSClientMemoryManager

    if _manager is not None:
        return _get_existing(mode)

    manager = GMSClientMemoryManager(socket_path, device=device)
    manager.connect(mode, timeout_ms=timeout_ms)

    if manager.granted_lock_type == GrantedLockType.RW:
        pool = _ensure_callbacks_initialized()
        # Only set globals after mempool succeeds (avoids partial singleton)
        _manager = manager
        _tag = tag
        _mem_pool = pool
        logger.info("[GMS] Created RW allocator (device=%d)", device)
        return manager, pool
    else:
        _manager = manager
        _tag = tag
        logger.info("[GMS] Created RO allocator (device=%d)", device)
        return manager, None


def _get_existing(
    mode: RequestedLockType,
) -> Tuple["GMSClientMemoryManager", Optional["MemPool"]]:
    """Return existing allocator if mode-compatible."""
    assert _manager is not None
    current = _manager.granted_lock_type

    if mode == RequestedLockType.RW:
        if current == GrantedLockType.RW:
            return _manager, _mem_pool
        raise RuntimeError(f"Cannot get RW allocator: existing is in {current} mode")

    if mode == RequestedLockType.RO:
        if current == GrantedLockType.RO:
            return _manager, None
        raise RuntimeError(f"Cannot get RO allocator: existing is in {current} mode")

    # RW_OR_RO: return whatever exists
    effective_pool = _mem_pool if current == GrantedLockType.RW else None
    return _manager, effective_pool


def get_gms_client_memory_manager() -> Optional["GMSClientMemoryManager"]:
    """Get the active GMS client memory manager, or None."""
    return _manager
