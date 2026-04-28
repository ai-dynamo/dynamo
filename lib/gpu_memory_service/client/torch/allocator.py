# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU Memory Service allocator registry for PyTorch integration.

Two manager flavors share the same tagged-mempool dispatch:
- GMSClientMemoryManager: server-backed (used for weights, anything that
  needs cross-engine import).
- ClientLocalMemoryManager: pure client-local CUDA VMM (used for kv_cache,
  whether shadow-failover scratch or plain enable_sleep_mode buffers).

Both implement the same minimal interface — create_mapping, destroy_mapping,
unmap_all_vas, remap_all_vas, is_unmapped, mappings, abort — so the
allocator's malloc/free dispatch is content-blind.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Optional, Union

from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType

if TYPE_CHECKING:
    import torch
    from gpu_memory_service.client.client_local_memory_manager import (
        ClientLocalMemoryManager,
    )
    from gpu_memory_service.client.memory_manager import GMSClientMemoryManager
    from torch.cuda.memory import MemPool

    AnyManager = Union[GMSClientMemoryManager, ClientLocalMemoryManager]

logger = logging.getLogger(__name__)


@dataclass
class _TagState:
    manager: "AnyManager"
    mem_pool: "MemPool | None"
    device: int


_tag_states: dict[str, _TagState] = {}
_active_tag: ContextVar[str | None] = ContextVar(
    "gpu_memory_service_active_tag",
    default=None,
)
_callbacks_initialized = False
_pluggable_alloc: Any | None = None


def _gms_malloc(size: int, device: int, stream: int) -> int:
    # Active tag (set by gms_use_mem_pool) selects the manager; the manager's
    # create_mapping is virtual — server-backed or client-local depending on
    # which class was registered for this tag.
    tag = _active_tag.get()
    if tag is None:
        raise RuntimeError("No active GMS allocation tag")
    state = _tag_states.get(tag)
    if state is None:
        raise RuntimeError(f"Unknown GMS allocation tag: {tag}")
    va = state.manager.create_mapping(size=int(size), tag=tag)
    logger.debug("[GMS] malloc(tag=%s): va=0x%x size=%d", tag, va, size)
    return va


def _gms_free(ptr: int, size: int, device: int, stream: int) -> None:
    # Content-driven dispatch: torch only gives us a VA, no tag context.
    # Each manager's destroy_mapping returns True iff it owned the VA.
    va = int(ptr)
    for tag, state in _tag_states.items():
        if state.manager.destroy_mapping(va):
            logger.debug("[GMS] free(tag=%s): va=0x%x size=%d", tag, va, size)
            return
    logger.warning("[GMS] free: no manager owns va=0x%x, ignoring", va)


def _ensure_callbacks_initialized() -> None:
    global _callbacks_initialized, _pluggable_alloc

    from gpu_memory_service.client.torch.extensions import _allocator_ext as cumem
    from torch.cuda import CUDAPluggableAllocator

    if _callbacks_initialized:
        return

    _pluggable_alloc = CUDAPluggableAllocator(cumem.__file__, "my_malloc", "my_free")
    cumem.init_module(_gms_malloc, _gms_free)
    _callbacks_initialized = True


def _create_mem_pool() -> "MemPool":
    from torch.cuda.memory import MemPool

    assert _pluggable_alloc is not None
    return MemPool(allocator=_pluggable_alloc.allocator())


def get_or_create_gms_client_memory_manager(
    socket_path: str,
    device: int,
    mode: RequestedLockType,
    *,
    tag: str = "weights",
    timeout_ms: Optional[int] = None,
) -> "GMSClientMemoryManager":
    """Construct + register a server-backed manager."""
    from gpu_memory_service.client.memory_manager import GMSClientMemoryManager

    state = _tag_states.get(tag)
    if state is not None:
        manager = state.manager
        if not isinstance(manager, GMSClientMemoryManager):
            raise RuntimeError(
                f"GMS allocator tag={tag} is registered as non-server-backed; "
                "cannot reuse via get_or_create_gms_client_memory_manager"
            )
        if manager.socket_path != socket_path or state.device != device:
            raise RuntimeError(
                f"GMS allocator tag={tag} was initialized for "
                f"{manager.socket_path} on device {state.device}, not "
                f"{socket_path} on device {device}"
            )
        if not manager.is_connected:
            if manager.mappings or manager.is_unmapped or manager.granted_lock_type:
                raise RuntimeError(
                    f"GMS allocator tag={tag} is disconnected but still owns "
                    "preserved state; recreate the process instead of reusing it"
                )
            manager._client = None
            manager._granted_lock_type = None
            _tag_states.pop(tag, None)
            state = None

    if state is not None:
        current = state.manager.granted_lock_type
        if mode == RequestedLockType.RW and current != GrantedLockType.RW:
            raise RuntimeError(
                f"Cannot get RW allocator for tag {tag}: existing is in {current} mode"
            )
        if mode == RequestedLockType.RO and current != GrantedLockType.RO:
            raise RuntimeError(
                f"Cannot get RO allocator for tag {tag}: existing is in {current} mode"
            )
        return state.manager

    manager = GMSClientMemoryManager(socket_path, device=device, tag=tag)
    manager.connect(mode, timeout_ms=timeout_ms)

    # Mempool only when we have RW: the pluggable allocator routes torch
    # allocations through us, and only RW clients are allowed to allocate.
    # RO clients consume preserved imports and don't use the mempool.
    mem_pool = None
    if manager.granted_lock_type == GrantedLockType.RW:
        _ensure_callbacks_initialized()
        mem_pool = _create_mem_pool()

    _tag_states[tag] = _TagState(
        manager=manager,
        mem_pool=mem_pool,
        device=device,
    )
    logger.info(
        "[GMS] Created %s allocator for tag=%s (device=%d)",
        manager.granted_lock_type.value,
        tag,
        device,
    )
    return manager


def get_or_create_client_local_manager(
    device: int,
    *,
    tag: str = "kv_cache",
    aliased: bool = False,
) -> "ClientLocalMemoryManager":
    """Construct + register a client-local manager (no GMS server).

    aliased=True  installs scratch-aliased physical at create_mapping (one
                  granularity-sized handle aliased N times across the VA).
                  remap_all_vas always allocates real per-tensor backing,
                  collapsing aliased state on first wake.
    aliased=False installs full-size real physical at create_mapping.
    """
    from gpu_memory_service.client.client_local_memory_manager import (
        ClientLocalMemoryManager,
    )

    state = _tag_states.get(tag)
    if state is not None:
        manager = state.manager
        if not isinstance(manager, ClientLocalMemoryManager):
            raise RuntimeError(
                f"GMS allocator tag={tag} is already registered as server-backed; "
                "cannot reuse via get_or_create_client_local_manager"
            )
        if state.device != device or manager.aliased != aliased:
            raise RuntimeError(
                f"GMS allocator tag={tag} was initialized for "
                f"device={state.device} aliased={manager.aliased}; "
                f"got device={device} aliased={aliased}"
            )
        return manager

    manager = ClientLocalMemoryManager(device=device, aliased=aliased, tag=tag)
    _ensure_callbacks_initialized()
    mem_pool = _create_mem_pool()

    _tag_states[tag] = _TagState(
        manager=manager,
        mem_pool=mem_pool,
        device=device,
    )
    logger.info(
        "[GMS] Registered client-local allocator for tag=%s (device=%d, aliased=%s)",
        tag,
        device,
        aliased,
    )
    return manager


def get_gms_client_memory_manager(
    tag: str = "weights",
) -> "AnyManager | None":
    state = _tag_states.get(tag)
    if state is None:
        return None
    return state.manager


def get_gms_client_memory_managers() -> tuple["AnyManager", ...]:
    return tuple(state.manager for state in _tag_states.values())


def evict_gms_client_memory_manager(manager: "AnyManager") -> None:
    for tag, state in list(_tag_states.items()):
        if state.manager is manager:
            _tag_states.pop(tag, None)
            return


@contextmanager
def gms_use_mem_pool(tag: str, device: "torch.device | int") -> Iterator[None]:
    import torch

    state = _tag_states.get(tag)
    if state is None:
        raise RuntimeError(f"No GMS allocator initialized for tag={tag}")
    if state.mem_pool is None:
        raise RuntimeError(f"GMS allocator tag={tag} does not have a mempool")

    token = _active_tag.set(tag)
    try:
        with torch.cuda.use_mem_pool(state.mem_pool, device=device):
            yield
    finally:
        _active_tag.reset(token)
