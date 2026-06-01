# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU Memory Service allocator registry for PyTorch integration."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Optional

from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType

if TYPE_CHECKING:
    import torch
    from gpu_memory_service.client.memory_manager import GMSClientMemoryManager
    from torch.cuda.memory import MemPool

logger = logging.getLogger(__name__)


@dataclass
class _TagState:
    manager: "GMSClientMemoryManager"
    mem_pool: "MemPool | None"
    socket_path: str
    device: int
    is_scratch: bool = False


_tag_states: dict[str, _TagState] = {}
_active_tag: ContextVar[str | None] = ContextVar(
    "gpu_memory_service_active_tag",
    default=None,
)
_callbacks_initialized = False
_pluggable_alloc: Any | None = None


def _gms_malloc(size: int, device: int, stream: int) -> int:
    # Tag-context dispatch: the active tag (set by gms_use_mem_pool) selects
    # the registry; state.is_scratch decides scratch vs server-backed routing.
    tag = _active_tag.get()
    if tag is None:
        raise RuntimeError("No active GMS allocation tag")

    state = _tag_states.get(tag)
    if state is None:
        raise RuntimeError(f"Unknown GMS allocation tag: {tag}")

    if state.is_scratch:
        va = state.manager.create_scratch_mapping(size=int(size), tag=tag)
        logger.debug("[GMS] scratch malloc(tag=%s): va=0x%x size=%d", tag, va, size)
        return va

    va = state.manager.create_mapping(size=int(size), tag=tag)
    logger.debug("[GMS] malloc(tag=%s): va=0x%x size=%d", tag, va, size)
    return va


def _gms_free(ptr: int, size: int, device: int, stream: int) -> None:
    # Content-driven dispatch: torch only gives us a VA, no tag context.
    # Try the scratch registry first across all managers, then standard.
    va = int(ptr)
    for tag, state in _tag_states.items():
        if state.manager.destroy_scratch_mapping(va):
            logger.debug("[GMS] scratch free(tag=%s): va=0x%x size=%d", tag, va, size)
            return
    for tag, state in _tag_states.items():
        if va not in state.manager.mappings:
            continue
        logger.debug("[GMS] free(tag=%s): va=0x%x size=%d", tag, va, size)
        state.manager.destroy_mapping(va)
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
    from gpu_memory_service.client.memory_manager import GMSClientMemoryManager

    state = _tag_states.get(tag)
    if state is not None:
        if state.socket_path != socket_path or state.device != device:
            raise RuntimeError(
                f"GMS allocator tag={tag} was initialized for "
                f"{state.socket_path} on device {state.device}, not {socket_path} "
                f"on device {device}"
            )

        manager = state.manager
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
        socket_path=socket_path,
        device=device,
    )
    logger.info(
        "[GMS] Created %s allocator for tag=%s (device=%d)",
        manager.granted_lock_type.value,
        tag,
        device,
    )
    return manager


def get_or_create_scratch_manager(
    socket_path: str,
    device: int,
    *,
    tag: str = "kv_cache",
) -> "GMSClientMemoryManager":
    """Register an unconnected manager for client-local scratch allocation.

    The manager is constructed but .connect() is NOT called. _gms_malloc routes
    via create_scratch_mapping while is_scratch is True. Caller must invoke
    .connect(...) before any server-backed operation, then call
    ensure_scratch_disabled(manager) to flip routing to the standard
    create_mapping path.
    """
    from gpu_memory_service.client.memory_manager import GMSClientMemoryManager

    state = _tag_states.get(tag)
    if state is not None:
        if state.socket_path != socket_path or state.device != device:
            raise RuntimeError(
                f"GMS allocator tag={tag} was initialized for "
                f"{state.socket_path} on device {state.device}, not {socket_path} "
                f"on device {device}"
            )
        if not state.is_scratch:
            raise RuntimeError(
                f"GMS allocator tag={tag} already registered as non-scratch; "
                "use get_or_create_gms_client_memory_manager instead"
            )
        return state.manager

    manager = GMSClientMemoryManager(socket_path, device=device, tag=tag)
    _ensure_callbacks_initialized()
    mem_pool = _create_mem_pool()

    _tag_states[tag] = _TagState(
        manager=manager,
        mem_pool=mem_pool,
        socket_path=socket_path,
        device=device,
        is_scratch=True,
    )
    logger.info(
        "[GMS] Registered scratch allocator for tag=%s (device=%d)", tag, device
    )
    return manager


def is_scratch(manager: "GMSClientMemoryManager") -> bool:
    """True if the manager's tag is currently in scratch routing.

    Routes through manager.tag → _tag_states. Raises if the manager is not
    registered.
    """
    if manager.tag is None:
        raise RuntimeError("manager has no tag; not registered in allocator")
    state = _tag_states.get(manager.tag)
    if state is None:
        raise RuntimeError(f"tag {manager.tag!r} not in _tag_states")
    return state.is_scratch


def ensure_scratch_disabled(manager: "GMSClientMemoryManager") -> None:
    """Flip the manager's tag out of scratch routing.

    After this, _gms_malloc routes via create_mapping (server-backed) on the
    tag's mempool. Idempotent. Raises if the manager is not registered or
    not currently RW-connected — server-backed allocations require RW.

    Call after migrating scratch entries into _mappings via
    prepare_scratch_for_reallocation, before reallocate_all_handles.
    """
    if manager.tag is None:
        raise RuntimeError("manager has no tag; not registered in allocator")
    state = _tag_states.get(manager.tag)
    if state is None:
        raise RuntimeError(f"tag {manager.tag!r} not in _tag_states")
    if manager.granted_lock_type != GrantedLockType.RW:
        raise RuntimeError(
            f"ensure_scratch_disabled requires RW grant on tag={manager.tag!r}; "
            f"got granted_lock_type={manager.granted_lock_type}. "
            "Did you forget to .connect(RequestedLockType.RW) first?"
        )
    state.is_scratch = False


def get_gms_client_memory_manager(
    tag: str = "weights",
) -> "GMSClientMemoryManager | None":
    state = _tag_states.get(tag)
    if state is None:
        return None
    return state.manager


def get_gms_client_memory_managers() -> tuple["GMSClientMemoryManager", ...]:
    return tuple(state.manager for state in _tag_states.values())


def evict_gms_client_memory_manager(manager: "GMSClientMemoryManager") -> None:
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
