# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU Memory Service allocator registry for PyTorch integration."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Optional

from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType
from gpu_memory_service.common.vmm import VMMDeviceType, get_vmm_device_type

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

# Torch drives one malloc per caching-allocator segment through us, and every
# resulting server allocation costs ~15 ms to re-import when a sleeping engine
# wakes (export round trip + cuMemImport + cuMemSetAccess; cuMemMap is free).
# Packing those mallocs into a few large slabs is what keeps wake proportional
# to the weight set rather than to the segment count: GLM-5.2 TP8 goes from 585
# allocations to ~28, and weights remap from ~8.7 s to well under a second.
#
# The cost is committed memory: a slab is allocated whole, so its unused tail
# stays resident. Set DYN_GMS_SLAB_SIZE=0 to go back to one server allocation
# per malloc.
DEFAULT_SLAB_SIZE = 2 * 1024 * 1024 * 1024


def _slab_size_for(tag: str) -> int:
    """Slab size for a tag's mempool, overridable via DYN_GMS_SLAB_SIZE."""
    raw = os.environ.get("DYN_GMS_SLAB_SIZE")
    if raw is None or raw == "":
        return DEFAULT_SLAB_SIZE
    return int(raw)


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
        if not state.manager.owns(va):
            continue
        logger.debug("[GMS] free(tag=%s): va=0x%x size=%d", tag, va, size)
        state.manager.destroy_mapping(va, int(size))
        return
    logger.warning("[GMS] free: no manager owns va=0x%x, ignoring", va)


def _ensure_callbacks_initialized() -> None:
    global _callbacks_initialized, _pluggable_alloc

    if get_vmm_device_type() != VMMDeviceType.CUDA:
        raise NotImplementedError(
            f"GMS torch mempool integration is CUDA-only; device_type={get_vmm_device_type().value} "
        )

    from gpu_memory_service.client.torch.extensions import _allocator_ext as cumem
    from torch.cuda import CUDAPluggableAllocator

    if _callbacks_initialized:
        return

    _pluggable_alloc = CUDAPluggableAllocator(cumem.__file__, "my_malloc", "my_free")
    cumem.init_module(_gms_malloc, _gms_free)
    _callbacks_initialized = True


def _create_mem_pool() -> "MemPool":
    if get_vmm_device_type() != VMMDeviceType.CUDA:
        raise NotImplementedError(
            f"GMS torch mempool integration is CUDA-only; device_type={get_vmm_device_type().value} "
        )

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

    manager = GMSClientMemoryManager(
        socket_path, device=device, tag=tag, slab_size=_slab_size_for(tag)
    )
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
    scratch_size: int = 512 * 1024 * 1024,
) -> "GMSClientMemoryManager":
    """Register an unconnected manager for client-local scratch allocation.

    The manager is constructed but .connect() is NOT called. _gms_malloc routes
    via create_scratch_mapping while is_scratch is True. Caller must invoke
    .connect(...) before any server-backed operation, then call
    manager.prepare_scratch_for_reallocation() to move preserved-VA bookkeeping
    and flip routing to the standard create_mapping path.
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
        if state.manager.scratch_size != scratch_size:
            raise RuntimeError(
                f"GMS scratch allocator tag={tag} was initialized with "
                f"scratch_size={state.manager.scratch_size}, not {scratch_size}"
            )
        return state.manager

    manager = GMSClientMemoryManager(
        socket_path,
        device=device,
        tag=tag,
        scratch_size=scratch_size,
    )
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


def get_gms_client_memory_manager(
    tag: str = "weights",
) -> "GMSClientMemoryManager | None":
    state = _tag_states.get(tag)
    if state is None:
        return None
    return state.manager


def get_gms_client_memory_managers() -> tuple["GMSClientMemoryManager", ...]:
    return tuple(state.manager for state in _tag_states.values())


def release_weight_mempool(manager: "GMSClientMemoryManager") -> None:
    """Drop the tag's MemPool so torch returns its dead blocks to GMS.

    This is how load-time scratch is reclaimed. PyTorch's caching allocator
    holds freed blocks instead of handing them back, and ``empty_cache()`` is
    a no-op while live GMS mempool mappings exist, so the blocks are only
    released when the pool itself is destroyed: torch then calls the free
    callback for every cached, unreferenced block. Blocks still owned by live
    Parameter storage stay mapped and become the committed weight set.

    Reclaim therefore happens at torch-block granularity, which is what slab
    packing needs -- a keep-set of allocation IDs could only free whole server
    allocations, and one live tensor would pin an entire slab.

    The pool is not needed again: after publication the manager serves the
    weights read-only.
    """
    if manager.tag is None:
        raise RuntimeError("cannot release the mempool of an untagged manager")
    state = _tag_states.get(manager.tag)
    if state is None or state.mem_pool is None:
        return

    from gpu_memory_service.integrations.common.utils import torch_device

    torch_device().synchronize(manager.device)

    before_bytes = manager.total_bytes
    before_count = len(manager.mappings)

    import gc

    mem_pool = state.mem_pool
    state.mem_pool = None
    del mem_pool
    gc.collect()

    logger.info(
        "[GMS] Released the %s mempool: %d -> %d allocations, "
        "%.2f -> %.2f GiB live",
        manager.tag,
        before_count,
        len(manager.mappings),
        before_bytes / (1 << 30),
        manager.total_bytes / (1 << 30),
    )


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

    if get_vmm_device_type() != VMMDeviceType.CUDA:
        raise NotImplementedError(
            f"gms_use_mem_pool is CUDA-only; device_type={get_vmm_device_type().value} "
        )

    token = _active_tag.set(tag)
    try:
        with torch.cuda.use_mem_pool(state.mem_pool, device=device):
            yield
    finally:
        _active_tag.reset(token)
