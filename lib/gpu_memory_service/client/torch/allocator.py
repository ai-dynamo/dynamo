# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU Memory Service allocator registry for PyTorch integration."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Optional

from gpu_memory_service.common.types import GrantedLockType, RequestedLockType

if TYPE_CHECKING:
    import torch
    from gpu_memory_service.client.memory_manager import GMSClientMemoryManager
    from torch.cuda.memory import MemPool

logger = logging.getLogger(__name__)


@dataclass
class _ScopeState:
    manager: "GMSClientMemoryManager"
    tag: str
    mem_pool: "MemPool | None"


_scope_states: dict[str, _ScopeState] = {}
_active_scope: ContextVar[str | None] = ContextVar(
    "gpu_memory_service_active_scope",
    default=None,
)
_callbacks_initialized = False
_pluggable_alloc: Any | None = None


def _gms_malloc(size: int, device: int, stream: int) -> int:
    scope = _active_scope.get()
    if scope is None:
        raise RuntimeError("No active GMS allocation scope")

    state = _scope_states.get(scope)
    if state is None:
        raise RuntimeError(f"Unknown GMS allocation scope: {scope}")

    va = state.manager.create_mapping(size=int(size), tag=state.tag)
    logger.debug("[GMS] malloc(scope=%s): va=0x%x size=%d", scope, va, size)
    return va


def _gms_free(ptr: int, size: int, device: int, stream: int) -> None:
    va = int(ptr)
    for scope, state in _scope_states.items():
        if va not in state.manager.mappings:
            continue
        logger.debug("[GMS] free(scope=%s): va=0x%x size=%d", scope, va, size)
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
    scope: str = "weights",
    tag: str | None = None,
    timeout_ms: Optional[int] = None,
) -> "GMSClientMemoryManager":
    from gpu_memory_service.client.memory_manager import GMSClientMemoryManager

    state = _scope_states.get(scope)
    if state is not None:
        current = state.manager.granted_lock_type
        if mode == RequestedLockType.RW and current != GrantedLockType.RW:
            raise RuntimeError(
                f"Cannot get RW allocator for scope {scope}: existing is in {current} mode"
            )
        if mode == RequestedLockType.RO and current != GrantedLockType.RO:
            raise RuntimeError(
                f"Cannot get RO allocator for scope {scope}: existing is in {current} mode"
            )
        return state.manager

    manager = GMSClientMemoryManager(socket_path, device=device)
    manager.connect(mode, timeout_ms=timeout_ms)

    mem_pool = None
    if manager.granted_lock_type == GrantedLockType.RW:
        _ensure_callbacks_initialized()
        mem_pool = _create_mem_pool()

    _scope_states[scope] = _ScopeState(
        manager=manager,
        tag=tag or scope,
        mem_pool=mem_pool,
    )
    logger.info(
        "[GMS] Created %s allocator for scope=%s (device=%d)",
        manager.granted_lock_type.value,
        scope,
        device,
    )
    return manager


def get_gms_client_memory_manager(
    scope: str = "weights",
) -> "GMSClientMemoryManager | None":
    state = _scope_states.get(scope)
    if state is None:
        return None
    return state.manager


def get_gms_client_memory_managers() -> tuple["GMSClientMemoryManager", ...]:
    return tuple(state.manager for state in _scope_states.values())


@contextmanager
def gms_use_mem_pool(scope: str, device: "torch.device | int") -> Iterator[None]:
    import torch

    state = _scope_states.get(scope)
    if state is None:
        raise RuntimeError(f"No GMS allocator initialized for scope={scope}")
    if state.mem_pool is None:
        raise RuntimeError(f"GMS allocator scope={scope} does not have a mempool")

    token = _active_scope.set(scope)
    try:
        with torch.cuda.use_mem_pool(state.mem_pool, device=device):
            yield
    finally:
        _active_scope.reset(token)
