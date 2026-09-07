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
    # Persistent-namespace routing: when True, _gms_malloc routes
    # through create_persistent_mapping (KV-pool flow) keyed by
    # (persistent_engine_id, per-allocation auto tag) instead of
    # create_mapping (weights flow).
    is_persistent: bool = False
    persistent_engine_id: str = ""
    # Counter for auto-generated per-allocation tags so multiple
    # torch.empty() calls inside the same `with` block produce
    # distinct persistent allocations.
    persistent_alloc_seq: int = 0
    persistent_tag_plan: list[str] | None = None
    persistent_shared: bool = False


_tag_states: dict[str, _TagState] = {}
# Tags for which the unsafe ordinal-persistent-tag warning was already emitted.
_ordinal_persistent_warned: set[str] = set()
_active_tag: ContextVar[str | None] = ContextVar(
    "gpu_memory_service_active_tag",
    default=None,
)
_active_pool: ContextVar[tuple[str, int] | None] = ContextVar(
    "gpu_memory_service_active_pool",
    default=None,
)
_callbacks_initialized = False
_pluggable_alloc: Any | None = None


def _device_index(torch_mod: Any, device: "torch.device | int") -> int:
    if isinstance(device, int):
        return int(device)
    index = getattr(device, "index", None)
    if index is not None:
        return int(index)
    return int(torch_mod.cuda.current_device())


@contextmanager
def _use_gms_pool_context(
    tag: str,
    device: "torch.device | int",
    mem_pool: "MemPool",
) -> Iterator[None]:
    import torch

    device_index = _device_index(torch, device)
    active_pool = _active_pool.get()
    if active_pool is not None:
        if active_pool == (tag, device_index):
            yield
            return
        raise RuntimeError(
            "Nested GMS mempool contexts must use the same tag and CUDA device: "
            f"active={active_pool}, requested={(tag, device_index)}"
        )

    tag_token = _active_tag.set(tag)
    pool_token = _active_pool.set((tag, device_index))
    try:
        with torch.cuda.use_mem_pool(mem_pool, device=device):
            yield
    finally:
        _active_pool.reset(pool_token)
        _active_tag.reset(tag_token)


def _gms_malloc(size: int, device: int, stream: int) -> int:
    # Tag-context dispatch: the active tag (set by gms_use_mem_pool /
    # gms_use_persistent_pool) selects the registry; the state's
    # is_persistent flag decides routing.
    tag = _active_tag.get()
    if tag is None:
        raise RuntimeError("No active GMS allocation tag")

    state = _tag_states.get(tag)
    if state is None:
        raise RuntimeError(f"Unknown GMS allocation tag: {tag}")

    if state.is_persistent:
        # Auto-generate a per-allocation sub-tag so successive
        # torch.empty() calls inside the same persistent scope get
        # distinct persistent allocations (one per layer / per buffer).
        # Private-bootstrap shadows allocate VA-only scratch first and later
        # remap those VAs onto the shared namespace. Their tags must therefore
        # match the shared pool tags exactly.
        if state.persistent_tag_plan is not None:
            # A semantic plan is supposed to cover EVERY persistent allocation.
            # If the malloc sequence overflows it, the callback fired more times
            # than planned (the CUDA caching allocator can fire per-segment, not
            # per-tensor), so an ordinal fallback here could bind this tensor to
            # another layer's physical pages on reattach -> silent KV corruption.
            # Fail closed instead.
            if state.persistent_alloc_seq >= len(state.persistent_tag_plan):
                raise RuntimeError(
                    f"GMS persistent tag plan exhausted for tag {tag!r}: "
                    f"allocation #{state.persistent_alloc_seq} has no planned "
                    f"semantic tag (malloc sequence desynced from the plan). "
                    f"Refusing an ordinal fallback that could bind KV to the "
                    f"wrong physical pages on reattach."
                )
            sub_tag = state.persistent_tag_plan[state.persistent_alloc_seq]
        else:
            # No semantic plan: ordinal sub-tags are only stable within a single
            # process. Reattach across restart depends on reproducing the exact
            # malloc order, which the caching allocator does not guarantee, so
            # warn that reattach may bind the wrong pages.
            if tag not in _ordinal_persistent_warned:
                _ordinal_persistent_warned.add(tag)
                logger.warning(
                    "[GMS] persistent tag %s uses ordinal sub-tags (no semantic "
                    "tag plan); reattach across restart may bind tensors to the "
                    "wrong physical pages. Install a semantic tag plan.",
                    tag,
                )
            sub_tag = f"{tag}#{state.persistent_alloc_seq}"
        state.persistent_alloc_seq += 1
        va = state.manager.create_persistent_mapping(
            engine_id=state.persistent_engine_id,
            tag=sub_tag,
            size=int(size),
            shared=state.persistent_shared,
        )
        logger.debug(
            "[GMS] persistent malloc(eng=%s tag=%s): va=0x%x size=%d",
            state.persistent_engine_id,
            sub_tag,
            va,
            size,
        )
        return va

    va = state.manager.create_mapping(size=int(size), tag=tag)
    logger.debug("[GMS] malloc(tag=%s): va=0x%x size=%d", tag, va, size)
    return va


def _gms_free(ptr: int, size: int, device: int, stream: int) -> None:
    # Content-driven dispatch: torch only gives us a VA, no tag context.
    # Try the scratch registry first across all managers, then standard.
    va = int(ptr)
    for tag, state in _tag_states.items():
        if va not in state.manager.mappings:
            continue
        logger.debug("[GMS] free(tag=%s): va=0x%x size=%d", tag, va, size)
        state.manager.destroy_mapping(va)
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


def set_persistent_allocator_tag_plan(tag: str, planned_tags: list[str]) -> None:
    """Set semantic persistent allocation tags for the next allocation pass.

    vLLM V2 exposes the semantic KV tensor list before it calls torch allocation.
    Using those tags avoids relying on raw allocation ordinal order, which can
    differ across primary/private-bootstrap engines for heterogeneous KV specs.
    """
    state = _tag_states.get(tag)
    if state is None or not state.is_persistent:
        raise RuntimeError(f"GMS persistent allocator tag={tag!r} is not registered")
    state.persistent_tag_plan = list(planned_tags)
    state.persistent_alloc_seq = 0


def clear_persistent_allocator_tag_plan(tag: str) -> None:
    state = _tag_states.get(tag)
    if state is None:
        return
    state.persistent_tag_plan = None


def get_gms_client_memory_manager(
    tag: str = "weights",
) -> "GMSClientMemoryManager | None":
    state = _tag_states.get(tag)
    if state is None:
        return None
    return state.manager


def get_gms_client_memory_managers() -> tuple["GMSClientMemoryManager", ...]:
    return tuple(state.manager for state in _tag_states.values())


def prune_allocations(
    manager: "GMSClientMemoryManager",
    *,
    referenced_allocation_ids: set[str],
    synchronize: bool = True,
) -> None:
    """Free GMS allocations that are not in an explicit torch keep-set.

    Callers provide the allocation IDs that remain valid; this helper does not
    infer liveness from Python GC.  Weight loaders call it after registering
    module tensors, treating other allocations as load-time scratch/cache that
    PyTorch's caching allocator may leave behind because ``empty_cache()`` is a
    no-op while live GMS mempool mappings exist.

    Args:
        manager: GMS manager whose local mappings should be pruned.
        referenced_allocation_ids: Allocation IDs that must remain mapped and
            committed.
        synchronize: Synchronize CUDA before freeing unreferenced mappings.  The
            default avoids freeing a block while prior GPU work may still be
            using it.  Callers that have already synchronized can pass
            ``False``.

    """
    if manager.granted_lock_type != GrantedLockType.RW or manager.is_unmapped:
        return

    if not any(mapping.handle != 0 for mapping in manager.mappings.values()):
        return

    if synchronize:
        from gpu_memory_service.integrations.common.utils import torch_device

        torch_device().synchronize(manager.device)

    keep = {str(allocation_id) for allocation_id in referenced_allocation_ids}

    pruned_allocations = 0
    pruned_bytes = 0
    for va, mapping in list(manager.mappings.items()):
        if str(mapping.allocation_id) in keep:
            continue
        if mapping.handle == 0:
            continue
        pruned_allocations += 1
        pruned_bytes += int(mapping.aligned_size)
        manager.destroy_mapping(va)

    if pruned_allocations:
        logger.info(
            "[GMS] Pruned %d unreferenced allocations (%.2f GiB); "
            "kept %d registered allocations",
            pruned_allocations,
            pruned_bytes / (1 << 30),
            len(keep),
        )


def evict_gms_client_memory_manager(manager: "GMSClientMemoryManager") -> None:
    for tag, state in list(_tag_states.items()):
        if state.manager is manager:
            _tag_states.pop(tag, None)
            return


@contextmanager
def gms_use_mem_pool(tag: str, device: "torch.device | int") -> Iterator[None]:
    state = _tag_states.get(tag)
    if state is None:
        raise RuntimeError(f"No GMS allocator initialized for tag={tag}")
    if state.mem_pool is None:
        raise RuntimeError(f"GMS allocator tag={tag} does not have a mempool")

    with _use_gms_pool_context(tag, device, state.mem_pool):
        yield


def get_or_create_persistent_allocator(
    socket_path: str,
    device: int,
    engine_id: str,
    tag: str = "kv_pool",
    *,
    shared: bool = False,
) -> "GMSClientMemoryManager":
    """Register a Torch-routable allocator that creates persistent
    allocations on each ``torch.empty()`` inside ``gms_use_persistent_pool``.

    Unlike the weights flow this:
      - never commits / publishes a layout,
      - keys allocations by ``(engine_id, sub_tag)`` so engine restart
        re-attaches to the same physical pages,
      - allows the daemon to read/write the SAME PHYSICAL PAGES
        directly via its ``va_daemon`` mapping.
    """
    from gpu_memory_service.client.memory_manager import GMSClientMemoryManager

    state = _tag_states.get(tag)
    if state is not None:
        if state.socket_path != socket_path or state.device != device:
            raise RuntimeError(
                f"GMS allocator tag={tag} was initialized for "
                f"{state.socket_path} on device {state.device}, not "
                f"{socket_path} on device {device}"
            )
        if not state.is_persistent:
            raise RuntimeError(
                f"GMS allocator tag={tag} already registered as non-persistent; "
                "use a distinct tag for persistent KV pools"
            )
        if state.persistent_engine_id != engine_id:
            raise RuntimeError(
                f"GMS allocator tag={tag} already bound to engine_id="
                f"{state.persistent_engine_id!r}, not {engine_id!r}"
            )
        if state.persistent_shared != shared:
            raise RuntimeError(
                f"GMS allocator tag={tag} already registered with shared="
                f"{state.persistent_shared}, not {shared}"
            )
        return state.manager

    # Persistent mode uses a KV-only session that bypasses the normal
    # weights-layout RW/RO FSM. Multiple engines may keep these sessions
    # open concurrently when shared=True and coordinate writes via KV leases.
    manager = GMSClientMemoryManager(socket_path, device=device, tag=tag)
    manager.connect(RequestedLockType.RW_PERSISTENT)
    _ensure_callbacks_initialized()
    mem_pool = _create_mem_pool()

    _tag_states[tag] = _TagState(
        manager=manager,
        mem_pool=mem_pool,
        socket_path=socket_path,
        device=device,
        is_persistent=True,
        persistent_engine_id=engine_id,
        persistent_shared=shared,
    )
    logger.info(
        "[GMS] Registered persistent allocator tag=%s engine_id=%s device=%d",
        tag,
        engine_id,
        device,
    )
    return manager


@contextmanager
def gms_use_persistent_pool(
    tag: str,
    device: "torch.device | int",
) -> Iterator[None]:
    """Route torch.empty() / zeros() inside this block through GMS
    persistent allocations. Re-attach-on-reconnect, daemon owns the
    physical pages, engine restart preserves bytes.

    Caller must have previously registered the tag via
    ``get_or_create_persistent_allocator``.
    """
    state = _tag_states.get(tag)
    if state is None:
        raise RuntimeError(f"No GMS persistent allocator for tag={tag}")
    if not state.is_persistent:
        raise RuntimeError(
            f"GMS allocator tag={tag} is not in persistent mode; "
            "use gms_use_mem_pool instead"
        )
    if state.mem_pool is None:
        raise RuntimeError(f"GMS persistent allocator tag={tag} has no mempool")

    with _use_gms_pool_context(tag, device, state.mem_pool):
        yield
