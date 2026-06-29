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
    persistent_defer_physical: bool = False


_tag_states: dict[str, _TagState] = {}
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
    # is_scratch / is_persistent flags decide routing.
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
        if state.persistent_tag_plan is not None and state.persistent_alloc_seq < len(
            state.persistent_tag_plan
        ):
            sub_tag = state.persistent_tag_plan[state.persistent_alloc_seq]
        else:
            sub_tag = f"{tag}#{state.persistent_alloc_seq}"
        state.persistent_alloc_seq += 1
        if state.persistent_defer_physical:
            scratch_backed = os.environ.get(
                "GMS_PERSISTENT_DEFER_PHYSICAL_SCRATCH_BACKED", "0"
            ).lower() not in {"0", "false", "no", "off"}
            va = state.manager.create_scratch_mapping(
                size=int(size),
                tag=sub_tag,
                map_scratch=scratch_backed,
            )
            logger.debug(
                "[GMS] deferred persistent malloc(eng=%s tag=%s): "
                "va=0x%x size=%d scratch_backed=%s",
                state.persistent_engine_id,
                sub_tag,
                va,
                size,
                scratch_backed,
            )
            return va
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
        import torch

        torch.cuda.synchronize(manager.device)

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
    defer_physical: bool = False,
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
        if state.persistent_defer_physical != defer_physical:
            raise RuntimeError(
                f"GMS allocator tag={tag} already registered with "
                f"defer_physical={state.persistent_defer_physical}, not "
                f"{defer_physical}"
            )
        return state.manager

    # Persistent mode uses a KV-only session that bypasses the normal
    # weights-layout RW/RO FSM. Multiple engines may keep these sessions
    # open concurrently when shared=True and coordinate writes via KV leases.
    manager = GMSClientMemoryManager(socket_path, device=device, tag=tag)
    if not defer_physical:
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
        persistent_defer_physical=defer_physical,
    )
    logger.info(
        "[GMS] Registered persistent allocator tag=%s engine_id=%s device=%d "
        "defer_physical=%s",
        tag,
        engine_id,
        device,
        defer_physical,
    )
    return manager


def retarget_persistent_allocator(tag: str, engine_id: str, *, shared: bool) -> None:
    """Update persistent allocation routing after a VA-preserving remap.

    Warm-standby failover can initialize vLLM on a private bootstrap namespace,
    sleep/unmap that namespace, then remap the same tensor VAs to the shared
    primary namespace during wake. The allocator state must follow the remap so
    any future persistent allocations use the promoted namespace.
    """
    state = _tag_states.get(tag)
    if state is None:
        raise RuntimeError(f"No GMS persistent allocator for tag={tag}")
    if not state.is_persistent:
        raise RuntimeError(f"GMS allocator tag={tag} is not persistent")
    state.persistent_engine_id = engine_id
    state.persistent_shared = shared
    state.persistent_defer_physical = False


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
