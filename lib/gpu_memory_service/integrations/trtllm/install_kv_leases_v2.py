# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT-LLM KVCacheManager V2 SlotAllocator lease integration."""

from __future__ import annotations

import itertools
import logging
import os
import threading
from typing import Callable

from gpu_memory_service.integrations.common.kv_lease_client import (
    GMSKVLeaseClient,
    KVLeaseClient,
    kv_leases_enabled,
    log_lease_pressure,
    resolve_lease_device,
)

logger = logging.getLogger(__name__)

_patched = False
_factory: Callable[[object, int, str], KVLeaseClient] | None = None
_ALLOC_STATE: dict[int, dict[str, object]] = {}
_GROUP_ORDINALS = itertools.count()
_RECLAIM_LOCK = threading.RLock()

# During CUDA-graph capture warmup the executor allocates/releases dummy KV slots. Those
# slots are transient (released immediately after capture) and never need a failover lease,
# but the per-slot GMS lease acquire is a cross-rank flock + shared-mmap round-trip that can
# return different results on each TP rank -> the ranks build different warmup batches -> one
# rank enters the captured NCCL all-reduce while the other skips it -> collective deadlock
# (both ranks spin at 100% CPU, capture never finishes). When this flag is set, the patched
# allocator falls back to the engine's LOCAL allocator (identical/deterministic on every rank,
# exactly like the no-GMS path), so capture proceeds. Real per-request leases (and thus the
# pause/resume failover path) are unaffected — only warmup-phase allocations are bypassed.
_warmup_bypass = False


def set_warmup_bypass(value: bool) -> None:
    """Toggle local-only KV slot allocation for the CUDA-graph/warmup phase."""
    global _warmup_bypass
    _warmup_bypass = bool(value)


def _env_enabled(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("", "0", "false", "no", "off")


def _v2_transition_reclaim_enabled() -> bool:
    """Gate V2 proactive reclaim behind host-tier mirroring/fallback.

    Reclaiming a TRT-LLM V2 GPU page is only useful for shadow transition when
    the same cached KV has already been mirrored into the GMS CPU host tier.
    Keep this opt-in via the existing CPU-tier knobs, with an explicit TRT V2
    override for tests and controlled rollouts.
    """

    explicit = os.environ.get("GMS_TRTLLM_V2_TRANSITION_RECLAIM")
    if explicit is None:
        explicit = os.environ.get("GMS_TRTLLM_TRANSITION_RECLAIM")
    if explicit is not None:
        return explicit.strip().lower() not in ("", "0", "false", "no", "off")
    return _env_enabled("GMS_KVR_HOST_TIER_FALLBACK") or _env_enabled(
        "GMS_KVR_HOST_TIER_MIRROR"
    )


def reclaim_cached_kv_v2_for_allocator(allocator_id: int, target_blocks: int) -> int:
    """Free evictable TRT-LLM V2 GPU pages for one leased pool group.

    This intentionally uses TRT-LLM V2's own eviction controller. Active decode
    pages are locked/held by the manager and are not in the evictable queue, so
    the operation only drops reusable cached pages. Slot release then flows
    through our patched ``SlotAllocator.release`` path, which releases the
    matching GMS lease and gives the shadow owner headroom.
    """

    target = max(0, int(target_blocks))
    if target <= 0:
        return 0
    with _RECLAIM_LOCK:
        st = _ALLOC_STATE.get(int(allocator_id))
        if st is None:
            return 0
        manager = st.get("manager")
        allocator = st.get("allocator")
        pool_group_index = st.get("pool_group_index")
        if manager is None or allocator is None or pool_group_index is None:
            return 0
        try:
            pg_idx = int(pool_group_index)
            storage = getattr(manager, "_storage")
            gpu_level = 0
            levels = getattr(storage, "_levels")
            controller = levels[gpu_level].controller
            evictable = int(controller.num_evictable_pages(pg_idx))
            to_evict = min(target, evictable)
            if to_evict <= 0:
                return 0

            before_free = int(getattr(allocator, "num_free_slots"))
            goals = [0] * int(storage.num_pool_groups)
            goals[pg_idx] = to_evict
            storage.force_evict(gpu_level, goals)
            after_free = int(getattr(allocator, "num_free_slots"))
            reclaimed = max(0, after_free - before_free)
            return reclaimed if reclaimed > 0 else to_evict
        except Exception:  # noqa: BLE001
            logger.debug(
                "[GMS-KVLease] TRT-LLM V2 reclaim failed",
                exc_info=True,
            )
            return 0


def reclaim_cached_kv_v2(target_blocks: int) -> int:
    """Best-effort aggregate V2 reclaim across registered GPU pool groups."""

    remaining = max(0, int(target_blocks))
    reclaimed = 0
    for allocator_id in list(_ALLOC_STATE):
        if remaining <= 0:
            break
        n = reclaim_cached_kv_v2_for_allocator(allocator_id, remaining)
        reclaimed += int(n)
        remaining -= int(n)
    return reclaimed


def _start_reclaim_watcher(allocator_id: int, st: dict[str, object]) -> None:
    if st.get("reclaim_watcher") is not None:
        return
    if not _v2_transition_reclaim_enabled():
        return
    suffix = st.get("namespace_suffix")
    if not suffix:
        return
    try:
        from gpu_memory_service.integrations.common.transition_reclaim import (
            start_kv_transition_reclaim_watcher,
        )

        watcher = start_kv_transition_reclaim_watcher(
            "trtllm",
            lambda target: reclaim_cached_kv_v2_for_allocator(allocator_id, target),
            namespace_suffix=str(suffix),
        )
    except Exception:  # noqa: BLE001
        logger.debug(
            "[GMS-KVLease] TRT-LLM V2 transition reclaim watcher not started",
            exc_info=True,
        )
        return
    if watcher is not None:
        st["reclaim_watcher"] = watcher


def _register_manager_for_reclaim(manager: object) -> None:
    """Bind TRT-LLM V2 KVCacheManager storage to leased pool groups."""

    try:
        storage = getattr(manager, "_storage")
        gpu_storage = getattr(storage, "_levels")[0].storage
        pool_groups = getattr(gpu_storage, "_pool_groups")
    except Exception:  # noqa: BLE001
        logger.debug(
            "[GMS-KVLease] TRT-LLM V2 manager has no GPU pool groups",
            exc_info=True,
        )
        return
    for pg_idx, group in enumerate(pool_groups):
        allocator = getattr(group, "_slot_allocator", None)
        if allocator is None:
            continue
        st = _ALLOC_STATE.get(id(allocator))
        if st is None:
            continue
        st["manager"] = manager
        st["pool_group_index"] = int(pg_idx)
        _start_reclaim_watcher(id(allocator), st)


def _unregister_manager_for_reclaim(manager: object) -> None:
    # Module globals may already be cleared while KVCacheManager.__del__ runs
    # during interpreter shutdown.
    alloc_state = _ALLOC_STATE
    if alloc_state is None:
        return
    for st in list(alloc_state.values()):
        if st.get("manager") is not manager:
            continue
        watcher = st.pop("reclaim_watcher", None)
        if watcher is not None:
            try:
                watcher.stop()
            except Exception:  # noqa: BLE001
                pass
        st.pop("manager", None)
        st.pop("pool_group_index", None)


def install(factory: Callable[[object, int, str], KVLeaseClient] | None = None) -> bool:
    global _patched, _factory
    if factory is not None:
        _factory = factory
    if _patched:
        return False
    if _factory is None and not kv_leases_enabled("trtllm"):
        return False

    try:
        from tensorrt_llm.runtime.kv_cache_manager_v2._storage import _core
    except Exception:  # noqa: BLE001
        logger.debug("[GMS-KVLease] TRT-LLM V2 storage not importable", exc_info=True)
        return False
    try:
        from tensorrt_llm.runtime.kv_cache_manager_v2._core._kv_cache_manager import (
            KVCacheManager,
        )
    except Exception:  # noqa: BLE001
        KVCacheManager = None
        logger.debug(
            "[GMS-KVLease] TRT-LLM V2 KVCacheManager not importable; "
            "lease patch will run without transition reclaim",
            exc_info=True,
        )

    GpuPoolGroup = _core.GpuPoolGroup
    SlotAllocator = _core.SlotAllocator
    Slot = _core.Slot
    SlotId = _core.SlotId
    CachedCudaEvent = _core.CachedCudaEvent
    OutOfPagesError = _core.OutOfPagesError

    orig_gpu_group_init = GpuPoolGroup.__init__
    orig_allocate = SlotAllocator.allocate
    orig_allocate_multiple = SlotAllocator.allocate_multiple
    orig_release = SlotAllocator.release
    if KVCacheManager is not None:
        orig_manager_init = KVCacheManager.__init__
        orig_manager_shutdown = KVCacheManager.shutdown

    def _make_client(group, total_slots: int, suffix: str) -> KVLeaseClient:
        if _factory is not None:
            return _factory(group, total_slots, suffix)
        # The lease MUST bind the same GPU the V2 pool is allocated on. Under the
        # MPI proxy executor each TP rank is a Comm-spawned worker; the
        # LOCAL_RANK / OMPI_COMM_WORLD_LOCAL_RANK env heuristic is unreliable there
        # (all spawned ranks can report 0 -> every rank binds gpu0 -> the non-zero
        # ranks touch gpu0 memory from their own context => CUDA 700). The V2 pool
        # is created on the worker's *current* CUDA device, so resolve from that;
        # an explicit GMS_TRTLLM_KV_LEASE_DEVICE still overrides for tests.
        device = None
        _explicit = os.environ.get("GMS_TRTLLM_KV_LEASE_DEVICE")
        if _explicit is not None:
            try:
                device = int(_explicit)
            except ValueError:
                device = None
        if device is None:
            try:
                import torch

                if torch.cuda.is_available():
                    device = int(torch.cuda.current_device())
            except Exception:  # noqa: BLE001
                device = None
        if device is None:
            device = resolve_lease_device(
                "GMS_TRTLLM_KV_LEASE_DEVICE",
                fallback_env_names=("LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK"),
            )
        return GMSKVLeaseClient.from_env(
            "trtllm",
            device,
            total_blocks=total_slots,
            namespace_suffix=suffix,
        )

    def patched_gpu_group_init(self, num_slots, slot_size_list, shared_phys_mem_pool):
        orig_gpu_group_init(self, num_slots, slot_size_list, shared_phys_mem_pool)
        ordinal = next(_GROUP_ORDINALS)
        suffix = (
            f"v2-gpu-pool-{ordinal}:slots{int(num_slots)}:"
            f"sizes{','.join(str(int(x)) for x in slot_size_list)}"
        )
        client = _make_client(self, int(num_slots), suffix)
        _ALLOC_STATE[id(self._slot_allocator)] = {
            "allocator": self._slot_allocator,
            "client": client,
            "leases_by_slot": {},
            "namespace_suffix": suffix,
        }
        logger.info(
            "[GMS-KVLease] TRT-LLM V2 GPU pool leases enabled namespace=%s owner=%s slots=%d",
            getattr(client, "namespace", "?"),
            getattr(client, "owner_id", "?"),
            int(num_slots),
        )

    def _state(allocator) -> dict[str, object] | None:
        return _ALLOC_STATE.get(id(allocator))

    def _capacity(allocator) -> int:
        return min(int(allocator.num_slots), int(allocator._target_capacity))

    def _safe_free_count(client: KVLeaseClient) -> int:
        try:
            return int(client.free_count())
        except Exception:  # noqa: BLE001
            logger.debug(
                "[GMS-KVLease] TRT-LLM V2 free-count read failed", exc_info=True
            )
            return -1

    def _preferred_slots(allocator, limit: int | None = None) -> list[int]:
        allocator._scrub_events()
        preferred: list[int] = []
        seen: set[int] = set()

        def add(slot_id: int) -> bool:
            if slot_id in seen:
                return False
            preferred.append(slot_id)
            seen.add(slot_id)
            return limit is not None and len(preferred) >= int(limit)

        for slot in allocator._recycled_slots:
            if add(int(slot.slot_id)):
                return preferred
        for slot_id in range(int(allocator._num_active_slots), _capacity(allocator)):
            if add(int(slot_id)):
                return preferred
        return preferred

    def _take_local_slot(allocator, slot_id: int):
        allocator._scrub_events()
        for idx, slot in enumerate(list(allocator._recycled_slots)):
            if int(slot.slot_id) != int(slot_id):
                continue
            del allocator._recycled_slots[idx]
            if idx < allocator._num_ready_recycled_slots:
                allocator._num_ready_recycled_slots -= 1
            allocator._occupied_mask.set(slot.slot_id)
            return slot
        if int(allocator._num_active_slots) <= slot_id < _capacity(allocator):
            for skipped in range(int(allocator._num_active_slots), slot_id):
                allocator._recycled_slots.append(
                    Slot(SlotId(skipped), CachedCudaEvent.NULL)
                )
                allocator._num_ready_recycled_slots += 1
            slot = Slot(SlotId(slot_id), CachedCudaEvent.NULL)
            allocator._num_active_slots = slot_id + 1
            allocator._occupied_mask.set(slot.slot_id)
            return slot
        raise OutOfPagesError(f"No local free slot compatible with GMS lease {slot_id}")

    def patched_allocate(self):
        st = _state(self)
        if st is None or _warmup_bypass:
            return orig_allocate(self)
        client = st["client"]
        if self.num_free_slots == 0:
            log_lease_pressure(
                logger,
                f"trtllm:{getattr(client, 'namespace', '?')}:local-exhausted",
                "[GMS-KVLease] TRT-LLM V2 allocation blocked by local free slots",
                namespace=getattr(client, "namespace", "?"),
                owner_id=getattr(client, "owner_id", "?"),
                active_slots=int(getattr(self, "_num_active_slots", 0)),
                capacity=_capacity(self),
                active_leases=len(st["leases_by_slot"]),
            )
            raise OutOfPagesError("No free slots")

        def acquire_with_preferred(limit: int | None) -> tuple[list[object], list[int]]:
            preferred = _preferred_slots(self, limit=limit)
            leases = client.acquire(
                1,
                preferred_blocks=preferred,
                strict_preferred=True,
            )
            return leases, preferred

        try:
            leases, preferred = acquire_with_preferred(1)
        except Exception as exc:  # noqa: BLE001
            try:
                leases, preferred = acquire_with_preferred(None)
            except Exception as fallback_exc:  # noqa: BLE001
                log_lease_pressure(
                    logger,
                    f"trtllm:{getattr(client, 'namespace', '?')}:acquire-error",
                    "[GMS-KVLease] TRT-LLM V2 lease acquire failed",
                    namespace=getattr(client, "namespace", "?"),
                    owner_id=getattr(client, "owner_id", "?"),
                    local_free=int(self.num_free_slots),
                    shared_free=_safe_free_count(client),
                    preferred_count=len(_preferred_slots(self, limit=16)),
                    error=type(fallback_exc).__name__,
                )
                raise OutOfPagesError("No GMS-leased slot available") from exc
        if len(leases) != 1:
            log_lease_pressure(
                logger,
                f"trtllm:{getattr(client, 'namespace', '?')}:acquire-short",
                "[GMS-KVLease] TRT-LLM V2 lease acquire returned unexpected count",
                namespace=getattr(client, "namespace", "?"),
                owner_id=getattr(client, "owner_id", "?"),
                requested=1,
                returned=len(leases),
                shared_free=_safe_free_count(client),
                preferred_count=len(preferred),
            )
            client.release(leases)
            raise OutOfPagesError("No GMS-leased slot available")
        lease = leases[0]
        try:
            slot = _take_local_slot(self, int(lease.block_id))
        except Exception:
            client.release(leases)
            raise
        lease_map = st["leases_by_slot"]
        assert isinstance(lease_map, dict)
        lease_map[int(slot.slot_id)] = lease
        return slot

    def patched_allocate_multiple(self, num_slots: int):
        st = _state(self)
        if st is None or _warmup_bypass:
            return orig_allocate_multiple(self, num_slots)
        if self.num_free_slots < num_slots:
            raise OutOfPagesError("Not enough free slots")
        allocated = []
        try:
            for _ in range(int(num_slots)):
                allocated.append(patched_allocate(self))
        except Exception:
            for slot in allocated:
                patched_release(self, slot)
            raise
        return allocated

    def patched_release(self, slot):
        st = _state(self)
        if st is None or _warmup_bypass:
            return orig_release(self, slot)
        slot_id = int(slot.slot_id)
        orig_release(self, slot)
        lease_map = st["leases_by_slot"]
        assert isinstance(lease_map, dict)
        lease = lease_map.pop(slot_id, None)
        if lease is not None:
            st["client"].release([lease])
        else:
            client = st["client"]
            log_lease_pressure(
                logger,
                f"trtllm:{getattr(client, 'namespace', '?')}:missing-release",
                "[GMS-KVLease] TRT-LLM V2 releasing slot without matching lease",
                namespace=getattr(client, "namespace", "?"),
                owner_id=getattr(client, "owner_id", "?"),
                slot_id=slot_id,
                active_leases=len(lease_map),
            )

    if KVCacheManager is not None:

        def patched_manager_init(self, config):
            orig_manager_init(self, config)
            _register_manager_for_reclaim(self)

        def patched_manager_shutdown(self):
            _unregister_manager_for_reclaim(self)
            return orig_manager_shutdown(self)

    GpuPoolGroup.__init__ = patched_gpu_group_init  # type: ignore[method-assign]
    SlotAllocator.allocate = patched_allocate  # type: ignore[method-assign]
    SlotAllocator.allocate_multiple = patched_allocate_multiple  # type: ignore[method-assign]
    SlotAllocator.release = patched_release  # type: ignore[method-assign]
    if KVCacheManager is not None:
        KVCacheManager.__init__ = patched_manager_init  # type: ignore[method-assign]
        KVCacheManager.shutdown = patched_manager_shutdown  # type: ignore[method-assign]

    _patched = True
    logger.info("[GMS-KVLease] patched TRT-LLM V2 GPU SlotAllocator")
    return True


if kv_leases_enabled("trtllm"):
    try:
        install()
    except Exception:  # noqa: BLE001
        logger.exception("[GMS-KVLease] TRT-LLM V2 auto-install failed")
