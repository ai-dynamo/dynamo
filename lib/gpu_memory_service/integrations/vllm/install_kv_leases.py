# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM BlockPool integration for GMS KV block leases."""

from __future__ import annotations

from collections.abc import Callable
import logging
import os

from gms_kv_ring.common.content_directory import ContentDirectory

from gpu_memory_service.integrations.common.kv_lease_client import (
    GMSKVLeaseClient,
    KVLease,
    KVLeaseClient,
    kv_leases_enabled,
    log_lease_pressure,
    resolve_lease_device,
)

logger = logging.getLogger(__name__)

_patched = False
_factory: Callable[[int], KVLeaseClient] | None = None
_engine_core_hook_patched = False
_original_run_engine_core = None
_gms_block_pool_class = None


class GMSKVLeaseUnavailable(ValueError):
    """Shared KV leases were temporarily unavailable for this allocation."""


def _failover_directory_standby() -> bool | None:
    """Derive the content-directory role before vLLM creates its BlockPool.

    The engine-core process inherits the replica identity but does not run
    WorkerFactory's post-init failover orchestration. In failover mode that
    identity is authoritative: a non-primary replica must start its directory
    reader as a standby so it can hydrate protected HBM after promotion.
    """
    enabled = any(
        os.environ.get(name, "").lower() not in {"", "0", "false", "no", "off"}
        for name in ("DYN_GMS_FAILOVER_SHADOW_MODE", "DYN_VLLM_GMS_SHADOW_MODE")
    )
    if not enabled:
        return None
    return os.environ.get("ENGINE_ID", "0") != os.environ.get(
        "DYN_GMS_FAILOVER_PRIMARY_ENGINE_ID", "0"
    )


def _preferred_block_ids(free_block_queue, limit: int) -> list[int]:
    """Return a bounded prefix of local free block IDs without walking
    vLLM's entire free list. The real queue is a linked list; unit-test
    fakes may only expose get_all_free_blocks().
    """
    if limit <= 0:
        return []
    out: list[int] = []
    head = getattr(free_block_queue, "fake_free_list_head", None)
    block = getattr(head, "next_free_block", None) if head is not None else None
    while block is not None and getattr(block, "next_free_block", None) is not None:
        if not getattr(block, "is_null", False):
            out.append(int(block.block_id))
            if len(out) >= limit:
                return out
        block = getattr(block, "next_free_block", None)

    get_all = getattr(free_block_queue, "get_all_free_blocks", None)
    if get_all is None:
        return out
    for block in get_all():
        if getattr(block, "is_null", False):
            continue
        block_id = int(block.block_id)
        if block_id in out:
            continue
        out.append(block_id)
        if len(out) >= limit:
            break
    return out


def _preferred_candidate_limit(num_blocks: int) -> int:
    configured = os.environ.get("GMS_VLLM_KV_LEASE_PREFERRED_CANDIDATES")
    if configured:
        try:
            return max(num_blocks, int(configured))
        except ValueError:
            logger.warning(
                "Ignoring invalid GMS_VLLM_KV_LEASE_PREFERRED_CANDIDATES=%r",
                configured,
            )
    return num_blocks


def _fallback_preferred_candidate_limit(num_blocks: int) -> int:
    return max(num_blocks, min(max(num_blocks * 4, 256), 4096))


def install_gms_engine_core_sleep() -> bool:
    """Install the no-clear sleep utility in this EngineCore process.

    Multi-process/TP vLLM spawns EngineCoreProc without importing GMSWorker, so
    this must run from the scheduler-process bootstrap as well as the worker
    import path.
    """
    try:
        from concurrent.futures import Future

        from vllm.v1.engine.core import EngineCore
    except Exception:
        logger.debug("[GMS] EngineCore sleep utility patch skipped", exc_info=True)
        return False

    if hasattr(EngineCore, "gms_sleep_no_clear"):
        return False

    def gms_sleep_no_clear(self, level: int = 1, mode: str = "abort"):
        pause_future = self.pause_scheduler(mode=mode, clear_cache=False)
        if level < 1:
            return pause_future

        model_executor = self.model_executor

        def flush_directory() -> None:
            manager = getattr(self.scheduler, "kv_cache_manager", None)
            pool = getattr(manager, "block_pool", None)
            directory = getattr(pool, "_gms_kv_directory", None)
            flush = getattr(directory, "flush_deferred", None)
            if flush is not None and not flush(timeout=2.0):
                raise TimeoutError("GMS directory mutation flush timed out")

        if pause_future is None:
            flush_directory()
            model_executor.sleep(level)
            return None

        future = Future()

        def pause_complete(completed):
            try:
                completed.result()
                flush_directory()
                future.set_result(model_executor.sleep(level))
            except Exception as exc:  # noqa: BLE001
                future.set_exception(exc)

        logger.info("[GMS] Waiting for in-flight requests before no-clear sleep")
        pause_future.add_done_callback(pause_complete)
        return future

    EngineCore.gms_sleep_no_clear = gms_sleep_no_clear
    logger.info("[GMS] Installed EngineCore.gms_sleep_no_clear utility")
    return True


def _install_engine_core_process_hooks() -> None:
    """Install scheduler-process GMS hooks before EngineCore construction."""
    install_gms_engine_core_sleep()
    try:
        install()
    except Exception:  # noqa: BLE001
        logger.exception("[GMS-KVLease] EngineCore BlockPool lease install failed")
        raise

    try:
        from gpu_memory_service.integrations.vllm.install_vmm_ipc_kv import (
            install_geometry_patch,
        )

        install_geometry_patch()
    except Exception:  # noqa: BLE001
        logger.exception("[GMS-KVLease] EngineCore geometry patch install failed")
        raise


def run_engine_core_with_gms_kv_leases(*args, **kwargs):
    """Picklable wrapper for vLLM EngineCore subprocess bootstrap.

    vLLM builds BlockPool in the EngineCore scheduler process, not in the CUDA
    worker process that imports GMSWorker. The wrapper keeps the integration
    local to GMS while making spawned/forked EngineCore processes install the
    lease and geometry patches before scheduler construction.
    """
    global _original_run_engine_core

    original = _original_run_engine_core
    if original is None:
        from vllm.v1.engine.core import EngineCoreProc

        original = EngineCoreProc.run_engine_core
        if getattr(original, "_gms_kv_lease_engine_core_wrapper", False):
            raise RuntimeError("GMS EngineCore KV lease wrapper recursion detected")
        _original_run_engine_core = original

    _install_engine_core_process_hooks()
    return original(*args, **kwargs)


def install_engine_core_hook() -> bool:
    """Patch vLLM's EngineCore process target so scheduler KV leases install.

    The target must be a module-level function so it remains valid when vLLM
    uses a spawn multiprocessing context. Forked children reuse the saved
    original; spawned children resolve the original after importing vLLM.
    """
    global _engine_core_hook_patched, _original_run_engine_core
    if _engine_core_hook_patched:
        return False
    if not kv_leases_enabled("vllm"):
        return False

    try:
        from vllm.v1.engine.core import EngineCoreProc
    except Exception:  # noqa: BLE001
        logger.debug("[GMS-KVLease] EngineCoreProc not importable", exc_info=True)
        return False

    current = EngineCoreProc.run_engine_core
    if getattr(current, "_gms_kv_lease_engine_core_wrapper", False):
        _engine_core_hook_patched = True
        return False

    _original_run_engine_core = current
    run_engine_core_with_gms_kv_leases._gms_kv_lease_engine_core_wrapper = True
    EngineCoreProc.run_engine_core = staticmethod(run_engine_core_with_gms_kv_leases)
    _engine_core_hook_patched = True
    logger.info("[GMS-KVLease] patched vLLM EngineCore process bootstrap")
    return True


# The one remaining vLLM method wrapper translates an atomic lease race into
# the scheduler's existing backpressure result. BlockPool behavior itself is
# provided by a subclass installed at its single construction site.
orig_allocate_slots = None


def _make_client(total_blocks: int) -> KVLeaseClient:
    if _factory is not None:
        return _factory(total_blocks)
    device = resolve_lease_device("GMS_VLLM_KV_LEASE_DEVICE")
    return GMSKVLeaseClient.from_env(
        "vllm",
        device,
        total_blocks=total_blocks,
        namespace_suffix="block-pool",
        reserved_blocks=[0],
    )

def _make_directory(hash_block_size: int) -> ContentDirectory:
    socket_path = (
        os.environ.get("GMS_KV_DIRECTORY_SOCKET")
        or os.environ.get("GMS_VLLM_DAEMON_SOCKET")
        or ""
    )
    return ContentDirectory(
        socket_path,
        engine="vllm",
        block_size=int(hash_block_size),
        mode=os.environ.get("GMS_KV_DIRECTORY_MODE"),
        keyspace="vllm-native-hbm-v1",
        standby=_failover_directory_standby(),
    )

def _initialize_gms_block_pool(self) -> None:
    client = _make_client(int(self.num_gpu_blocks))
    self._gms_kv_lease_client = client
    self._gms_kv_leases_by_block: dict[int, KVLease] = {}
    self._gms_kv_directory = _make_directory(int(self.hash_block_size))
    start_directory_sync = getattr(self._gms_kv_directory, "start_async_read", None)
    if start_directory_sync is not None:
        start_directory_sync()
    hydrate = os.environ.get("GMS_VLLM_HYDRATE_HBM")
    if hydrate is None:
        self._gms_hydrate_hbm = bool(
            getattr(self._gms_kv_directory, "_standby", False)
        )
    else:
        self._gms_hydrate_hbm = hydrate.lower() not in (
            "0",
            "false",
            "no",
            "off",
            "",
        )
    logger.info(
        "[GMS-KVLease] vLLM BlockPool leases enabled namespace=%s owner=%s blocks=%d",
        getattr(client, "namespace", "?"),
        getattr(client, "owner_id", "?"),
        self.num_gpu_blocks,
    )

def _directory_pool_id() -> str:
    return str(
        os.environ.get("GMS_VLLM_ENGINE_ID")
        or os.environ.get("GMS_KVR_ENGINE_ID")
        or "0"
    )

def _publish_hbm_blocks(self, blocks, *, active: bool) -> None:
    directory = getattr(self, "_gms_kv_directory", None)
    client = getattr(self, "_gms_kv_lease_client", None)
    if client is None:
        return
    lease_map = self._gms_kv_leases_by_block
    pairs = [
        (block, lease_map.get(int(block.block_id)))
        for block in blocks
        if getattr(block, "block_hash", None) is not None
    ]
    pairs = [(block, lease) for block, lease in pairs if lease is not None]
    if not pairs:
        return
    leases = [lease for _block, lease in pairs]
    client.seal(leases)
    if directory is None or not directory.enabled:
        return
    try:
        directory.publish(
            [
                {
                    "content_hash": bytes(block.block_hash),
                    "engine_id": _directory_pool_id(),
                    "slot_id": int(block.block_id),
                    "generation": int(lease.generation),
                    "tier": "hbm",
                    "active": active,
                }
                for block, lease in pairs
            ]
        )
    except Exception:  # noqa: BLE001
        logger.warning(
            "[GMS-KVLease] vLLM HBM directory publication failed",
            exc_info=True,
        )
        if directory.authoritative:
            raise

def _drop_directory_hashes(directory, entries) -> None:
    try:
        directory.publish(
            [
                {
                    "content_hash": content_hash,
                    "engine_id": _directory_pool_id(),
                    "slot_ids": entry.get("slot_ids") or [],
                    "generations": entry.get("generations") or [],
                    "tier": "hbm",
                    "sealed": False,
                }
                for content_hash, entry in entries
                if entry is not None
            ]
        )
    except Exception:  # noqa: BLE001
        logger.warning(
            "[GMS-KVLease] failed to invalidate stale HBM directory hits",
            exc_info=True,
        )
        if directory.authoritative:
            raise

def _hydrate_hbm_directory(self, exclude: set[bytes]) -> int:
    """Adopt a bounded recovery batch on vLLM's scheduler thread."""
    if not getattr(self, "_gms_hydrate_hbm", False):
        return 0
    directory = getattr(self, "_gms_kv_directory", None)
    client = getattr(self, "_gms_kv_lease_client", None)
    read_items = getattr(directory, "read_view_items", None)
    if directory is None or client is None or read_items is None:
        return 0
    try:
        limit = max(1, int(os.environ.get("GMS_VLLM_HYDRATE_BATCH", "256")))
    except ValueError:
        limit = 256
    candidates = []
    for key, entry in read_items(tier="hbm"):
        if key in exclude or self.cached_block_hash_to_block.get_one_block(key):
            continue
        slots = entry.get("slot_ids") or []
        generations = entry.get("generations") or []
        if len(slots) != 1 or len(generations) != 1:
            continue
        block_id = int(slots[0])
        if not 0 < block_id < len(self.blocks):
            continue
        block = self.blocks[block_id]
        if block.ref_cnt != 0 or block.block_hash is not None:
            continue
        candidates.append((key, entry))
        if len(candidates) >= limit:
            break
    if not candidates:
        if getattr(directory, "read_view_is_current_writer", False):
            self._gms_hydrate_hbm = False
        return 0

    keys = [key for key, _entry in candidates]
    token = None
    acquired = []
    installed = []
    claimed_entries = []
    try:
        entries, token = directory.lookup_and_claim(keys)
        selected = []
        for key, entry in zip(keys, entries):
            if entry is None or entry.get("tier") != "hbm":
                continue
            slots = entry.get("slot_ids") or []
            generations = entry.get("generations") or []
            if len(slots) != 1 or len(generations) != 1:
                continue
            block_id = int(slots[0])
            block = self.blocks[block_id]
            if block.ref_cnt != 0 or block.block_hash is not None:
                continue
            selected.append((key, entry, KVLease(block_id, int(generations[0]))))
        if not selected or token is None:
            return 0

        # Lease adoption is intentionally atomic for each batch. A stale
        # generation in the read snapshot must not discard every other
        # valid recovery candidate, so retain the one-call fast path and
        # bisect only failed groups. The directory claim fences all entries
        # throughout this bounded recovery operation.
        pending = [selected]
        adopted_pairs = []
        stale = []
        while pending:
            group = pending.pop()
            group_leases = client.adopt([old for _key, _entry, old in group])
            if group_leases:
                expected_ids = [old.block_id for _key, _entry, old in group]
                if [lease.block_id for lease in group_leases] != expected_ids:
                    raise RuntimeError("bulk HBM adoption returned different slots")
                adopted_pairs.extend(zip(group, group_leases))
            elif len(group) == 1:
                stale.extend(group)
            else:
                middle = len(group) // 2
                pending.extend((group[middle:], group[:middle]))

        if not adopted_pairs:
            directory.release_claim(token)
            token = None
            _drop_directory_hashes(
                directory, [(key, entry) for key, entry, _old in stale]
            )
            return 0

        acquired = [lease for _selected, lease in adopted_pairs]
        adopted = directory.adopt_claim(
            token,
            [
                {
                    "content_hash": selected_item[0],
                    "generations": [int(lease.generation)],
                }
                for selected_item, lease in adopted_pairs
            ],
        )
        token = None
        if adopted != len(adopted_pairs):
            raise RuntimeError("bulk HBM directory adoption was incomplete")
        if stale:
            _drop_directory_hashes(
                directory, [(key, entry) for key, entry, _old in stale]
            )

        for (key, entry, _old), lease in adopted_pairs:
            block = self.blocks[int(lease.block_id)]
            self._insert_block_hash(key, block, self.hash_block_size)
            self._gms_kv_leases_by_block[int(block.block_id)] = lease
            installed.append(block)
            claimed_entries.append((key, entry))

        # The adopted blocks are sealed cache entries, not immediately
        # allocatable slots. A fresh vLLM BlockPool orders its free queue
        # by block ID, so leaving recovered entries in place can put a
        # large run of unavailable leases at the head. The next allocation
        # then misses every preferred ID and the native lease ring rescans
        # from slot zero. Hydration is a cache touch: move the recovered
        # entries to the MRU tail once, off the steady-state request path,
        # so genuinely free queue heads remain aligned with free leases.
        for block in installed:
            self.free_block_queue.remove(block)
        self.free_block_queue.append_n(installed)

        # Bulk-hydrated entries are native evictable cache blocks, not
        # active request blocks. Seal and return them to READY immediately.
        client.seal(acquired)
        directory.mark_hbm_dormant(
            [selected_item[0] for selected_item, _lease in adopted_pairs]
        )
        if len(candidates) < limit and getattr(
            directory, "read_view_is_current_writer", False
        ):
            self._gms_hydrate_hbm = False
        log_hydration = (
            logger.warning
            if os.environ.get("GMS_KV_DIRECTORY_DIAGNOSTICS")
            else logger.info
        )
        log_hydration(
            "[GMS-KVDirectory] vLLM bulk_hydrated_hbm_blocks=%d",
            len(installed),
        )
        return len(installed)
    except Exception:  # noqa: BLE001
        for block in installed:
            self._maybe_evict_cached_block(block)
            self._gms_kv_leases_by_block.pop(int(block.block_id), None)
        if acquired:
            client.release(acquired)
        if claimed_entries:
            _drop_directory_hashes(directory, claimed_entries)
        logger.warning(
            "[GMS-KVLease] vLLM bulk HBM hydration failed",
            exc_info=True,
        )
        return 0
    finally:
        if token is not None:
            directory.release_claim(token)

def _get_cached_block(self, native_get_cached_block, block_hash, kv_cache_group_ids):
    local = native_get_cached_block(block_hash, kv_cache_group_ids)
    if local is not None:
        return local
    directory = getattr(self, "_gms_kv_directory", None)
    client = getattr(self, "_gms_kv_lease_client", None)
    if directory is None or not directory.enabled or client is None:
        return None
    hydration_was_complete = not getattr(self, "_gms_hydrate_hbm", False)
    # Once recovery hydration is complete and this engine is the fenced
    # writer, its native block-hash map is authoritative for HBM. Check
    # before constructing directory keys: ordinary native misses must not
    # pay content-hash conversion or replicated-view lookup costs.
    if hydration_was_complete and getattr(
        directory, "read_view_is_current_writer", False
    ):
        return None

    from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id

    keys = [
        bytes(make_block_hash_with_group_id(block_hash, group_id))
        for group_id in kv_cache_group_ids
    ]
    _hydrate_hbm_directory(self, set(keys))
    token = None
    entries = []
    acquired = []
    installed = []
    try:
        entries, token = directory.lookup_and_claim(keys)
        if len(entries) != len(keys) or any(
            entry is None or entry.get("tier") != "hbm" for entry in entries
        ):
            return None
        if directory.mode == "shadow":
            return None
        slot_ids = []
        old_leases = []
        for entry in entries:
            assert entry is not None
            slots = entry.get("slot_ids") or []
            generations = entry.get("generations") or []
            if len(slots) != 1 or len(generations) != 1:
                return None
            slot_ids.append(int(slots[0]))
            old_leases.append(KVLease(int(slots[0]), int(generations[0])))
        if len(set(slot_ids)) != len(slot_ids):
            return None

        # The failover lock and directory claim fence the former owner.
        # Atomic adoption changes owner/generation without a FREE window,
        # so the preserved HBM bytes cannot be reused between operations.
        acquired = client.adopt(old_leases)
        if [int(lease.block_id) for lease in acquired] != slot_ids:
            raise RuntimeError("GMS HBM adoption returned different slots")
        adopted = directory.adopt_claim(
            token,
            [
                {
                    "content_hash": key,
                    "generations": [int(lease.generation)],
                }
                for key, lease in zip(keys, acquired)
            ],
        )
        token = None
        if adopted != len(keys):
            raise RuntimeError("GMS HBM directory adoption was incomplete")

        out = []
        for key, lease in zip(keys, acquired):
            block = self.blocks[int(lease.block_id)]
            if block.ref_cnt != 0 or block.block_hash is not None:
                raise RuntimeError("adopted HBM slot is not locally free")
            self._insert_block_hash(key, block, self.hash_block_size)
            self._gms_kv_leases_by_block[int(block.block_id)] = lease
            installed.append(block)
            out.append(block)
        log_adoption = (
            logger.warning
            if os.environ.get("GMS_KV_DIRECTORY_DIAGNOSTICS")
            else logger.info
        )
        log_adoption("[GMS-KVDirectory] vLLM adopted_hbm_blocks=%d", len(out))
        return out
    except Exception:  # noqa: BLE001
        for block in installed:
            self._maybe_evict_cached_block(block)
            self._gms_kv_leases_by_block.pop(int(block.block_id), None)
        if acquired:
            client.release(acquired)
        if entries:
            _drop_directory_hashes(directory, list(zip(keys, entries)))
        logger.warning(
            "[GMS-KVLease] vLLM HBM directory adoption failed",
            exc_info=True,
        )
        return None
    finally:
        if token is not None:
            directory.release_claim(token)

def _evict_dormant_directory_blocks(self, required_blocks: int) -> int:
    directory = getattr(self, "_gms_kv_directory", None)
    client = getattr(self, "_gms_kv_lease_client", None)
    if directory is None or not directory.enabled or client is None:
        return 0
    victims = directory.ensure_hbm_capacity(required_blocks)
    leases = []
    restored = []
    for victim in victims:
        for block_id, generation in zip(victim["slot_ids"], victim["generations"]):
            block_id = int(block_id)
            block = self.blocks[block_id]
            if block.ref_cnt != 0:
                lease = self._gms_kv_leases_by_block.get(block_id)
                if lease is not None and block.block_hash is not None:
                    restored.append(block)
                continue
            if getattr(block, "block_hash", None) is not None:
                self._maybe_evict_cached_block(block)
            lease = self._gms_kv_leases_by_block.pop(block_id, None)
            leases.append(lease or KVLease(block_id, int(generation)))
    if restored:
        _publish_hbm_blocks(self, restored, active=True)
    client.release(leases)
    return len(leases)

def _reserve_dormant_headroom(self, recent_blocks: int) -> int:
    """Retire cold READY entries before the next allocation needs them.

    vLLM treats cached blocks in its free queue as immediately reusable.
    A sealed GMS block needs one extra ordered transition: remove directory
    visibility, evict the native hash, then release the lease. Doing that
    at request finalization preserves the same ordering while keeping the
    following request's allocation on the local shared-memory fast path.
    """
    directory = getattr(self, "_gms_kv_directory", None)
    client = getattr(self, "_gms_kv_lease_client", None)
    if (
        recent_blocks <= 0
        or directory is None
        or not directory.authoritative
        or client is None
    ):
        return 0
    shortage = int(recent_blocks) - int(client.free_count())
    if shortage <= 0:
        return 0
    return _evict_dormant_directory_blocks(self, shortage)

def _get_num_free_blocks(self, native_get_num_free_blocks) -> int:
    client = getattr(self, "_gms_kv_lease_client", None)
    if client is None:
        return native_get_num_free_blocks()
    local_free = native_get_num_free_blocks()
    directory = getattr(self, "_gms_kv_directory", None)
    if directory is not None and directory.authoritative:
        return local_free
    return min(local_free, int(client.free_count()))

def _get_new_blocks(self, native_get_num_free_blocks, num_blocks: int):
    client = getattr(self, "_gms_kv_lease_client", None)
    assert client is not None
    local_free = native_get_num_free_blocks()
    if num_blocks > local_free:
        log_lease_pressure(
            logger,
            f"vllm:{getattr(client, 'namespace', '?')}:local-exhausted",
            "[GMS-KVLease] vLLM allocation blocked by local free blocks",
            namespace=getattr(client, "namespace", "?"),
            owner_id=getattr(client, "owner_id", "?"),
            requested=int(num_blocks),
            local_free=local_free,
            active_leases=len(getattr(self, "_gms_kv_leases_by_block", {})),
        )
        raise GMSKVLeaseUnavailable(
            f"Cannot get {num_blocks} free blocks from the local pool"
        )

    try:
        shared_free = int(client.free_count())
        if shared_free < int(num_blocks):
            _evict_dormant_directory_blocks(self, int(num_blocks) - shared_free)
    except Exception:  # noqa: BLE001
        logger.debug(
            "[GMS-KVLease] dormant HBM capacity reclaim failed",
            exc_info=True,
        )
    preferred = _preferred_block_ids(
        self.free_block_queue,
        _preferred_candidate_limit(int(num_blocks)),
    )

    def acquire_with_preferred(
        candidates: list[int], *, strict: bool = False
    ) -> list[KVLease]:
        leases = client.acquire(
            int(num_blocks),
            preferred_blocks=candidates,
            strict_preferred=strict,
        )
        if len(leases) != num_blocks:
            client.release(leases)
            raise GMSKVLeaseUnavailable(
                f"GMS returned {len(leases)} leases, expected {num_blocks}"
            )
        return leases

    try:
        leases = acquire_with_preferred(preferred, strict=False)
    except Exception as exc:  # noqa: BLE001
        fallback_limit = _fallback_preferred_candidate_limit(int(num_blocks))
        if fallback_limit <= len(preferred):
            refresh = getattr(client, "refresh_free_count", None)
            if refresh is not None:
                try:
                    refresh()
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "[GMS-KVLease] free-count refresh failed after acquire error",
                        exc_info=True,
                    )
            raise GMSKVLeaseUnavailable(
                f"Cannot get {num_blocks} leased free blocks from the pool"
            ) from exc

        fallback_preferred = _preferred_block_ids(
            self.free_block_queue,
            fallback_limit,
        )
        try:
            leases = acquire_with_preferred(fallback_preferred, strict=False)
            preferred = fallback_preferred
        except Exception as fallback_exc:  # noqa: BLE001
            refresh = getattr(client, "refresh_free_count", None)
            if refresh is not None:
                try:
                    refresh()
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "[GMS-KVLease] free-count refresh failed after acquire error",
                        exc_info=True,
                    )
            raise GMSKVLeaseUnavailable(
                f"Cannot get {num_blocks} leased free blocks from the pool"
            ) from fallback_exc

    lease_block_ids = [int(lease.block_id) for lease in leases]
    try:
        if lease_block_ids == preferred[:num_blocks] and hasattr(
            self.free_block_queue, "popleft_n"
        ):
            ret = self.free_block_queue.popleft_n(num_blocks)
        else:
            ret = []
            for block_id in lease_block_ids:
                block = self.blocks[block_id]
                self.free_block_queue.remove(block)
                ret.append(block)
    except Exception:
        client.release(leases)
        raise

    for block, lease in zip(ret, leases):
        if self.enable_caching:
            self._maybe_evict_cached_block(block)
        assert block.ref_cnt == 0
        block.ref_cnt += 1
        self._gms_kv_leases_by_block[int(block.block_id)] = lease
        if self.metrics_collector:
            self.metrics_collector.on_block_allocated(block)
    return ret


def _free_blocks(self, ordered_blocks):
    client = getattr(self, "_gms_kv_lease_client", None)
    assert client is not None

    blocks_list = list(ordered_blocks)
    for block in blocks_list:
        block.ref_cnt -= 1

    free_blocks = [
        block for block in blocks_list if block.ref_cnt == 0 and not block.is_null
    ]
    leases = []
    missing_lease_blocks = []
    retained = []
    invalidated = []
    directory = getattr(self, "_gms_kv_directory", None)
    # Retaining a freed block's prefix hash (sealing its lease instead of
    # releasing + evicting) preserves cross-request prefix caching, which the
    # release-on-every-free path otherwise silently disables under
    # GMS_KV_LEASES=1. It is safe whenever no peer may overwrite the block:
    # (a) an authoritative content directory fences slot reuse, or (b) the
    # lease namespace is not shared (single writer), which operators opt into
    # with GMS_KV_LEASES_RETAIN_PREFIX_CACHE=1. Default keeps the previous
    # conservative eviction so shared-namespace correctness is unchanged.
    retain_without_directory = os.environ.get(
        "GMS_KV_LEASES_RETAIN_PREFIX_CACHE", "0"
    ).lower() not in {"0", "false", "no", "off", ""}
    for block in free_blocks:
        lease = self._gms_kv_leases_by_block.get(int(block.block_id))
        retain_dormant = bool(
            self.enable_caching
            and block.block_hash is not None
            and lease is not None
            and (
                (directory is not None and directory.authoritative)
                or retain_without_directory
            )
        )
        if retain_dormant:
            retained.append(block)
            continue
        content_hash = (
            bytes(block.block_hash) if block.block_hash is not None else None
        )
        if self.enable_caching and block.block_hash is not None:
            self._maybe_evict_cached_block(block)
        lease = self._gms_kv_leases_by_block.pop(int(block.block_id), None)
        if content_hash is not None and directory is not None and directory.enabled:
            invalidated.append(
                (
                    content_hash,
                    {
                        "slot_ids": [int(block.block_id)],
                        "generations": [
                            0 if lease is None else int(lease.generation)
                        ],
                    },
                )
            )
        if lease is not None:
            leases.append(lease)
        else:
            missing_lease_blocks.append(int(block.block_id))
    if missing_lease_blocks:
        log_lease_pressure(
            logger,
            f"vllm:{getattr(client, 'namespace', '?')}:missing-release",
            "[GMS-KVLease] vLLM releasing blocks without matching leases",
            namespace=getattr(client, "namespace", "?"),
            owner_id=getattr(client, "owner_id", "?"),
            missing_count=len(missing_lease_blocks),
            first_missing_block=missing_lease_blocks[0],
            active_leases=len(getattr(self, "_gms_kv_leases_by_block", {})),
        )
    if retained:
        # Request finalization is the only HBM durability boundary. Seal
        # every completed slot as one lease-ring operation, then publish
        # one READY batch. A crash before publication leaves safely
        # undiscoverable sealed slots; a crash after it leaves an
        # adoptable directory generation. Publishing ACTIVE earlier adds
        # scheduler work but cannot make an incomplete block recoverable.
        _publish_hbm_blocks(self, retained, active=False)
        _reserve_dormant_headroom(self, len(retained))
    if invalidated:
        _drop_directory_hashes(directory, invalidated)
    client.release(leases)
    # Match current vLLM's reuse policy: unhashed blocks are hot reusable
    # capacity and go to the LIFO head, while retained prefix-cache entries go
    # to the FIFO/LRU tail.  Appending every block was inherited from the old
    # ``prepend=`` API and regressed locality after vLLM removed that argument.
    reuse_first = [
        block
        for block in free_blocks
        if block.block_hash is None or not self.enable_caching
    ]
    reuse_last = [
        block
        for block in free_blocks
        if block.block_hash is not None and self.enable_caching
    ]
    self.free_block_queue.prepend_n(reuse_first)
    self.free_block_queue.append_n(reuse_last)


def patched_allocate_slots(self, *args, **kwargs):
    try:
        return orig_allocate_slots(self, *args, **kwargs)
    except GMSKVLeaseUnavailable:
        # Lease contention must NEVER crash the engine. Returning None
        # signals vLLM's scheduler to defer/preempt this request (normal
        # backpressure), regardless of whether it had prefix/computed
        # blocks. The previous code re-raised for prefix-cache-hit
        # requests, which propagated out of EngineCore.step and killed
        # the whole engine under routine shared-lease contention. Always
        # backpressuring also removes the fragile positional-arg parsing
        # that a vLLM signature change would have silently broken.
        log_lease_pressure(
            logger,
            "vllm:allocate-slots-backpressure",
            "[GMS-KVLease] vLLM scheduler backpressured by shared leases",
        )
        logger.debug(
            "[GMS-KVLease] vLLM allocation backpressured by shared leases",
            exc_info=True,
        )
        return None



def _build_gms_block_pool_class(block_pool_class):
    """Build the lease-aware BlockPool without modifying vLLM's base class."""

    class GMSBlockPool(block_pool_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            _initialize_gms_block_pool(self)

        def get_cached_block(self, block_hash, kv_cache_group_ids):
            return _get_cached_block(
                self,
                super().get_cached_block,
                block_hash,
                kv_cache_group_ids,
            )

        def get_num_free_blocks(self) -> int:
            return _get_num_free_blocks(self, super().get_num_free_blocks)

        def get_new_blocks(self, num_blocks: int):
            return _get_new_blocks(
                self,
                super().get_num_free_blocks,
                num_blocks,
            )

        def free_blocks(self, ordered_blocks):
            return _free_blocks(self, ordered_blocks)

    GMSBlockPool.__name__ = "GMSBlockPool"
    GMSBlockPool.__qualname__ = "GMSBlockPool"
    GMSBlockPool.__module__ = __name__
    return GMSBlockPool


def install(factory: Callable[[int], KVLeaseClient] | None = None) -> bool:
    """Install a lease-aware BlockPool at vLLM's construction site."""

    global _patched, _factory, _gms_block_pool_class, orig_allocate_slots

    if factory is not None:
        _factory = factory
    if _patched:
        return False
    if _factory is None and not kv_leases_enabled("vllm"):
        return False

    try:
        from vllm.v1.core import kv_cache_coordinator
        from vllm.v1.core.block_pool import BlockPool
        from vllm.v1.core.kv_cache_manager import KVCacheManager
    except Exception:  # noqa: BLE001
        logger.debug(
            "[GMS-KVLease] vLLM scheduler allocation API not importable",
            exc_info=True,
        )
        return False

    orig_allocate_slots = KVCacheManager.allocate_slots
    KVCacheManager.allocate_slots = patched_allocate_slots  # type: ignore[method-assign]
    _gms_block_pool_class = _build_gms_block_pool_class(BlockPool)
    kv_cache_coordinator.BlockPool = _gms_block_pool_class
    _patched = True
    logger.info("[GMS-KVLease] installed vLLM GMSBlockPool")
    return True


def lease_hooks_installed() -> bool:
    """Verify the construction binding and atomic-contention guard."""
    try:
        from vllm.v1.core import kv_cache_coordinator
        from vllm.v1.core.block_pool import BlockPool
        from vllm.v1.core.kv_cache_manager import KVCacheManager
    except Exception:  # noqa: BLE001
        return False
    installed = kv_cache_coordinator.BlockPool
    return bool(
        installed is _gms_block_pool_class
        and installed is not BlockPool
        and issubclass(installed, BlockPool)
        and KVCacheManager.allocate_slots is patched_allocate_slots
    )
