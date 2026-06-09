# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM BlockPool integration for GMS KV block leases."""

from __future__ import annotations

import logging
import os
from typing import Callable

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


class GMSKVLeaseUnavailable(ValueError):
    """Shared KV leases were temporarily unavailable for this allocation."""


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


def _install_engine_core_process_hooks() -> None:
    """Install scheduler-process GMS KV hooks before BlockPool creation."""
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


def install(factory: Callable[[int], KVLeaseClient] | None = None) -> bool:
    """Patch vLLM's BlockPool so block allocation is lease-gated."""

    global _patched, _factory
    if factory is not None:
        _factory = factory
    if _patched:
        return False
    if _factory is None and not kv_leases_enabled("vllm"):
        return False

    try:
        from vllm.v1.core.block_pool import BlockPool
    except Exception:  # noqa: BLE001
        logger.debug("[GMS-KVLease] vLLM BlockPool not importable", exc_info=True)
        return False

    orig_init = BlockPool.__init__
    orig_get_new_blocks = BlockPool.get_new_blocks
    orig_free_blocks = BlockPool.free_blocks
    orig_get_num_free_blocks = BlockPool.get_num_free_blocks

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

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        client = _make_client(int(self.num_gpu_blocks))
        self._gms_kv_lease_client = client
        self._gms_kv_leases_by_block: dict[int, KVLease] = {}
        logger.info(
            "[GMS-KVLease] vLLM BlockPool leases enabled namespace=%s owner=%s blocks=%d",
            getattr(client, "namespace", "?"),
            getattr(client, "owner_id", "?"),
            self.num_gpu_blocks,
        )

    def patched_get_num_free_blocks(self) -> int:
        client = getattr(self, "_gms_kv_lease_client", None)
        if client is None:
            return orig_get_num_free_blocks(self)
        return min(orig_get_num_free_blocks(self), int(client.free_count()))

    def patched_get_new_blocks(self, num_blocks: int):
        client = getattr(self, "_gms_kv_lease_client", None)
        if client is None:
            return orig_get_new_blocks(self, num_blocks)
        local_free = orig_get_num_free_blocks(self)
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

    def patched_free_blocks(self, ordered_blocks, prepend: bool = False):
        client = getattr(self, "_gms_kv_lease_client", None)
        if client is None:
            return orig_free_blocks(self, ordered_blocks, prepend=prepend)

        blocks_list = list(ordered_blocks)
        for block in blocks_list:
            block.ref_cnt -= 1

        free_blocks = [
            block for block in blocks_list if block.ref_cnt == 0 and not block.is_null
        ]
        leases = []
        missing_lease_blocks = []
        for block in free_blocks:
            if self.enable_caching:
                self._maybe_evict_cached_block(block)
            lease = self._gms_kv_leases_by_block.pop(int(block.block_id), None)
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
        client.release(leases)
        if prepend:
            self.free_block_queue.prepend_n(free_blocks)
        else:
            self.free_block_queue.append_n(free_blocks)

    try:
        from vllm.v1.core.kv_cache_manager import KVCacheManager
    except Exception:  # noqa: BLE001
        KVCacheManager = None  # type: ignore[assignment]

    if KVCacheManager is not None:
        orig_allocate_slots = KVCacheManager.allocate_slots

        def patched_allocate_slots(self, *args, **kwargs):
            try:
                return orig_allocate_slots(self, *args, **kwargs)
            except GMSKVLeaseUnavailable:
                num_new_computed_tokens = (
                    args[2]
                    if len(args) > 2
                    else kwargs.get("num_new_computed_tokens", 0)
                )
                new_computed_blocks = (
                    args[3] if len(args) > 3 else kwargs.get("new_computed_blocks")
                )
                num_external_computed_tokens = (
                    args[5]
                    if len(args) > 5
                    else kwargs.get("num_external_computed_tokens", 0)
                )
                if new_computed_blocks is None:
                    has_new_computed_blocks = False
                else:
                    groups = getattr(new_computed_blocks, "blocks", ())
                    has_new_computed_blocks = any(len(group) > 0 for group in groups)
                safe_to_backpressure = (
                    int(num_new_computed_tokens or 0) == 0
                    and not has_new_computed_blocks
                    and int(num_external_computed_tokens or 0) == 0
                )
                if not safe_to_backpressure:
                    raise
                log_lease_pressure(
                    logger,
                    "vllm:allocate-slots-backpressure",
                    "[GMS-KVLease] vLLM scheduler backpressured by shared leases",
                    requested_tokens=int(num_new_computed_tokens or 0),
                    external_tokens=int(num_external_computed_tokens or 0),
                )
                logger.debug(
                    "[GMS-KVLease] vLLM allocation backpressured by shared leases",
                    exc_info=True,
                )
                return None

        KVCacheManager.allocate_slots = patched_allocate_slots  # type: ignore[method-assign]

    BlockPool.__init__ = patched_init  # type: ignore[method-assign]
    BlockPool.get_new_blocks = patched_get_new_blocks  # type: ignore[method-assign]
    BlockPool.free_blocks = patched_free_blocks  # type: ignore[method-assign]
    BlockPool.get_num_free_blocks = patched_get_num_free_blocks  # type: ignore[method-assign]
    _patched = True
    logger.info("[GMS-KVLease] patched vLLM BlockPool")
    return True


if kv_leases_enabled("vllm"):
    try:
        install()
    except Exception:  # noqa: BLE001
        logger.exception("[GMS-KVLease] vLLM auto-install failed")
