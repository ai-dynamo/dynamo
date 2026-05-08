# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM monkey-patches applied at GMSWorker import.

Patches:
  - MemorySnapshot.measure: adds GMS-committed bytes to free_memory in RO mode.
  - request_memory: bypasses the free>=requested check during deferred-KV init.
  - NixlConnector.register_kv_caches: defers registration during the scratch
    phase and stashes the dict for replay at wake.

The torch.cuda.empty_cache patch lives in integrations/common/patches.py.
"""

from __future__ import annotations

import logging

from gpu_memory_service.client.torch.allocator import get_gms_client_memory_manager
from gpu_memory_service.common.locks import GrantedLockType
from gpu_memory_service.common.utils import is_scratch_kv_enabled

logger = logging.getLogger(__name__)

_memory_snapshot_patched = False
_request_memory_patched = False
_register_kv_caches_patched = False


# =============================================================================
# Core GMS patch (always applied)
# =============================================================================


def patch_memory_snapshot() -> None:
    """Add committed GMS bytes to MemorySnapshot.free_memory"""
    global _memory_snapshot_patched

    if _memory_snapshot_patched:
        return

    try:
        from vllm.utils.mem_utils import MemorySnapshot
    except ImportError:
        logger.debug("[GMS Patch] MemorySnapshot not available")
        return

    original_measure = MemorySnapshot.measure

    def patched_measure(self):
        original_measure(self)

        manager = get_gms_client_memory_manager("weights")
        assert manager is not None, "GMS client is not initialized"

        if manager.granted_lock_type == GrantedLockType.RO:
            allocations = manager.list_handles()
            committed_bytes = sum(alloc.aligned_size for alloc in allocations)
        else:
            # NOTE: by design, we want to assume we have the whole GPU when writing
            # weights for the first time, so we don't make an adjustment.
            committed_bytes = 0
            logger.info("[GMS] RW mode - skipping committed memory adjustment")

        original_free = self.free_memory
        self.free_memory += committed_bytes

        if committed_bytes > 0:
            logger.info(
                "[GMS Patch] Adjusted free_memory: %.2f GiB + %.2f GiB = %.2f GiB",
                original_free / (1 << 30),
                committed_bytes / (1 << 30),
                self.free_memory / (1 << 30),
            )

    MemorySnapshot.measure = patched_measure
    _memory_snapshot_patched = True
    logger.info("[GMS Patch] Patched MemorySnapshot.measure")


# =============================================================================
# Shadow mode patches
# =============================================================================


def patch_request_memory() -> None:
    """Bypass free >= requested check (shadow shares GPU with active engine)."""
    global _request_memory_patched

    if _request_memory_patched:
        return

    try:
        from vllm.v1.worker import utils as worker_utils
    except ImportError:
        logger.debug("[GMS Patch] vllm.v1.worker.utils not available")
        return

    def patched_request_memory(init_snapshot, cache_config):
        requested_memory = int(
            init_snapshot.total_memory * cache_config.gpu_memory_utilization
        )
        logger.info(
            "[GMS Patch] Shadow mode: bypassing memory check "
            "(requested=%.2f GiB, free=%.2f GiB)",
            requested_memory / (1 << 30),
            init_snapshot.free_memory / (1 << 30),
        )
        return requested_memory

    worker_utils.request_memory = patched_request_memory
    _request_memory_patched = True
    logger.info("[GMS Patch] Patched request_memory for shadow mode")


def patch_register_kv_caches() -> None:
    """Defer NixlConnector.register_kv_caches while KV backing is scratch-aliased.

    Registering NIXL MRs over scratch would pin a soon-stale page into the NIC;
    sleep tears down scratch and wake remaps real backing at the same VAs.
    Stash the dict during the scratch phase and let GMSWorker.wake_up replay
    it after remap.
    """
    global _register_kv_caches_patched

    if _register_kv_caches_patched:
        return

    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (
            NixlConnector,
        )
    except ImportError:
        logger.debug("[GMS Patch] NixlConnector not available")
        return

    original_register = NixlConnector.register_kv_caches

    def patched_register_kv_caches(self, kv_caches):
        from gpu_memory_service.client.torch.allocator import (
            get_gms_client_memory_manager,
            is_scratch,
        )

        # Fail closed on lookup errors: falling through to original_register
        # would pin an MR onto a scratch page that sleep is about to free,
        # exactly the bug this patch exists to prevent.
        try:
            kv_mgr = get_gms_client_memory_manager("kv_cache")
            has_deferred = kv_mgr is not None and is_scratch(kv_mgr)
        except (LookupError, AttributeError, RuntimeError) as exc:
            logger.warning(
                "[GMS Patch] Cannot determine deferred-KV state — "
                "raising to avoid pinning a stale scratch MR: %s",
                exc,
                exc_info=True,
            )
            raise

        if has_deferred:
            self._scratch_kv_pending = kv_caches
            logger.info(
                "[GMS Patch] Deferring NIXL KV cache registration "
                "(stashed %d layers for wake replay)",
                len(kv_caches),
            )
            return
        return original_register(self, kv_caches)

    NixlConnector.register_kv_caches = patched_register_kv_caches
    _register_kv_caches_patched = True
    logger.info("[GMS Patch] Patched NixlConnector.register_kv_caches")


# =============================================================================
# Patch application helper
# =============================================================================


def apply_scratch_kv_patches() -> None:
    """Apply scratch-KV monkey-patches. No-ops when scratch KV is disabled."""
    if not is_scratch_kv_enabled():
        return

    patch_request_memory()
    patch_register_kv_caches()
    logger.info("[GMS Patch] applied")
