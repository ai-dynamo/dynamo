# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM monkey-patches applied at GMSWorker import.

Patches:
  - MemorySnapshot.measure: adds GMS-committed bytes to free_memory in RO mode.
  - request_memory: bypasses the free>=requested check during deferred-KV init.
  - NixlBaseConnector KV registration: defers normal or cross-layer
    registration during the scratch phase and stashes it for replay at wake.
  - init_kv_cache: scopes the scratch mem-pool to the raw KV tensors only, so
    BlockTables / workspace / pointer tensors keep real (un-aliased) memory.

The torch.cuda.empty_cache patch lives in integrations/common/patches.py.
"""

from __future__ import annotations

import importlib
import logging

from gpu_memory_service.client.torch.allocator import (
    get_gms_client_memory_manager,
    is_scratch,
)
from gpu_memory_service.common.locks import GrantedLockType
from gpu_memory_service.common.utils import is_scratch_kv_enabled

logger = logging.getLogger(__name__)

_memory_snapshot_patched = False
_request_memory_patched = False
_register_kv_caches_patched = False
_kv_cache_pool_scope_patched = False
_NIXL_MODULE = "vllm.distributed.kv_transfer.kv_connector.v1.nixl"


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
    """Defer NIXL KV registration while KV backing is scratch-aliased.

    Registering NIXL MRs over scratch would pin a soon-stale page into the NIC;
    sleep tears down scratch and wake remaps real backing at the same VAs.
    Stash the normal dict or cross-layer tensor during the scratch phase and
    let GMSWorker.wake_up replay it after remap.
    """
    global _register_kv_caches_patched

    if _register_kv_caches_patched:
        return

    # Keep this optional-backend import deferred. GMS is collected in images
    # that do not install vLLM, and the connector is only required when this
    # vLLM-specific patch is enabled.
    try:
        nixl_module = importlib.import_module(_NIXL_MODULE)
    except ModuleNotFoundError as exc:
        # Treat a missing vLLM package (or missing connector package) as an
        # unavailable optional backend. Missing dependencies imported from an
        # installed connector must remain visible.
        missing_module = exc.name
        if missing_module and (
            missing_module == _NIXL_MODULE
            or _NIXL_MODULE.startswith(f"{missing_module}.")
        ):
            logger.debug("[GMS Patch] NixlBaseConnector not available")
            return
        raise

    # vLLM 0.27 exports NixlConnector as an alias for NixlPullConnector while
    # NixlPushConnector is its sibling. Patch their common base so both modes
    # retain the scratch-registration safety gate.
    nixl_base_connector = nixl_module.NixlBaseConnector
    original_register = nixl_base_connector.register_kv_caches
    original_register_cross_layers = nixl_base_connector.register_cross_layers_kv_cache

    def has_deferred_kv_backing() -> bool:
        """Fail closed when scratch-KV state cannot be determined."""
        try:
            kv_mgr = get_gms_client_memory_manager("kv_cache")
            return kv_mgr is not None and is_scratch(kv_mgr)
        except (LookupError, AttributeError, RuntimeError) as exc:
            logger.warning(
                "[GMS Patch] Cannot determine deferred-KV state — "
                "raising to avoid pinning a stale scratch MR: %s",
                exc,
                exc_info=True,
            )
            raise

    def patched_register_kv_caches(self, kv_caches):
        if has_deferred_kv_backing():
            self._scratch_kv_pending = kv_caches
            logger.info(
                "[GMS Patch] Deferring NIXL KV cache registration "
                "(stashed %d layers for wake replay)",
                len(kv_caches),
            )
            return
        return original_register(self, kv_caches)

    def patched_register_cross_layers_kv_cache(self, kv_cache, attn_backend):
        if has_deferred_kv_backing():
            self._scratch_cross_layers_kv_pending = (kv_cache, attn_backend)
            logger.info(
                "[GMS Patch] Deferring NIXL cross-layer KV cache registration "
                "for wake replay"
            )
            return
        return original_register_cross_layers(self, kv_cache, attn_backend)

    nixl_base_connector.register_kv_caches = patched_register_kv_caches
    nixl_base_connector.register_cross_layers_kv_cache = (
        patched_register_cross_layers_kv_cache
    )
    _register_kv_caches_patched = True
    logger.info("[GMS Patch] Patched NixlBaseConnector KV registration")


# =============================================================================
# Patch application helper
# =============================================================================


def patch_kv_cache_pool_scope() -> None:
    """Scope the scratch mem-pool to init_kv_cache (the raw KV tensors) only.

    Keeps BlockTables / workspace / the block-table pointer tensor on real memory.
    Single-block scratch aliases everything in the pool onto one granule, so a KV
    write over that pointer tensor would corrupt the block-table gather kernel
    (-> illegal memory access).
    """
    global _kv_cache_pool_scope_patched

    if _kv_cache_pool_scope_patched:
        return

    try:
        import torch
        from gpu_memory_service.client.torch.allocator import gms_use_mem_pool
        from vllm.v1.worker.gpu import model_runner as gpu_model_runner

        original_init_kv_cache = gpu_model_runner.init_kv_cache
    except (ImportError, AttributeError) as exc:
        logger.debug("[GMS Patch] init_kv_cache pool-scope not available: %s", exc)
        return

    def patched_init_kv_cache(*args, **kwargs):
        # Installed only in shadow mode, so always scope to the pool. init_kv_cache
        # allocates on the worker's current device.
        assert torch.cuda.is_available(), "GMS scratch KV requires CUDA"
        device = torch.device("cuda", torch.cuda.current_device())
        with gms_use_mem_pool("kv_cache", device):
            return original_init_kv_cache(*args, **kwargs)

    gpu_model_runner.init_kv_cache = patched_init_kv_cache
    _kv_cache_pool_scope_patched = True
    logger.info(
        "[GMS Patch] Scoped scratch mem-pool to init_kv_cache (KV tensors only)"
    )


def apply_scratch_kv_patches() -> None:
    """Apply scratch-KV monkey-patches. No-ops when scratch KV is disabled."""
    if not is_scratch_kv_enabled():
        return

    # Resolve the optional connector before mutating the other scratch-specific
    # vLLM entry points. A broken installed NIXL module must fail startup rather
    # than leave a partially applied scratch configuration.
    patch_register_kv_caches()
    patch_request_memory()
    patch_kv_cache_pool_scope()
    logger.info("[GMS Patch] applied")
