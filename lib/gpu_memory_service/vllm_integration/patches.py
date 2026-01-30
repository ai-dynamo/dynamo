# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utility patches for GPU Memory Service vLLM integration.

This module contains patches applied when the GMSWorker module is imported:

Core GMS patches (always applied):
- torch.cuda.empty_cache patch (prevents segfaults with VMM allocations)
- MemorySnapshot.measure patch (adjusts free memory for read mode)

Shadow mode patches (applied when SHADOW_SKIP_KV_CACHE=1):
- request_memory patch (bypasses memory check for shadow engines)
- register_kv_caches patch (skips NIXL registration when no KV cache)
- initialize_kv_cache_tensors patch (no-ops during shadow init phase)
- allocate_kv_cache_on_wake method (allocates KV cache on wake)

Shadow Mode State Management:
-----------------------------
Shadow mode uses a `_shadow_init_phase` attribute on model_runner to control
patch behavior:

1. GMSWorker.load_model() sets model_runner._shadow_init_phase = True
2. Patches check this flag to decide whether to no-op
3. GMSWorker.wake_up() clears the flag before allocating KV cache

This approach is cleaner than env vars because:
- State lives on the instance that patches operate on
- No string parsing or env var manipulation
- Easy to test and debug
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import torch
from gpu_memory_service import get_gms_client_memory_manager
from gpu_memory_service.common.types import GrantedLockType

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# Patch state tracking (to prevent double-patching)
# =============================================================================

_empty_cache_patched = False
_memory_snapshot_patched = False
_request_memory_patched = False
_register_kv_caches_patched = False
_initialize_kv_cache_tensors_patched = False
_allocate_kv_cache_on_wake_added = False


def _is_shadow_mode() -> bool:
    """Check if shadow mode is enabled via environment variable.

    This is used for patches that need to check at import/init time.
    For runtime behavior, patches should check model_runner._shadow_init_phase.
    """
    return os.environ.get("SHADOW_SKIP_KV_CACHE", "0") == "1"


# =============================================================================
# Core GMS patches (always applied)
# =============================================================================


def patch_empty_cache() -> None:
    """Patch torch.cuda.empty_cache to prevent segfaults with VMM allocations.

    Impact:
    -------
    When weights are allocated through our VMM-based pluggable allocator, calling
    torch.cuda.empty_cache() causes segfaults because the native caching allocator
    tries to release blocks that were allocated through VMM APIs.

    This patch makes empty_cache() a no-op when VMM allocations exist.

    Must be called at module import time before any empty_cache calls.
    """
    global _empty_cache_patched

    if _empty_cache_patched:
        return

    _original_empty_cache = torch.cuda.empty_cache

    def safe_empty_cache() -> None:
        """Safe replacement that skips when VMM allocations exist."""
        manager = get_gms_client_memory_manager()
        if manager is not None and len(manager.mappings) > 0:
            return
        _original_empty_cache()

    torch.cuda.empty_cache = safe_empty_cache
    _empty_cache_patched = True
    logger.info("[GMS Patch] Patched torch.cuda.empty_cache")


def patch_memory_snapshot() -> None:
    """Patch MemorySnapshot.measure to add committed bytes to free_memory.

    Impact:
    -------
    When a shadow engine starts with GMS in read-only mode, the weights are
    already committed to GPU memory by the primary engine. vLLM's memory
    snapshot would see this as "used" memory, causing incorrect free memory
    calculations.

    This patch adds the committed GMS bytes back to free_memory so vLLM
    correctly accounts for the shared weights.
    """
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

        manager = get_gms_client_memory_manager()
        assert manager is not None, "GMS client is not initialized"

        if manager.mode == GrantedLockType.RO:
            allocations = manager.list_allocations()
            committed_bytes = sum(alloc.get("aligned_size", 0) for alloc in allocations)
        else:
            # RW mode - assume we have the whole GPU
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
    """Patch request_memory to bypass memory check in shadow mode.

    Impact:
    -------
    vLLM's request_memory() checks that free_memory >= requested_memory at
    startup. This fails for shadow engines because the primary engine is
    consuming GPU memory for its KV cache.

    In shadow mode, we skip this check entirely because:
    1. Shadow engines share weights via GMS (already accounted for)
    2. Shadow engines skip KV cache allocation at startup
    3. Memory will be available when the primary dies and shadow wakes

    Note: This patch checks SHADOW_SKIP_KV_CACHE env var (not _shadow_init_phase)
    because it runs before model_runner exists.
    """
    global _request_memory_patched

    if _request_memory_patched:
        return

    try:
        from vllm.v1.worker import utils as worker_utils
    except ImportError:
        logger.debug("[GMS Patch] vllm.v1.worker.utils not available")
        return

    original_request_memory = worker_utils.request_memory

    def patched_request_memory(init_snapshot, cache_config):
        """Patched request_memory that skips check in shadow mode."""
        if _is_shadow_mode():
            # Shadow mode: bypass the memory check entirely
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

        # Normal mode: delegate to original
        return original_request_memory(init_snapshot, cache_config)

    worker_utils.request_memory = patched_request_memory
    _request_memory_patched = True
    logger.info("[GMS Patch] Patched request_memory for shadow mode support")


def patch_register_kv_caches() -> None:
    """Patch NixlConnector.register_kv_caches to no-op when empty.

    Impact:
    -------
    vLLM calls register_kv_caches() to register KV cache memory regions with
    NIXL for RDMA transfers. When shadow engines skip KV cache allocation,
    calling register_kv_caches({}) with empty caches causes errors.

    This patch makes register_kv_caches() a no-op when kv_caches is empty.
    When the shadow engine wakes and allocates KV cache, registration happens
    in allocate_kv_cache_on_wake().
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
        """Patched register_kv_caches that skips when empty."""
        if not kv_caches:
            logger.info("[GMS Patch] Skipping KV cache registration (empty kv_caches)")
            return
        return original_register(self, kv_caches)

    NixlConnector.register_kv_caches = patched_register_kv_caches
    _register_kv_caches_patched = True
    logger.info("[GMS Patch] Patched NixlConnector.register_kv_caches")


def patch_initialize_kv_cache_tensors() -> None:
    """Patch GPUModelRunner.initialize_kv_cache_tensors to no-op during shadow init.

    Impact:
    -------
    During shadow engine initialization, we want to skip KV cache allocation
    to minimize GPU memory footprint. This patch checks the _shadow_init_phase
    flag on self (model_runner):

    - _shadow_init_phase = True: Store config for later, return empty dict
    - _shadow_init_phase = False (or not set): Proceed normally

    The flag is set by GMSWorker.load_model() and cleared by GMSWorker.wake_up().
    """
    global _initialize_kv_cache_tensors_patched

    if _initialize_kv_cache_tensors_patched:
        return

    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except ImportError:
        logger.debug("[GMS Patch] GPUModelRunner not available")
        return

    original_initialize_kv_cache_tensors = GPUModelRunner.initialize_kv_cache_tensors

    def patched_initialize_kv_cache_tensors(self, kv_cache_config, kernel_block_sizes):
        """Patched initialize_kv_cache_tensors that no-ops during shadow init."""
        # Check if we're in shadow init phase
        if getattr(self, "_shadow_init_phase", False):
            # Shadow init phase: store config for wake, return empty
            self._shadow_kv_cache_config = kv_cache_config
            self._shadow_kernel_block_sizes = kernel_block_sizes
            logger.info(
                "[Shadow] Init phase: stored config, skipping KV cache allocation"
            )
            return {}

        # Normal operation (or wake): proceed with actual allocation
        return original_initialize_kv_cache_tensors(
            self, kv_cache_config, kernel_block_sizes
        )

    GPUModelRunner.initialize_kv_cache_tensors = patched_initialize_kv_cache_tensors
    _initialize_kv_cache_tensors_patched = True
    logger.info("[GMS Patch] Patched GPUModelRunner.initialize_kv_cache_tensors")


def patch_allocate_kv_cache_on_wake() -> None:
    """Add allocate_kv_cache_on_wake method to GPUModelRunner.

    Impact:
    -------
    Shadow engines skip KV cache allocation at startup. When they wake up,
    they need to allocate the KV cache before serving inference. This patch
    adds the allocate_kv_cache_on_wake() method that:

    1. Calls initialize_kv_cache_tensors() with stored config (now proceeds normally)
    2. Binds the KV cache tensors via bind_kv_cache()
    3. Registers with KV transfer group if enabled

    This method is called by GMSWorker.wake_up() when kv_caches is empty.
    """
    global _allocate_kv_cache_on_wake_added

    if _allocate_kv_cache_on_wake_added:
        return

    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except ImportError:
        logger.debug("[GMS Patch] GPUModelRunner not available")
        return

    # Check if method already exists
    if hasattr(GPUModelRunner, "allocate_kv_cache_on_wake"):
        logger.debug("[GMS Patch] allocate_kv_cache_on_wake already exists")
        return

    def allocate_kv_cache_on_wake(self) -> dict:
        """Allocate KV cache tensors on wake using stored config.

        This method is used for shadow engines that skip KV cache allocation
        at startup. When the engine wakes up, this method allocates the full
        KV cache using the config that was stored during init.

        Note: initialize_kv_cache_tensors already calls bind_kv_cache internally,
        so we don't need to call it again here.

        Returns:
            dict: The allocated kv_caches dictionary.

        Raises:
            AssertionError: If required config attributes are not set.
        """
        # Verify stored config exists
        assert hasattr(self, "_shadow_kv_cache_config"), (
            "_shadow_kv_cache_config not set. "
            "Was _shadow_init_phase=True during initialize_kv_cache?"
        )
        assert hasattr(self, "_shadow_kernel_block_sizes"), (
            "_shadow_kernel_block_sizes not set. "
            "Was _shadow_init_phase=True during initialize_kv_cache?"
        )

        logger.info("[Shadow] Allocating KV cache on wake")

        # Now _shadow_init_phase is False, so this proceeds normally
        # Note: initialize_kv_cache_tensors internally calls bind_kv_cache
        # which populates self.kv_caches
        kv_caches = self.initialize_kv_cache_tensors(
            self._shadow_kv_cache_config,
            self._shadow_kernel_block_sizes,
        )

        # Register with KV transfer group if enabled
        try:
            from vllm.distributed.kv_transfer.kv_connector.v1.base import (
                get_kv_transfer_group,
                has_kv_transfer_group,
            )

            if has_kv_transfer_group() and kv_caches:
                kv_transfer_group = get_kv_transfer_group()
                kv_transfer_group.register_kv_caches(kv_caches)
                logger.debug("[Shadow] Registered KV caches with transfer group")
        except ImportError:
            logger.debug("[Shadow] KV transfer group not available")

        # Log allocation summary
        total_bytes = sum(t.numel() * t.element_size() for t in kv_caches.values())
        logger.info(
            "[Shadow] Allocated KV cache on wake: %.2f GiB (%d tensors)",
            total_bytes / (1 << 30),
            len(kv_caches),
        )

        return kv_caches

    GPUModelRunner.allocate_kv_cache_on_wake = allocate_kv_cache_on_wake
    _allocate_kv_cache_on_wake_added = True
    logger.info("[GMS Patch] Added GPUModelRunner.allocate_kv_cache_on_wake")


# =============================================================================
# Patch application helper
# =============================================================================


def apply_shadow_mode_patches() -> None:
    """Apply all shadow mode patches.

    This should be called at module import time in worker.py. The patches
    check _shadow_init_phase at runtime, so they're safe to apply even for
    non-shadow engines (they'll just pass through to original methods).
    """
    patch_request_memory()
    patch_register_kv_caches()
    patch_initialize_kv_cache_tensors()
    patch_allocate_kv_cache_on_wake()
    logger.info("[GMS Patch] Shadow mode patches applied")
