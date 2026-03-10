# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM-specific patches for GPU Memory Service integration.

This module contains vLLM-specific patches that are applied when the GMSWorker
module is imported:
- MemorySnapshot.measure patch (adjusts free memory for read mode)

Note: The torch.cuda.empty_cache patch is in integrations/common/patches.py

Shadow mode patches (applied when SHADOW_SKIP_KV_CACHE=1):
- request_memory patch (bypasses memory check for shadow engines)
- register_kv_caches patch (skips NIXL registration when no KV cache)
- initialize_kv_cache_tensors patch (no-ops during shadow init phase)
- _get_slot_mappings patch (returns None when kv_caches empty, enabling
  CUDA graph capture without KV caches during shadow init)
- allocate_kv_cache_on_wake method (allocates KV cache on wake)

Shadow Mode State Management:
-----------------------------
Shadow mode uses a `_shadow_init_phase` attribute on model_runner to control
patch behavior:

1. GMSWorker.load_model() sets model_runner._shadow_init_phase = True
2. Patches check this flag to decide whether to no-op
3. GMSWorker.wake_up() clears the flag before allocating KV cache
"""

from __future__ import annotations

import logging
import os

from gpu_memory_service import get_gms_client_memory_manager
from gpu_memory_service.common.types import GrantedLockType

logger = logging.getLogger(__name__)

# =============================================================================
# Patch state tracking (to prevent double-patching)
# =============================================================================

_memory_snapshot_patched = False
_request_memory_patched = False
_register_kv_caches_patched = False
_initialize_kv_cache_tensors_patched = False
_get_slot_mappings_patched = False
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


def patch_memory_snapshot() -> None:
    """Patch MemorySnapshot.measure to add committed bytes to free_memory."""
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

        if manager.granted_lock_type == GrantedLockType.RO:
            allocations = manager.list_handles()
            committed_bytes = sum(alloc.get("aligned_size", 0) for alloc in allocations)
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
    """Patch request_memory to bypass memory check in shadow mode.

    vLLM's request_memory() checks that free_memory >= requested_memory at
    startup. This fails for shadow engines because the primary engine is
    consuming GPU memory for its KV cache.

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

        return original_request_memory(init_snapshot, cache_config)

    worker_utils.request_memory = patched_request_memory
    _request_memory_patched = True
    logger.info("[GMS Patch] Patched request_memory for shadow mode support")


def patch_register_kv_caches() -> None:
    """Patch NixlConnector.register_kv_caches to no-op when empty.

    When shadow engines skip KV cache allocation, calling
    register_kv_caches({}) with empty caches causes errors.
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

    Checks _shadow_init_phase flag on model_runner:
    - True: Store config for later, return empty dict
    - False (or not set): Proceed normally

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
        if getattr(self, "_shadow_init_phase", False):
            self._shadow_kv_cache_config = kv_cache_config
            self._shadow_kernel_block_sizes = kernel_block_sizes
            logger.info(
                "[Shadow] Init phase: stored config, skipping KV cache allocation"
            )
            return {}

        return original_initialize_kv_cache_tensors(
            self, kv_cache_config, kernel_block_sizes
        )

    GPUModelRunner.initialize_kv_cache_tensors = patched_initialize_kv_cache_tensors
    _initialize_kv_cache_tensors_patched = True
    logger.info("[GMS Patch] Patched GPUModelRunner.initialize_kv_cache_tensors")


def patch_get_slot_mappings() -> None:
    """Patch GPUModelRunner._get_slot_mappings to return None when KV caches are empty.

    In vLLM v0.15.x, _get_slot_mappings computes slot mappings that are passed
    to the unified_kv_cache_update splitting op via ForwardContext. When KV cache
    tensors haven't been allocated (shadow init), the op tries to unbind an empty
    KV cache tensor and crashes.

    By returning (None, None) when self.kv_caches is empty, set_forward_context
    defaults slot_mapping to {}, and unified_kv_cache_update gracefully skips
    the KV write (it checks `slot_mapping.get(layer_name) is not None`).

    This allows compile_or_warm_up_model to run during shadow init, capturing
    CUDA graphs without KV caches. The graphs only contain MLP/norm regions
    (PIECEWISE mode), so they remain valid after KV caches are allocated on wake.
    """
    global _get_slot_mappings_patched

    if _get_slot_mappings_patched:
        return

    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except ImportError:
        logger.debug("[GMS Patch] GPUModelRunner not available")
        return

    original_get_slot_mappings = GPUModelRunner._get_slot_mappings

    def patched_get_slot_mappings(self, *args, **kwargs):
        """Return (None, None) when KV caches haven't been allocated."""
        if not self.kv_caches:
            return None, None
        return original_get_slot_mappings(self, *args, **kwargs)

    GPUModelRunner._get_slot_mappings = patched_get_slot_mappings
    _get_slot_mappings_patched = True
    logger.info("[GMS Patch] Patched GPUModelRunner._get_slot_mappings")


def patch_allocate_kv_cache_on_wake() -> None:
    """Add allocate_kv_cache_on_wake method to GPUModelRunner.

    Shadow engines skip KV cache allocation at startup. When they wake up,
    this method allocates the full KV cache using the config stored during init.
    """
    global _allocate_kv_cache_on_wake_added

    if _allocate_kv_cache_on_wake_added:
        return

    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except ImportError:
        logger.debug("[GMS Patch] GPUModelRunner not available")
        return

    if hasattr(GPUModelRunner, "allocate_kv_cache_on_wake"):
        logger.debug("[GMS Patch] allocate_kv_cache_on_wake already exists")
        return

    def allocate_kv_cache_on_wake(self) -> dict:
        """Allocate KV cache tensors on wake using stored config."""
        import time

        import torch

        assert hasattr(self, "_shadow_kv_cache_config"), (
            "_shadow_kv_cache_config not set. "
            "Was _shadow_init_phase=True during initialize_kv_cache?"
        )
        assert hasattr(self, "_shadow_kernel_block_sizes"), (
            "_shadow_kernel_block_sizes not set. "
            "Was _shadow_init_phase=True during initialize_kv_cache?"
        )

        config = self._shadow_kv_cache_config

        # Estimate memory needed for KV cache allocation
        needed_bytes = sum(
            group.kv_cache_spec.page_size_bytes * config.num_blocks
            for group in config.kv_cache_groups
        )

        # Poll until enough GPU memory is free. This handles the case where
        # the previous engine's processes haven't fully released GPU memory
        # yet (e.g., NCCL timeout, container termination delay).
        free_bytes = torch.cuda.mem_get_info()[0]
        if free_bytes < needed_bytes:
            logger.info(
                "[Shadow] Waiting for GPU memory before KV cache allocation "
                "(need %.2f GiB, free %.2f GiB)",
                needed_bytes / (1 << 30),
                free_bytes / (1 << 30),
            )
            while free_bytes < needed_bytes:
                time.sleep(0.5)
                free_bytes = torch.cuda.mem_get_info()[0]
            logger.info(
                "[Shadow] GPU memory available (free %.2f GiB), proceeding",
                free_bytes / (1 << 30),
            )

        logger.info("[Shadow] Allocating KV cache on wake")

        # vLLM 0.16+ requires the config context for internal operations
        # (e.g., attention backend selection) that run during KV cache init.
        from vllm.config import set_current_vllm_config

        with set_current_vllm_config(self.vllm_config):
            kv_caches = self.initialize_kv_cache_tensors(
                config,
                self._shadow_kernel_block_sizes,
            )

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


def patch_cudagraph_mode_escalation() -> None:
    """Prevent vLLM from escalating cudagraph_mode beyond PIECEWISE in shadow mode.

    vLLM's _check_and_update_cudagraph_mode resolves cudagraph_mode based on
    attention backend capabilities and may escalate PIECEWISE to
    FULL_AND_PIECEWISE. This escalation happens after initialize_attn_backend
    and right before initialize_cudagraph_keys. FULL mode graph capture
    requires slot_mappings which don't exist when KV cache is skipped.

    This patch wraps _check_and_update_cudagraph_mode to clamp the resolved
    mode back to PIECEWISE when in shadow init with empty KV caches.
    """
    if not _is_shadow_mode():
        return

    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except ImportError:
        return

    if getattr(GPUModelRunner, "_shadow_cg_escalation_patched", False):
        return

    from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

    original_init_keys = CudagraphDispatcher.initialize_cudagraph_keys

    def patched_initialize_cudagraph_keys(self, cudagraph_mode, *args, **kwargs):
        if _is_shadow_mode():
            from vllm.config import CUDAGraphMode

            if cudagraph_mode not in (CUDAGraphMode.PIECEWISE, CUDAGraphMode.NONE):
                logger.info(
                    "[Shadow] Clamping cudagraph_mode from %s to PIECEWISE "
                    "(FULL captures require KV cache)",
                    cudagraph_mode.name,
                )
                cudagraph_mode = CUDAGraphMode.PIECEWISE
        return original_init_keys(self, cudagraph_mode, *args, **kwargs)

    CudagraphDispatcher.initialize_cudagraph_keys = patched_initialize_cudagraph_keys
    GPUModelRunner._shadow_cg_escalation_patched = True
    logger.info("[GMS Patch] Patched cudagraph mode escalation for shadow mode")


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
    patch_get_slot_mappings()
    patch_allocate_kv_cache_on_wake()
    patch_cudagraph_mode_escalation()
    logger.info("[GMS Patch] Shadow mode patches applied")
