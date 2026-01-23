# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Monkey-patching logic for GPU Memory Service vLLM integration.

This module consolidates all monkey-patches in one place:
- torch.cuda.empty_cache patch (prevents segfaults with VMM allocations)
- Worker.load_model patch (corrects memory accounting)
- Worker.init_device patch (establishes early GMS connection)
- MemorySnapshot.measure patch (adjusts free memory for read mode)
- Worker._maybe_get_memory_pool_context patch (skips CuMemAllocator for weights)
- Worker.sleep/wake_up patches (VA-stable sleep/wake)

The actual implementation logic is in memory_ops.py.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import List, Optional

import torch

from gpu_memory_service.client.vllm_integration.config import GMS_ENABLED
from gpu_memory_service.client.vllm_integration.utils import get_vllm_worker_class

logger = logging.getLogger(__name__)


# =============================================================================
# Patch state tracking
# =============================================================================

_empty_cache_patched = False
_worker_patched = False
_memory_snapshot_patched = False
_sleep_wake_patched = False


# =============================================================================
# torch.cuda.empty_cache patch
# =============================================================================


def patch_empty_cache() -> None:
    """Patch torch.cuda.empty_cache to prevent segfaults with VMM allocations.

    Must be called at module import time before any empty_cache calls.
    """
    global _empty_cache_patched

    if _empty_cache_patched:
        return

    from gpu_memory_service.client.vllm_integration.memory_ops import safe_empty_cache

    torch.cuda.empty_cache = safe_empty_cache
    _empty_cache_patched = True

    logger.info("[GMS Patch] Patched torch.cuda.empty_cache for VMM safety")


# =============================================================================
# Worker.load_model patch (memory accounting)
# =============================================================================


def _create_load_model_patch(original_load_model):
    """Create the patched load_model function."""

    def patched_load_model(self):
        logger.debug("[GMS Patch] patched_load_model called")
        original_load_model(self)
        logger.debug("[GMS Patch] original load_model returned")

        try:
            from gpu_memory_service.client.vllm_integration.model_loader import (
                get_imported_weights_bytes,
            )

            imported_bytes = int(get_imported_weights_bytes())
            if (
                imported_bytes > 0
                and hasattr(self, "model_runner")
                and self.model_runner is not None
            ):
                old_usage = getattr(self.model_runner, "model_memory_usage", 0)
                self.model_runner.model_memory_usage = imported_bytes
                logger.info(
                    "[GMS Patch] Corrected model_memory_usage: %.2f GiB -> %.2f GiB",
                    old_usage / (1 << 30),
                    imported_bytes / (1 << 30),
                )
        except Exception:
            pass

    return patched_load_model


# =============================================================================
# Worker.init_device patch (early GMS connection)
# =============================================================================


def _create_init_device_patch(original_init_device):
    """Create the patched init_device function.

    The challenge is that:
    1. We need CUDA device set before establishing GMS connection
    2. We need GMS connection established before MemorySnapshot check
    3. original_init_device does both: sets CUDA device, then checks memory

    Solution: Replicate vLLM's device setup logic, establish GMS connection,
    then call original init_device.
    """

    def patched_init_device(self):
        from vllm.platforms import current_platform

        from gpu_memory_service.client.vllm_integration.memory_ops import (
            establish_early_gms_connection,
        )

        # Step 1: Compute device index using vLLM's exact logic from init_device
        # This replicates the DP adjustment logic for local_rank
        local_rank = self.local_rank
        if (
            self.parallel_config.data_parallel_size > 1
            and self.parallel_config.data_parallel_size_local > 0
            and self.parallel_config.distributed_executor_backend
            not in ["ray", "external_launcher"]
            and self.vllm_config.parallel_config.data_parallel_backend != "ray"
            and self.vllm_config.parallel_config.nnodes_within_dp == 1
        ):
            dp_local_rank = self.parallel_config.data_parallel_rank_local
            if dp_local_rank is None:
                dp_local_rank = self.parallel_config.data_parallel_rank
            tp_pp_world_size = (
                self.parallel_config.pipeline_parallel_size
                * self.parallel_config.tensor_parallel_size
            )
            local_rank += dp_local_rank * tp_pp_world_size

        # Step 2: Set CUDA device using vLLM's platform abstraction
        device = torch.device(f"cuda:{local_rank}")
        current_platform.set_device(device)
        logger.debug(
            "[GMS Patch] Pre-set CUDA device %d before GMS connection", local_rank
        )

        # Step 3: Establish GMS connection (now CUDA device is set)
        # This registers the manager so MemorySnapshot patch can query committed bytes
        establish_early_gms_connection()

        # Step 4: Call original init_device (will set device again, then check memory)
        # The MemorySnapshot.measure patch will now correctly adjust free_memory
        return original_init_device(self)

    return patched_init_device


# =============================================================================
# MemorySnapshot.measure patch (free memory adjustment)
# =============================================================================


def _patch_memory_snapshot() -> None:
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

        from gpu_memory_service.client.vllm_integration.memory_ops import (
            get_gms_committed_bytes,
        )

        committed_bytes = get_gms_committed_bytes()
        if committed_bytes > 0:
            original_free = self.free_memory
            self.free_memory += committed_bytes
            logger.info(
                "[GMS Patch] Adjusted free_memory: %.2f GiB + %.2f GiB = %.2f GiB",
                original_free / (1 << 30),
                committed_bytes / (1 << 30),
                self.free_memory / (1 << 30),
            )

    MemorySnapshot.measure = patched_measure
    _memory_snapshot_patched = True
    logger.info("[GMS Patch] Patched MemorySnapshot.measure for read mode")


# =============================================================================
# Worker._maybe_get_memory_pool_context patch
# =============================================================================


def _create_memory_pool_context_patch(original_fn):
    """Create the patched _maybe_get_memory_pool_context function."""

    def patched_get_memory_pool_context(self, tag: str):
        # When using GPU Memory Service, skip CuMemAllocator for weights
        if tag == "weights" and GMS_ENABLED:
            logger.debug("[GMS Patch] Skipping CuMemAllocator for weights")
            return nullcontext()
        return original_fn(self, tag)

    return patched_get_memory_pool_context


# =============================================================================
# Worker.sleep/wake_up patches
# =============================================================================


def _create_sleep_patch():
    """Create the patched sleep function."""

    def patched_sleep(self, level: int = 1):
        from gpu_memory_service.client.vllm_integration.memory_ops import worker_sleep_impl

        worker_sleep_impl(level)

    return patched_sleep


def _create_wake_up_patch():
    """Create the patched wake_up function."""

    def patched_wake_up(self, tags: Optional[List[str]] = None):
        from gpu_memory_service.client.vllm_integration.memory_ops import (
            reinit_fp8_kv_scales,
            worker_wake_impl,
        )

        worker_wake_impl(tags)

        # Reinitialize FP8 KV scales if needed
        if tags is None or "kv_cache" in tags:
            reinit_fp8_kv_scales(self)

    return patched_wake_up


# =============================================================================
# Main patch application functions
# =============================================================================


def patch_worker_for_gms() -> None:
    """Apply Worker patches for GPU Memory Service integration.

    Patches:
    - Worker.load_model - corrects model_memory_usage accounting
    - Worker.init_device - establishes early GMS connection
    - MemorySnapshot.measure - adjusts free_memory for read mode
    - Worker._maybe_get_memory_pool_context - skips CuMemAllocator for weights
    """
    global _worker_patched

    if _worker_patched:
        return

    Worker = get_vllm_worker_class()
    if Worker is None:
        logger.warning("[GMS Patch] Could not import vLLM Worker; patches not applied")
        return

    # Patch load_model
    if hasattr(Worker, "load_model"):
        Worker.load_model = _create_load_model_patch(Worker.load_model)
        logger.info("[GMS Patch] Patched Worker.load_model")

    # Patch init_device
    if hasattr(Worker, "init_device"):
        Worker.init_device = _create_init_device_patch(Worker.init_device)
        logger.info("[GMS Patch] Patched Worker.init_device")

    # Patch MemorySnapshot.measure
    _patch_memory_snapshot()

    # Patch _maybe_get_memory_pool_context
    if hasattr(Worker, "_maybe_get_memory_pool_context"):
        Worker._maybe_get_memory_pool_context = _create_memory_pool_context_patch(
            Worker._maybe_get_memory_pool_context
        )
        logger.info("[GMS Patch] Patched Worker._maybe_get_memory_pool_context")

    _worker_patched = True


def patch_sleep_wake() -> None:
    """Apply Worker.sleep/wake_up patches for GPU Memory Service integration.

    This patch is applied in the WORKER process (via model_loader.py) because
    the GPU Memory Service allocator is registered there.

    IMPORTANT: We do NOT call the original Worker.sleep()/wake_up() at all!
    The original methods try to copy GPU memory to CPU for backup, which causes
    segfaults when GPU Memory Service has already unmapped the weights.
    """
    global _sleep_wake_patched

    if _sleep_wake_patched:
        return

    Worker = get_vllm_worker_class()
    if Worker is None:
        logger.warning(
            "[GMS Patch] Could not import vLLM Worker; sleep/wake not patched"
        )
        return

    if not hasattr(Worker, "sleep") or not hasattr(Worker, "wake_up"):
        logger.debug("[GMS Patch] Worker does not have sleep/wake_up methods")
        return

    Worker.sleep = _create_sleep_patch()
    Worker.wake_up = _create_wake_up_patch()

    _sleep_wake_patched = True
    logger.info("[GMS Patch] Patched Worker.sleep() and Worker.wake_up()")
    logger.debug(
        "[GMS Patch] Sleep/wake: (1) GMS weights: VA-stable unmap/remap; "
        "(2) KV cache: discarded/reallocated; (3) Original methods NOT called"
    )


def apply_all_patches() -> None:
    """Apply all GPU Memory Service patches.

    This is the main entry point for applying patches. Call this once
    during plugin initialization.
    """
    patch_empty_cache()
    patch_worker_for_gms()
