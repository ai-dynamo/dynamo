"""GPU Memory Service memory operations for vLLM integration.

This module contains the implementation logic for:
- Sleep/wake operations for GPU Memory Service weights
- KV cache management via CuMemAllocator
- Safe torch.cuda.empty_cache replacement

These operations are called by the patched Worker methods (see patches.py).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, List

import torch

from dynamo.vllm.gpu_memory_service_adapters.utils import get_gms_memory_manager

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# Safe empty_cache implementation
# =============================================================================

_original_empty_cache = torch.cuda.empty_cache


def safe_empty_cache() -> None:
    """Safe replacement for torch.cuda.empty_cache that skips when VMM allocations exist.

    When weights are allocated through our VMM-based pluggable allocator, calling
    torch.cuda.empty_cache() causes segfaults because the native caching allocator
    tries to release blocks that were allocated through VMM APIs.
    """
    from dynamo.vllm.gpu_memory_service_adapters.utils import has_vmm_allocations

    if has_vmm_allocations():
        logger.debug(
            "[GMS] Blocking torch.cuda.empty_cache() - VMM allocations would be destroyed"
        )
        return

    logger.debug("[GMS] Allowing empty_cache (no VMM allocations)")
    _original_empty_cache()


# =============================================================================
# GPU Memory Service weight operations
# =============================================================================

def sleep_gms_weights() -> bool:
    """Sleep GPU Memory Service weights (VA-stable).

    Unmaps physical memory while preserving VA reservations.
    Tensor pointers remain valid but memory is released.

    Returns:
        True if sleep was performed, False otherwise.
    """
    manager = get_gms_memory_manager()
    if manager is None:
        return False
    if manager.is_sleeping:
        return False

    try:
        manager.sleep()
        logger.debug("[GMS] Slept weights (VA-stable)")
        return True
    except Exception as e:
        logger.warning("[GMS] Failed to sleep weights: %s", e)
        return False


def wake_gms_weights() -> bool:
    """Wake GPU Memory Service weights (VA-stable).

    Remaps physical memory to the same VAs.
    Tensor pointers remain valid after wake.

    Returns:
        True if wake was performed, False otherwise.

    Raises:
        StaleMemoryLayoutError: If memory layout was changed while sleeping.
    """
    manager = get_gms_memory_manager()
    if manager is None:
        return False
    if not manager.is_sleeping:
        return False

    # Note: StaleMemoryLayoutError may propagate to caller
    manager.wake()

    # Synchronize to ensure all remapping operations are visible
    if torch.cuda.is_available():
        device = manager.device
        torch.cuda.synchronize(device)
        logger.debug("[GMS] CUDA synchronized on device %s after wake", device)

    logger.debug("[GMS] Woke weights (VA-stable)")
    return True


# =============================================================================
# KV cache operations via CuMemAllocator
# =============================================================================

def sleep_kv_cache() -> bool:
    """Sleep KV cache via CuMemAllocator.

    Discards all KV cache blocks (no CPU backup). This is appropriate for
    GPU Memory Service integration where we want to free GPU memory completely.

    Returns:
        True if sleep was performed, False otherwise.
    """
    try:
        from vllm.device_allocator.cumem import CuMemAllocator

        allocator = CuMemAllocator.get_instance()
        # Empty tuple = discard everything, no CPU backup
        allocator.sleep(offload_tags=tuple())
        logger.debug("[GMS] Slept KV cache via CuMemAllocator (discarded)")
        return True
    except Exception as e:
        logger.warning("[GMS] Failed to sleep KV cache: %s", e)
        return False


def wake_kv_cache() -> bool:
    """Wake KV cache via CuMemAllocator.

    Reallocates KV cache memory blocks.

    Returns:
        True if wake was performed, False otherwise.
    """
    try:
        from vllm.device_allocator.cumem import CuMemAllocator

        allocator = CuMemAllocator.get_instance()
        allocator.wake_up(tags=["kv_cache"])
        logger.debug("[GMS] Woke KV cache via CuMemAllocator")
        return True
    except Exception as e:
        logger.warning("[GMS] Failed to wake KV cache: %s", e)
        return False


# =============================================================================
# Combined sleep/wake operations (called by patched Worker methods)
# =============================================================================

def worker_sleep_impl(level: int = 1) -> None:
    """Implementation of Worker.sleep() for GPU Memory Service integration.

    This function is called by the patched Worker.sleep() method.

    With GPU Memory Service, sleep means:
    1. GPU Memory Service allocator.sleep() - unmaps weights (VA preserved)
    2. CuMemAllocator.sleep() - discards KV cache (no CPU backup)

    The 'level' parameter is ignored - we always sleep everything.

    Note: We do NOT call original Worker.sleep() because it tries to copy
    GPU buffers to CPU, which segfaults on already-unmapped GMS memory.
    """
    free_bytes_before = torch.cuda.mem_get_info()[0]

    # 1. Sleep GPU Memory Service weights
    gms_slept = sleep_gms_weights()
    if gms_slept:
        logger.info("[GMS] VA-stable slept weights")

    # 2. Sleep KV cache
    sleep_kv_cache()

    # Log memory freed
    free_bytes_after, total = torch.cuda.mem_get_info()
    freed_bytes = free_bytes_after - free_bytes_before
    used_bytes = total - free_bytes_after
    logger.info(
        "[GMS] Sleep freed %.2f GiB, %.2f GiB still in use",
        freed_bytes / (1 << 30),
        used_bytes / (1 << 30),
    )


def worker_wake_impl(tags: Optional[List[str]] = None) -> None:
    """Implementation of Worker.wake_up() for GPU Memory Service integration.

    This function is called by the patched Worker.wake_up() method.

    With GPU Memory Service, wake means:
    1. GPU Memory Service allocator.wake() - remaps weights to same VAs
    2. CuMemAllocator.wake_up() - reallocates KV cache memory

    The 'tags' parameter controls which resources to wake.

    Note: We do NOT call original Worker.wake_up() - we manage allocators directly.
    """
    if tags is None:
        tags = ["weights", "kv_cache"]

    # 1. Wake GPU Memory Service weights
    if "weights" in tags:
        try:
            gms_woke = wake_gms_weights()
            if gms_woke:
                logger.info("[GMS] VA-stable woke weights")
        except Exception as e:
            logger.error("[GMS] Failed to wake weights: %s", e)
            raise

    # 2. Wake KV cache
    if "kv_cache" in tags:
        wake_kv_cache()


def reinit_fp8_kv_scales(worker) -> None:
    """Reinitialize FP8 KV scales after wake if needed.

    Args:
        worker: vLLM Worker instance
    """
    try:
        if (
            hasattr(worker, "cache_config")
            and hasattr(worker.cache_config, "cache_dtype")
            and worker.cache_config.cache_dtype.startswith("fp8")
            and hasattr(worker, "model_runner")
            and hasattr(worker.model_runner, "init_fp8_kv_scales")
        ):
            worker.model_runner.init_fp8_kv_scales()
            logger.debug("[GMS] Reinitialized FP8 KV scales")
    except Exception as e:
        logger.debug("[GMS] FP8 KV scale reinit skipped: %s", e)


# =============================================================================
# GMS connection and query operations
# =============================================================================

def establish_early_gms_connection() -> bool:
    """Establish GMS connection early, before model loading.

    Called during worker initialization to establish a persistent GMS connection
    that is reused for:
    1. Memory check adjustments (before model loading)
    2. Model weight loading/importing
    3. Sleep/wake operations

    Returns:
        True if connection was established successfully.
    """
    import os
    from gpu_memory_service.common.types import RequestedLockType

    try:
        from gpu_memory_service.client.torch.allocator import (
            get_or_create_gms_client_memory_manager
        )

        from dynamo.vllm.gpu_memory_service_adapters.config import resolve_socket_path

        socket_path = resolve_socket_path()
        device = torch.cuda.current_device() if torch.cuda.is_available() else 0

        manager, pool = get_or_create_gms_client_memory_manager(
            socket_path, device, mode=RequestedLockType.RW_OR_RO, tag="weights"
        )
        granted_mode = manager.mode

        logger.debug(
            "[GMS] Early connection established (socket=%s, device=%d, mode=%s)",
            socket_path, device, granted_mode
        )
        return True
    except TimeoutError:
        logger.debug("[GMS] Early connection timed out - will retry during model loading")
        return False
    except ConnectionError as e:
        logger.debug("[GMS] Early connection failed (server not ready): %s", e)
        return False
    except Exception as e:
        logger.warning("[GMS] Early connection failed: %s", e)
        return False


def get_gms_committed_bytes() -> int:
    """Query committed bytes from GPU Memory Service.

    Used by the MemorySnapshot patch to account for weights that are already
    in GMS and will be imported (not newly allocated).

    Returns:
        Total committed bytes, or 0 if not applicable.
    """
    from gpu_memory_service.common.types import GrantedLockType

    manager = get_gms_memory_manager()
    if manager is None:
        logger.debug("[GMS] No manager registered - no adjustment")
        return 0

    if manager._client is None:
        logger.debug("[GMS] Manager has no client - no adjustment")
        return 0

    if manager.mode != GrantedLockType.RO:
        logger.debug("[GMS] Manager in write mode - no adjustment needed")
        return 0

    try:
        allocations = manager._client.list_allocations()
        total_bytes = sum(alloc.get("aligned_size", 0) for alloc in allocations)
        if total_bytes > 0:
            logger.debug(
                "[GMS] Queried committed bytes: %.2f GiB (%d allocations)",
                total_bytes / (1 << 30), len(allocations)
            )
        return total_bytes
    except Exception as e:
        logger.debug("[GMS] Failed to query committed bytes: %s", e)
        return 0
