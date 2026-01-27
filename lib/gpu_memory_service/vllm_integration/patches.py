# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utility patches for GPU Memory Service vLLM integration.

This module contains non-Worker patches that are applied when the GMSWorker
module is imported:
- torch.cuda.empty_cache patch (prevents segfaults with VMM allocations)
- MemorySnapshot.measure patch (adjusts free memory for read mode)
"""

from __future__ import annotations

import logging

import torch
from gpu_memory_service import get_gms_client_memory_manager
from gpu_memory_service.common.types import GrantedLockType

logger = logging.getLogger(__name__)


_empty_cache_patched = False
_memory_snapshot_patched = False


def patch_empty_cache() -> None:
    """Patch torch.cuda.empty_cache to prevent segfaults with VMM allocations.

    Must be called at module import time before any empty_cache calls.
    """
    global _empty_cache_patched

    if _empty_cache_patched:
        return

    _original_empty_cache = torch.cuda.empty_cache

    def safe_empty_cache() -> None:
        """Safe replacement for torch.cuda.empty_cache that skips when VMM allocations exist.

        When weights are allocated through our VMM-based pluggable allocator, calling
        torch.cuda.empty_cache() causes segfaults because the native caching allocator
        tries to release blocks that were allocated through VMM APIs.
        """
        manager = get_gms_client_memory_manager()
        if manager is not None and len(manager.mappings) > 0:
            return

        _original_empty_cache()

    torch.cuda.empty_cache = safe_empty_cache
    _empty_cache_patched = True
    logger.info("[GMS Patch] Patched torch.cuda.empty_cache")


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

        if manager.mode == GrantedLockType.RO:
            allocations = manager.list_allocations()
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
