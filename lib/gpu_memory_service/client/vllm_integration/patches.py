# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utility patches for GPU Memory Service vLLM integration.

This module contains non-Worker patches that are applied when the GMSWorker
module is imported:
- torch.cuda.empty_cache patch (prevents segfaults with VMM allocations)
- MemorySnapshot.measure patch (adjusts free memory for read mode)

Worker-level functionality (sleep/wake, init_device, load_model, etc.) is
handled by the GMSWorker subclass in worker.py, not by monkey-patching.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


# =============================================================================
# Patch state tracking
# =============================================================================

_empty_cache_patched = False
_memory_snapshot_patched = False


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
# MemorySnapshot.measure patch (free memory adjustment)
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
