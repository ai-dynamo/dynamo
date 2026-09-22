# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM memory-accounting patch applied at GMSWorker import."""

from __future__ import annotations

import logging

from gpu_memory_service.client.torch.allocator import get_gms_client_memory_manager
from gpu_memory_service.common.locks import GrantedLockType

logger = logging.getLogger(__name__)

_memory_snapshot_patched = False


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
