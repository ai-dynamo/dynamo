# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM monkey-patches applied at GMSWorker import.

Patches:
  - MemorySnapshot.measure: adds GMS-committed bytes to free_memory in RO mode.

The torch.cuda.empty_cache patch lives in integrations/common/patches.py.
"""

from __future__ import annotations

import logging

from gpu_memory_service.client.torch.allocator import get_gms_client_memory_manager
from gpu_memory_service.common.locks import GrantedLockType
from gpu_memory_service.integrations.vllm.kv_identity import allocation_engine_id

logger = logging.getLogger(__name__)

_memory_snapshot_patched = False


def patch_memory_snapshot() -> None:
    """Add committed GMS bytes to MemorySnapshot.free_memory."""
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

        committed_bytes = 0
        if manager.granted_lock_type == GrantedLockType.RO:
            allocations = manager.list_handles()
            committed_bytes += sum(alloc.aligned_size for alloc in allocations)
        else:
            # In write mode the engine should account for the weights it loaded.
            logger.info("[GMS] RW mode - skipping committed weight memory adjustment")

        kv_manager = get_gms_client_memory_manager("kv_pool")
        if kv_manager is not None and kv_manager.is_connected:
            try:
                device = getattr(self.device_, "index", None)
                if device is None:
                    device = kv_manager.device
                engine_id = allocation_engine_id(int(device))
                persistent = kv_manager.list_persistent(engine_id=engine_id)
                kv_bytes = sum(alloc.aligned_size for alloc in persistent)
                committed_bytes += kv_bytes
                if kv_bytes > 0:
                    logger.info(
                        "[GMS Patch] Accounting for %.2f GiB existing persistent KV",
                        kv_bytes / (1 << 30),
                    )
            except Exception as exc:
                logger.debug(
                    "[GMS Patch] Persistent KV memory accounting skipped: %s", exc
                )

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
