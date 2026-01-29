# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU Memory Service Worker subclass for vLLM integration.

This module provides a custom Worker class that properly integrates with
GPU Memory Service for VA-stable weight sharing and unmap/remap functionality.

Usage:
    Set --worker-cls=gpu_memory_service.integrations.vllm.worker:GMSWorker
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import List, Optional

import torch
from gpu_memory_service import (
    get_gms_client_memory_manager,
    get_or_create_gms_client_memory_manager,
)
from gpu_memory_service.common.types import RequestedLockType
from gpu_memory_service.common.utils import get_socket_path
from gpu_memory_service.integrations.common import patch_empty_cache
from gpu_memory_service.integrations.vllm.model_loader import register_gms_loader
from gpu_memory_service.integrations.vllm.patches import patch_memory_snapshot

logger = logging.getLogger(__name__)

# Trigger model loader registration and utility patches on import
register_gms_loader()
patch_empty_cache()
patch_memory_snapshot()

logger.info(
    "[GMS] Worker module loaded - model loader registered, utility patches applied"
)

# Import Worker after patches are applied
from vllm.v1.worker.gpu_worker import Worker  # noqa: E402


class GMSWorker(Worker):
    """vLLM Worker subclass with GMS integration."""

    def init_device(self) -> None:
        """Initialize device with early GMS connection.

        We set CUDA device and establish GMS connection BEFORE calling super()
        so that MemorySnapshot.measure can query committed bytes.
        """
        from vllm.platforms import current_platform

        # Set CUDA device first (vLLM provides self.local_rank)
        device = self.local_rank
        current_platform.set_device(torch.device(f"cuda:{device}"))

        # Establish GMS connection (so MemorySnapshot can query committed bytes)
        socket_path = get_socket_path(device)
        get_or_create_gms_client_memory_manager(
            socket_path, device, mode=RequestedLockType.RW_OR_RO, tag="weights"
        )

        # Parent will set device again (harmless) and do memory checks
        super().init_device()

    def load_model(self, *args, **kwargs) -> None:
        """Load model with corrected memory accounting.

        After the parent loads the model, we correct the model_memory_usage
        to reflect the actual bytes imported from GMS (not the delta measured
        by vLLM's memory tracking).
        """
        super().load_model(*args, **kwargs)

        # Correct memory accounting for GMS-imported weights
        try:
            from gpu_memory_service.integrations.vllm.model_loader import (
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
                    "[GMS] Corrected model_memory_usage: %.2f GiB -> %.2f GiB",
                    old_usage / (1 << 30),
                    imported_bytes / (1 << 30),
                )
        except Exception as e:
            logger.debug("[GMS] Could not correct memory accounting: %s", e)

    def sleep(self, level: int = 1) -> None:
        """
        vLLM sleep implementation with GMS integration.

        NOTE: `level` is a no-op here: weights are only unmapped (but remain in GPU memory).
        NOTE: We do NOT call super().sleep() because it tries to copy GPU buffers to CPU,
              which segfaults on already-unmapped GMS memory.
        """
        from vllm.device_allocator.cumem import CuMemAllocator

        free_bytes_before = torch.cuda.mem_get_info()[0]

        # Unmap GMS weights (VA-stable unmap, no CPU backup needed)
        manager = get_gms_client_memory_manager()
        assert manager is not None, "GMS client is not initialized"
        assert not manager.is_unmapped, "GMS weights are already unmapped"
        manager.unmap()

        # Sleep KV cache via CuMemAllocator (discard, no CPU backup)
        allocator = CuMemAllocator.get_instance()
        allocator.sleep(offload_tags=tuple())

        # Ensure all CUDA VMM unmap operations complete before returning.
        # Without this sync, wake_up() may race with pending unmaps, causing OOM
        # when it tries to allocate new memory while old memory is still mapped.
        torch.cuda.synchronize()

        free_bytes_after, total = torch.cuda.mem_get_info()
        freed_bytes = free_bytes_after - free_bytes_before
        used_bytes = total - free_bytes_after
        logger.info(
            "Sleep freed %.2f GiB, %.2f GiB still in use.",
            freed_bytes / (1 << 30),
            used_bytes / (1 << 30),
        )

    def wake_up(self, tags: Optional[List[str]] = None) -> None:
        """vLLM wake implementation with GMS integration."""
        from vllm.device_allocator.cumem import CuMemAllocator

        if tags is None:
            tags = ["weights", "kv_cache"]

        if "weights" in tags:
            manager = get_gms_client_memory_manager()
            assert manager is not None, "GMS client is not initialized"
            assert manager.is_unmapped, "GMS weights are not unmapped"
            manager.remap()
            torch.cuda.synchronize()

        if "kv_cache" in tags:
            allocator = CuMemAllocator.get_instance()
            allocator.wake_up(tags=["kv_cache"])

            # Ensure KV cache mappings are complete before returning.
            # Without this sync, inference may start before mappings are ready.
            torch.cuda.synchronize()

            # Reinitialize FP8 KV scales if needed
            if self.cache_config.cache_dtype.startswith("fp8") and hasattr(
                self.model_runner, "init_fp8_kv_scales"
            ):
                self.model_runner.init_fp8_kv_scales()

    def _maybe_get_memory_pool_context(self, tag: str):
        """Skip CuMemAllocator for weights when using GMS.

        GMS manages its own memory pool for weights, so we don't want
        vLLM's CuMemAllocator to interfere.
        """
        if tag == "weights":
            logger.debug("[GMS] Skipping CuMemAllocator for weights")
            return nullcontext()
        return super()._maybe_get_memory_pool_context(tag)
