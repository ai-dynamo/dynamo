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
from gpu_memory_service.client.torch.allocator import gms_use_mem_pool
from gpu_memory_service.common.types import RequestedLockType
from gpu_memory_service.common.utils import get_socket_path
from gpu_memory_service.integrations.common import patch_empty_cache
from gpu_memory_service.integrations.common.utils import get_gms_lock_mode
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

        # Establish weights GMS connection (so MemorySnapshot can query committed bytes).
        # Fetch extra config from vLLM load_config to determine RW/RO lock mode.
        extra = (
            getattr(self.vllm_config.load_config, "model_loader_extra_config", {}) or {}
        )
        socket_path = get_socket_path(device, "weights")
        get_or_create_gms_client_memory_manager(
            socket_path,
            device,
            mode=get_gms_lock_mode(extra),
            tag="weights",
        )
        # Parent will set device again (harmless) and do memory checks
        super().init_device()

    def initialize_from_config(self, kv_cache_config) -> None:
        """Allocate KV cache with a dedicated RW-only GMS tag."""
        from vllm.distributed.kv_transfer import ensure_kv_transfer_initialized

        ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)

        if not self.vllm_config.model_config.enable_sleep_mode:
            self.model_runner.initialize_kv_cache(kv_cache_config)
            return

        device = self.local_rank
        get_or_create_gms_client_memory_manager(
            get_socket_path(device, "kv_cache"),
            device,
            mode=RequestedLockType.RW,
            tag="kv_cache",
        )
        with gms_use_mem_pool("kv_cache", torch.device(f"cuda:{device}")):
            self.model_runner.initialize_kv_cache(kv_cache_config)

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

        NOTE: We do NOT call super().sleep() because it tries to copy GPU buffers to CPU,
              which segfaults on already-unmapped GMS memory.
        """
        free_bytes_before = torch.cuda.mem_get_info()[0]

        # Unmap GMS weights: synchronize + unmap all VAs + disconnect
        weights_manager = get_gms_client_memory_manager("weights")
        assert weights_manager is not None, "GMS weights client is not initialized"
        assert not weights_manager.is_unmapped, "GMS weights are already unmapped"
        weights_manager.unmap_all_vas()
        weights_manager.abort()

        kv_cache_manager = get_gms_client_memory_manager("kv_cache")
        assert kv_cache_manager is not None, "GMS KV cache client is not initialized"
        assert not kv_cache_manager.is_unmapped, "GMS KV cache is already unmapped"
        kv_cache_manager.unmap_all_vas()
        kv_cache_manager.abort()

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
        if tags is None:
            tags = ["weights", "kv_cache"]

        if "weights" in tags:
            weights_manager = get_gms_client_memory_manager("weights")
            assert weights_manager is not None, "GMS weights client is not initialized"
            assert weights_manager.is_unmapped, "GMS weights are not unmapped"
            weights_manager.connect(RequestedLockType.RO)
            weights_manager.remap_all_vas()

        if "kv_cache" in tags:
            kv_cache_manager = get_gms_client_memory_manager("kv_cache")
            assert (
                kv_cache_manager is not None
            ), "GMS KV cache client is not initialized"
            assert kv_cache_manager.is_unmapped, "GMS KV cache is not unmapped"
            kv_cache_manager.connect(RequestedLockType.RW)
            kv_cache_manager.reallocate_all_handles(tag="kv_cache")
            kv_cache_manager.remap_all_vas()

            # Reinitialize FP8 KV scales if needed
            if self.cache_config.cache_dtype.startswith("fp8") and hasattr(
                self.model_runner, "init_fp8_kv_scales"
            ):
                self.model_runner.init_fp8_kv_scales()

    def _maybe_get_memory_pool_context(self, tag: str):
        """Route tag-scoped runtime allocations to the right allocator.

        Weight tensors are allocated explicitly in the GMS model-loader path,
        not through vLLM's tagged runtime allocator hook. For `weights` we
        therefore only suppress CuMemAllocator here so it does not interfere
        with the loader-managed GMS allocations. `kv_cache` is the tag that
        actually allocates through this hook, so it uses the dedicated GMS
        mempool.
        """
        if tag == "weights":
            logger.debug("[GMS] Skipping CuMemAllocator for weights")
            return nullcontext()
        if tag == "kv_cache":
            return gms_use_mem_pool("kv_cache", torch.device("cuda", self.local_rank))
        return super()._maybe_get_memory_pool_context(tag)
