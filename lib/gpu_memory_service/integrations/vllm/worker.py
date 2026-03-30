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
import sys
from contextlib import nullcontext
from typing import List, Optional

import torch
from gpu_memory_service import (
    get_gms_client_memory_manager,
    get_or_create_gms_client_memory_manager,
)
from gpu_memory_service.client.memory_manager import StaleMemoryLayoutError
from gpu_memory_service.common.types import RequestedLockType
from gpu_memory_service.common.utils import get_socket_path
from gpu_memory_service.integrations.common import patch_empty_cache
from gpu_memory_service.integrations.common.utils import get_gms_lock_mode
from gpu_memory_service.integrations.vllm.model_loader import register_gms_loader
from gpu_memory_service.integrations.vllm.patches import (
    apply_shadow_mode_patches,
    patch_memory_snapshot,
)
from gpu_memory_service.integrations.vllm.utils import is_shadow_mode

logger = logging.getLogger(__name__)

# Trigger model loader registration and utility patches on import
register_gms_loader()

# Apply core utility patches (always needed for GMS)
patch_empty_cache()
patch_memory_snapshot()

# Apply shadow mode patches if shadow mode is enabled
apply_shadow_mode_patches()

logger.info("[GMS] Worker module loaded - model loader registered, all patches applied")

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
        # Lock type is determined by model_loader_extra_config, set upstream by
        # configure_gms_lock_mode() in main.py.
        socket_path = get_socket_path(device)
        extra = (
            getattr(self.vllm_config.load_config, "model_loader_extra_config", {}) or {}
        )
        mode = get_gms_lock_mode(extra)
        get_or_create_gms_client_memory_manager(
            socket_path,
            device,
            mode=mode,
            tag="weights",
        )

        # Parent will set device again (harmless) and do memory checks
        super().init_device()

        # __class__ swap: preserves object identity so vLLM's internal
        # references see our overrides.
        if is_shadow_mode() and hasattr(self, "model_runner"):
            from gpu_memory_service.integrations.vllm.model_runner import (
                GMSShadowModelRunner,
            )

            self.model_runner.__class__ = GMSShadowModelRunner
            self.model_runner.enter_shadow_init()
            logger.info("[GMS] Injected GMSShadowModelRunner via __class__ swap")

    def determine_available_memory(self) -> int:
        """
        Determine actual available memory for the engine.

        During a failover scenario, this function may be called while there is an active engine colocated on the same device.
        We want our assessment to ignore the kv cache allocation of the active engine if there is one.
        """
        if not is_shadow_mode():
            return super().determine_available_memory()

        torch.cuda.reset_peak_memory_stats()
        self.model_runner.profile_run()
        torch.cuda.synchronize()
        torch_peak = torch.cuda.max_memory_allocated()

        # If weights are strictly loaded (RO), torch's memory accounting will miss them since we didn't go through the mempool
        # We therefore add in the memory of the weights into our accounting here
        # This is not an issue on engines that write the weights and then downgrade to RO
        weights_memory = int(getattr(self.model_runner, "model_memory_usage", 0))
        if torch_peak < weights_memory:
            non_kv_cache_memory = torch_peak + weights_memory
        else:
            non_kv_cache_memory = torch_peak

        projected_available = self.requested_memory - non_kv_cache_memory

        logger.info(
            "[GMS] Shadow mode: projected available memory "
            "%.2f GiB (requested=%.2f GiB, non_kv=%.2f GiB, "
            "torch_peak=%.2f GiB, weights=%.2f GiB)",
            projected_available / (1 << 30),
            self.requested_memory / (1 << 30),
            non_kv_cache_memory / (1 << 30),
            torch_peak / (1 << 30),
            weights_memory / (1 << 30),
        )

        return int(projected_available)

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

    def initialize_from_config(self, kv_cache_config) -> None:
        """Initialize from config with post-init cudagraph mode assertion.

        vLLM can try to upgrade the cudagraph mode in certain scenarios. We
        assert that the final resolved mode is still compatible with shadow mode.
        """
        super().initialize_from_config(kv_cache_config)
        if is_shadow_mode():
            from vllm.config import CUDAGraphMode

            mode = self.model_runner.compilation_config.cudagraph_mode
            if mode not in (CUDAGraphMode.PIECEWISE, CUDAGraphMode.NONE):
                raise RuntimeError(
                    f"Shadow mode requires PIECEWISE cudagraph mode after resolution, "
                    f"but got {mode.name}. vLLM's config resolution overrode it."
                )

    def sleep(self, level: int = 1) -> None:
        """
        vLLM sleep implementation with GMS integration.

        NOTE: `level` is a no-op here: weights are only unmapped (but remain in GPU memory).
        NOTE: We do NOT call super().sleep() because it tries to copy GPU buffers to CPU,
              which segfaults on already-unmapped GMS memory.

        Handles two cases for KV cache:
        1. Normal: KV cache was allocated, sleep via CuMemAllocator
        2. Shadow: KV cache was skipped at startup, nothing to do
        """
        free_bytes_before = torch.cuda.mem_get_info()[0]

        # Unmap GMS weights: synchronize + unmap all VAs + disconnect
        manager = get_gms_client_memory_manager()
        assert manager is not None, "GMS client is not initialized"
        assert not manager.is_unmapped, "GMS weights are already unmapped"
        manager.unmap_all_vas()
        manager.disconnect()

        # Sleep KV cache via CuMemAllocator (discard, no CPU backup)
        # If KV cache was never allocated (shadow engine mode), this is a no-op
        from vllm.device_allocator.cumem import CuMemAllocator

        kv_caches = getattr(self.model_runner, "kv_caches", None)
        if kv_caches:
            allocator = CuMemAllocator.get_instance()
            allocator.sleep(offload_tags=tuple())
        else:
            logger.info("[GMS] KV cache not allocated (shadow mode), skipping sleep")

        free_bytes_after, total = torch.cuda.mem_get_info()
        freed_bytes = free_bytes_after - free_bytes_before
        used_bytes = total - free_bytes_after
        logger.info(
            "Sleep freed %.2f GiB, %.2f GiB still in use.",
            freed_bytes / (1 << 30),
            used_bytes / (1 << 30),
        )

    def wake_up(self, tags: Optional[List[str]] = None) -> None:
        """vLLM wake implementation with GMS integration.

        Handles two cases for KV cache:
        1. Normal: KV cache was allocated at startup, reallocate via CuMemAllocator
        2. Shadow: KV cache was skipped at startup, allocate via allocate_kv_cache_on_wake()
        """
        if (
            hasattr(self.model_runner, "exit_shadow_init")
            and self.model_runner.in_shadow_init
        ):
            self.model_runner.exit_shadow_init()

        if tags is None:
            tags = ["weights", "kv_cache"]

        if "weights" in tags:
            manager = get_gms_client_memory_manager()
            assert manager is not None, "GMS client is not initialized"
            assert manager.is_unmapped, "GMS weights are not unmapped"

            # These errors are fatal and unrecoverable in a worker subprocess:
            # the worker cannot serve requests without weights. sys.exit(1)
            # ensures clean termination so the orchestrator (K8s) can restart.
            try:
                manager.connect(RequestedLockType.RO, timeout_ms=30_000)
                manager.remap_all_vas()
            except TimeoutError:
                logger.error(
                    "Fatal: timed out waiting for GMS RO lock during remap "
                    "(GMS may be down or RW lock held indefinitely)"
                )
                sys.exit(1)
            except StaleMemoryLayoutError as e:
                logger.error(
                    "Fatal: weight layout changed while unmapped, cannot remap: %s", e
                )
                sys.exit(1)
            except ConnectionError as e:
                logger.error("Fatal: cannot connect to GMS during remap: %s", e)
                sys.exit(1)

        if "kv_cache" in tags:
            # Check if KV cache was skipped at startup (shadow engine mode)
            kv_caches = getattr(self.model_runner, "kv_caches", None)
            if not kv_caches:
                logger.info("[GMS] KV cache not allocated - allocating on wake")
                self.model_runner.allocate_kv_cache_on_wake()
                logger.info("[GMS] Successfully allocated KV cache on wake")
            else:
                # Normal case: KV cache was allocated, reallocate via CuMemAllocator
                from vllm.device_allocator.cumem import CuMemAllocator

                allocator = CuMemAllocator.get_instance()
                allocator.wake_up(tags=["kv_cache"])

            # Reinitialize FP8 KV scales if needed
            if self.cache_config.cache_dtype.startswith("fp8") and hasattr(
                self.model_runner, "init_fp8_kv_scales"
            ):
                self.model_runner.init_fp8_kv_scales()

    def _maybe_get_memory_pool_context(self, tag: str):
        """Skip CuMemAllocator for weights when using GMS.

        GMS manages its own memory pool for weights, so we don't want vLLM's
        CuMemAllocator to interfere.
        """
        if tag == "weights":
            logger.debug("[GMS] Skipping CuMemAllocator for weights")
            return nullcontext()
        return super()._maybe_get_memory_pool_context(tag)
