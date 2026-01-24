# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU Memory Service Worker subclass for vLLM integration.

This module provides a custom Worker class that properly integrates with
GPU Memory Service for VA-stable weight sharing and sleep/wake functionality.

Usage:
    Set --worker-cls=gpu_memory_service.client.vllm_integration.worker:GMSWorker
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import TYPE_CHECKING, List, Optional

import torch

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# Trigger model loader registration and utility patches on import
from gpu_memory_service.client.vllm_integration import (  # noqa: E402
    register_gms_loader,
)
from gpu_memory_service.client.vllm_integration.patches import (  # noqa: E402
    patch_empty_cache,
    patch_memory_snapshot,
)

# Register model loader
register_gms_loader()

# Apply utility patches
patch_empty_cache()
patch_memory_snapshot()

logger.info(
    "[GMS] Worker module loaded - model loader registered, utility patches applied"
)


# Import Worker after patches are applied
from vllm.v1.worker.gpu_worker import Worker  # noqa: E402


class GMSWorker(Worker):
    """vLLM Worker subclass with GPU Memory Service integration.

    This Worker provides:
    - Early GMS connection establishment (before memory checks)
    - VA-stable sleep/wake for model weights
    - Correct memory accounting for imported weights
    - Proper memory pool context handling for GMS allocations
    """

    def init_device(self) -> None:
        """Initialize device with early GMS connection.

        We need to establish the GMS connection before the memory snapshot
        check runs, so that MemorySnapshot.measure can query committed bytes
        and adjust free_memory accordingly.
        """
        from gpu_memory_service.client.vllm_integration.memory_ops import (
            establish_early_gms_connection,
        )
        from vllm.platforms import current_platform

        # Compute device index using vLLM's exact logic
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

        # Set CUDA device before GMS connection
        device = torch.device(f"cuda:{local_rank}")
        current_platform.set_device(device)
        logger.debug("[GMS] Pre-set CUDA device %d before GMS connection", local_rank)

        # Establish GMS connection (so MemorySnapshot can query committed bytes)
        establish_early_gms_connection()

        # Call parent init_device (will set device again, then check memory)
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
                    "[GMS] Corrected model_memory_usage: %.2f GiB -> %.2f GiB",
                    old_usage / (1 << 30),
                    imported_bytes / (1 << 30),
                )
        except Exception as e:
            logger.debug("[GMS] Could not correct memory accounting: %s", e)

    def sleep(self, level: int = 1) -> None:
        """VA-stable sleep for GPU Memory Service.

        With GMS, sleep means:
        1. GMS allocator.sleep() - unmaps weights (VA preserved)
        2. CuMemAllocator.sleep() - discards KV cache (no CPU backup)

        We do NOT call super().sleep() because it tries to copy GPU buffers
        to CPU, which segfaults on already-unmapped GMS memory.
        """
        from gpu_memory_service.client.vllm_integration.memory_ops import (
            sleep_gms_weights,
            sleep_kv_cache,
        )

        free_bytes_before = torch.cuda.mem_get_info()[0]

        # Sleep GMS weights (VA-stable unmap)
        gms_slept = sleep_gms_weights()
        if gms_slept:
            logger.info("[GMS] VA-stable slept weights")

        # Sleep KV cache (discard, no CPU backup)
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

    def wake_up(self, tags: Optional[List[str]] = None) -> None:
        """VA-stable wake for GPU Memory Service.

        With GMS, wake means:
        1. GMS allocator.wake() - remaps weights to same VAs
        2. CuMemAllocator.wake_up() - reallocates KV cache memory

        We do NOT call super().wake_up() - we manage allocators directly.
        """
        from gpu_memory_service.client.vllm_integration.memory_ops import (
            reinit_fp8_kv_scales,
            wake_gms_weights,
            wake_kv_cache,
        )

        if tags is None:
            tags = ["weights", "kv_cache"]

        # Wake GMS weights (VA-stable remap)
        if "weights" in tags:
            try:
                gms_woke = wake_gms_weights()
                if gms_woke:
                    logger.info("[GMS] VA-stable woke weights")
            except Exception as e:
                logger.error("[GMS] Failed to wake weights: %s", e)
                raise

        # Wake KV cache
        if "kv_cache" in tags:
            wake_kv_cache()
            # Reinitialize FP8 KV scales if needed
            reinit_fp8_kv_scales(self)

    def _maybe_get_memory_pool_context(self, tag: str):
        """Skip CuMemAllocator for weights when using GMS.

        GMS manages its own memory pool for weights, so we don't want
        vLLM's CuMemAllocator to interfere.
        """
        if tag == "weights":
            logger.debug("[GMS] Skipping CuMemAllocator for weights")
            return nullcontext()
        return super()._maybe_get_memory_pool_context(tag)
