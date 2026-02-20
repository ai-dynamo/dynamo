# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang-specific patches for GPU Memory Service integration.

- patch_torch_memory_saver: Routes to GMS hybrid implementation
- patch_model_runner: Fixes memory accounting with pre-loaded weights
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from gpu_memory_service.common.utils import get_socket_path

logger = logging.getLogger(__name__)

_torch_memory_saver_patched = False
_model_runner_patched = False


def patch_torch_memory_saver() -> None:
    """Patch torch_memory_saver to use GPU Memory Service implementation.

    This function is idempotent - calling it multiple times has no effect.
    This patch is only applied when GMSModelLoader is imported (load_format="gms").
    """
    global _torch_memory_saver_patched
    if _torch_memory_saver_patched:
        return

    try:
        import torch_memory_saver.entrypoint as entrypoint_module
    except ImportError:
        logger.debug("[GMS] torch_memory_saver not installed, skipping patch")
        return

    # Store reference to original method
    original_ensure_initialized = entrypoint_module.TorchMemorySaver._ensure_initialized

    def patched_ensure_initialized(self):
        """Patched _ensure_initialized that uses GPU Memory Service implementation."""
        # Check if already initialized
        if self._impl is not None:
            logger.debug("[GMS] TorchMemorySaver already initialized, skipping")
            return

        # Check hook_mode - use GMS for None or explicit "gms"
        hook_mode = self._impl_ctor_kwargs.get("hook_mode")
        logger.info(f"[GMS] TorchMemorySaver initializing with hook_mode={hook_mode}")

        if hook_mode is None or hook_mode == "gms":
            # Use our GPU Memory Service implementation
            from gpu_memory_service.integrations.sglang.memory_saver import (
                GMSMemorySaverImpl,
            )
            from torch_memory_saver.entrypoint import _TorchMemorySaverImpl

            # Get device from torch.cuda.current_device() (already set by SGLang)
            device_index = torch.cuda.current_device()

            # Resolve socket path from env or default
            socket_path = get_socket_path(device_index)

            # Create underlying torch impl for non-weights tags (KV cache etc.)
            torch_impl = _TorchMemorySaverImpl(hook_mode="torch")

            # Create GPU Memory Service impl
            gms_impl = GMSMemorySaverImpl(
                torch_impl=torch_impl,
                socket_path=socket_path,
                device_index=device_index,
            )

            # Set _impl directly (accessible via gms_impl property)
            self._impl = gms_impl
            logger.info(
                "[GMS] Using GMS mode (device=%d, socket=%s, mode=%s)",
                device_index,
                socket_path,
                gms_impl.get_mode(),
            )
            del self._impl_ctor_kwargs
        else:
            # Fall back to original implementation
            logger.info("[GMS] Using default torch_memory_saver hook mode")
            original_ensure_initialized(self)

    entrypoint_module.TorchMemorySaver._ensure_initialized = patched_ensure_initialized

    # Add property to access GMS impl directly from the singleton
    from gpu_memory_service.integrations.sglang.memory_saver import GMSMemorySaverImpl

    @property
    def gms_impl(self) -> Optional[GMSMemorySaverImpl]:
        """Get the GMS impl if installed, None otherwise."""
        if isinstance(self._impl, GMSMemorySaverImpl):
            return self._impl
        return None

    entrypoint_module.TorchMemorySaver.gms_impl = gms_impl

    _torch_memory_saver_patched = True
    logger.debug("[GMS] Patched torch_memory_saver")


def patch_model_runner() -> None:
    """Patch SGLang's ModelRunner to fix memory accounting with pre-loaded weights.

    When weights are pre-loaded via GMS (import-only mode), SGLang's min_per_gpu_memory
    captured before loading is lower than device total. This causes under-reservation
    of overhead memory in KV cache calculation.
    """
    global _model_runner_patched

    if _model_runner_patched:
        return

    try:
        from sglang.srt.model_executor.model_runner import ModelRunner
    except ImportError:
        logger.warning("[GMS] Could not import ModelRunner, skipping patch")
        return

    if hasattr(ModelRunner, "_gms_patched"):
        return

    original_init_memory_pool = ModelRunner.init_memory_pool

    def patched_init_memory_pool(self, *args, **kwargs):
        """Patched init_memory_pool that uses device total for overhead calculation."""
        from gpu_memory_service.integrations.sglang.memory_saver import (
            get_gms_memory_saver_impl,
        )

        impl = get_gms_memory_saver_impl()
        if impl is not None and impl.get_imported_weights_bytes() > 0:
            total_memory = torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).total_memory
            if hasattr(self, "min_per_gpu_memory"):
                old_value = self.min_per_gpu_memory
                self.min_per_gpu_memory = total_memory
                logger.info(
                    "[GMS] Adjusted min_per_gpu_memory: %.2f GiB -> %.2f GiB",
                    old_value / (1 << 30),
                    total_memory / (1 << 30),
                )

        return original_init_memory_pool(self, *args, **kwargs)

    ModelRunner.init_memory_pool = patched_init_memory_pool
    ModelRunner._gms_patched = True
    _model_runner_patched = True
    logger.info("[GMS] Patched ModelRunner.init_memory_pool")
