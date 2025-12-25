# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Monkey-patch upstream torch_memory_saver to support GPU Memory Service mode.

This module patches the upstream torch_memory_saver package to add support for
the "gpu_memory_service" hook mode used by Dynamo's GPU Memory Service.

IMPORTANT: This module must be imported BEFORE any torch_memory_saver usage.
The patching happens at import time, so simply importing this module is sufficient.

Usage:
    # At the top of your module, before any torch_memory_saver imports:
    import dynamo.sglang.gpu_memory_service_adapters.torch_memory_saver_patch  # noqa: F401

    # Now torch_memory_saver will support GPU Memory Service mode
    from torch_memory_saver import torch_memory_saver
"""

import logging
import os

logger = logging.getLogger(__name__)

_patched = False


def _is_gpu_memory_service_mode() -> bool:
    """Check if GPU Memory Service mode is enabled via environment variable."""
    return (
        os.environ.get("GPU Memory Service_SGLANG_AUTO_REGISTER") == "1"
        or os.environ.get("GPU Memory Service_VLLM_AUTO_REGISTER") == "1"
    )


def patch_torch_memory_saver():
    """Patch torch_memory_saver to support GPU Memory Service mode.

    This function is idempotent - calling it multiple times has no effect.
    """
    global _patched
    if _patched:
        return

    try:
        import torch_memory_saver.entrypoint as entrypoint_module
    except ImportError:
        logger.debug(
            "[GPU Memory Service Patch] torch_memory_saver not installed, skipping patch"
        )
        return

    # Store reference to original method
    original_ensure_initialized = entrypoint_module.TorchMemorySaver._ensure_initialized

    def patched_ensure_initialized(self):
        """Patched _ensure_initialized that supports GPU Memory Service mode."""
        # Check if already initialized
        if self._impl is not None:
            return

        # Check if GPU Memory Service mode is enabled
        hook_mode = self._impl_ctor_kwargs.get("hook_mode")
        if hook_mode == "gpu_memory_service" or (
            hook_mode is None and _is_gpu_memory_service_mode()
        ):
            # Use our GPU Memory Service implementation
            from dynamo.sglang.gpu_memory_service_adapters.torch_memory_saver_impl import (
                GPUMemoryServiceMemorySaverImpl,
            )

            # Set _impl directly since all TorchMemorySaver methods access self._impl
            self._impl = GPUMemoryServiceMemorySaverImpl()
            logger.info(
                "[TorchMemorySaver] Using GPU Memory Service mode for VA-stable sleep/wake"
            )
            del self._impl_ctor_kwargs
        else:
            # Fall back to original implementation
            original_ensure_initialized(self)

    # Patch the method
    entrypoint_module.TorchMemorySaver._ensure_initialized = patched_ensure_initialized

    _patched = True
    logger.debug(
        "[GPU Memory Service Patch] Successfully patched torch_memory_saver for GPU Memory Service mode"
    )


# Auto-patch on import
patch_torch_memory_saver()
