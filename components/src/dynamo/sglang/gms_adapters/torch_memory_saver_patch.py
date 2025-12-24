# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Monkey-patch upstream torch_memory_saver to support GMS mode.

This module patches the upstream torch_memory_saver package to add support for
the "gms" hook mode used by Dynamo's GPU Memory Service.

IMPORTANT: This module must be imported BEFORE any torch_memory_saver usage.
The patching happens at import time, so simply importing this module is sufficient.

Usage:
    # At the top of your module, before any torch_memory_saver imports:
    import dynamo.sglang.gms_adapters.torch_memory_saver_patch  # noqa: F401

    # Now torch_memory_saver will support GMS mode
    from torch_memory_saver import torch_memory_saver
"""

import logging
import os

logger = logging.getLogger(__name__)

_patched = False


def _is_gms_mode() -> bool:
    """Check if GMS mode is enabled via environment variable."""
    return (
        os.environ.get("GMS_SGLANG_AUTO_REGISTER") == "1"
        or os.environ.get("GMS_VLLM_AUTO_REGISTER") == "1"
    )


def patch_torch_memory_saver():
    """Patch torch_memory_saver to support GMS mode.

    This function is idempotent - calling it multiple times has no effect.
    """
    global _patched
    if _patched:
        return

    try:
        import torch_memory_saver.entrypoint as entrypoint_module
    except ImportError:
        logger.debug("[GMS Patch] torch_memory_saver not installed, skipping patch")
        return

    # Store reference to original method
    original_ensure_initialized = entrypoint_module.TorchMemorySaver._ensure_initialized

    def patched_ensure_initialized(self):
        """Patched _ensure_initialized that supports GMS mode."""
        # Check if already initialized
        if self._impl is not None:
            return

        # Check if GMS mode is enabled
        hook_mode = self._impl_ctor_kwargs.get("hook_mode")
        if hook_mode == "gms" or (hook_mode is None and _is_gms_mode()):
            # Use our GMS implementation
            from dynamo.sglang.gms_adapters.torch_memory_saver_gms import (
                GMSMemorySaverImpl,
            )

            # Set _impl directly since all TorchMemorySaver methods access self._impl
            self._impl = GMSMemorySaverImpl()
            logger.info("[TorchMemorySaver] Using GMS mode for VA-stable sleep/wake")
            del self._impl_ctor_kwargs
        else:
            # Fall back to original implementation
            original_ensure_initialized(self)

    # Patch the method
    entrypoint_module.TorchMemorySaver._ensure_initialized = patched_ensure_initialized

    _patched = True
    logger.debug("[GMS Patch] Successfully patched torch_memory_saver for GMS mode")


# Auto-patch on import
patch_torch_memory_saver()
