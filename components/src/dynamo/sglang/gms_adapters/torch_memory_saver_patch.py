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
        # Check if already initialized (handle both standard and GMS impl)
        if self._impl is not None or getattr(self, "_gms_impl", None) is not None:
            return

        # Get hook mode from kwargs or auto-detect
        hook_mode = self._impl_ctor_kwargs.get("hook_mode")
        if hook_mode is None:
            # Check for GMS mode first (our extension)
            if _is_gms_mode():
                hook_mode = "gms"
            else:
                # Fall back to original detection logic
                hook_mode = entrypoint_module._detect_hook_mode()
            self._impl_ctor_kwargs["hook_mode"] = hook_mode

        if hook_mode == "gms":
            # Use our GMS implementation
            from dynamo.sglang.gms_adapters.torch_memory_saver_gms import (
                GMSMemorySaverImpl,
            )

            # Ensure _gms_impl attribute exists
            self._gms_impl = GMSMemorySaverImpl()
            logger.info("[TorchMemorySaver] Using GMS mode for VA-stable sleep/wake")
            del self._impl_ctor_kwargs
        else:
            # Fall back to original implementation
            original_ensure_initialized(self)

    # Patch the method
    entrypoint_module.TorchMemorySaver._ensure_initialized = patched_ensure_initialized

    # Also patch _get_impl to check for _gms_impl
    original_get_impl = entrypoint_module.TorchMemorySaver._get_impl

    def patched_get_impl(self):
        """Patched _get_impl that checks for GMS impl."""
        gms_impl = getattr(self, "_gms_impl", None)
        if gms_impl is not None:
            return gms_impl
        return original_get_impl(self)

    entrypoint_module.TorchMemorySaver._get_impl = patched_get_impl

    _patched = True
    logger.debug("[GMS Patch] Successfully patched torch_memory_saver for GMS mode")


# Auto-patch on import
patch_torch_memory_saver()
