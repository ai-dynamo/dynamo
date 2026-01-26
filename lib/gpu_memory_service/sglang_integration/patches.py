# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utility patches for GPU Memory Service SGLang integration.

This module contains patches applied when the GMS SGLang integration is imported:
- torch.cuda.empty_cache patch (prevents segfaults with VMM allocations)
- torch_memory_saver patch (routes to GMS implementation)
"""

from __future__ import annotations

import logging
import os

import torch
from gpu_memory_service import get_gms_client_memory_manager

logger = logging.getLogger(__name__)


_empty_cache_patched = False
_torch_memory_saver_patched = False


def patch_empty_cache() -> None:
    """Patch torch.cuda.empty_cache to prevent segfaults with VMM allocations.

    Must be called at module import time before any empty_cache calls.
    """
    global _empty_cache_patched

    if _empty_cache_patched:
        return

    _original_empty_cache = torch.cuda.empty_cache

    def safe_empty_cache() -> None:
        """Safe replacement for torch.cuda.empty_cache that skips when VMM allocations exist.

        When weights are allocated through our VMM-based pluggable allocator, calling
        torch.cuda.empty_cache() causes segfaults because the native caching allocator
        tries to release blocks that were allocated through VMM APIs.
        """
        manager = get_gms_client_memory_manager()
        if manager is not None and len(manager.mappings) > 0:
            logger.debug(
                "[GMS] Skipping torch.cuda.empty_cache() - %d VMM allocations active",
                len(manager.mappings),
            )
            return

        _original_empty_cache()

    torch.cuda.empty_cache = safe_empty_cache
    _empty_cache_patched = True
    logger.info("[GMS Patch] Patched torch.cuda.empty_cache")


def _is_gpu_memory_service_mode() -> bool:
    """Check if GPU Memory Service mode is enabled via environment variable."""
    return (
        os.environ.get("GPU_MEMORY_SERVICE_SGLANG_AUTO_REGISTER") == "1"
        or os.environ.get("GPU_MEMORY_SERVICE_VLLM_AUTO_REGISTER") == "1"
    )


def _resolve_socket_path(device_index: int) -> str:
    """Resolve socket path from environment or use default."""
    from gpu_memory_service.common.utils import get_socket_path

    return get_socket_path(device_index)


def patch_torch_memory_saver() -> None:
    """Patch torch_memory_saver to support GPU Memory Service mode.

    This function is idempotent - calling it multiple times has no effect.
    """
    global _torch_memory_saver_patched
    if _torch_memory_saver_patched:
        return

    try:
        import torch_memory_saver.entrypoint as entrypoint_module
    except ImportError:
        logger.debug(
            "[GMS Patch] torch_memory_saver not installed, skipping patch"
        )
        return

    # Store reference to original method
    original_ensure_initialized = entrypoint_module.TorchMemorySaver._ensure_initialized

    def patched_ensure_initialized(self):
        """Patched _ensure_initialized that supports GPU Memory Service mode."""
        # Check if already initialized
        if self._impl is not None:
            logger.debug("[GMS Patch] TorchMemorySaver already initialized, skipping")
            return

        # Check if GPU Memory Service mode is enabled
        hook_mode = self._impl_ctor_kwargs.get("hook_mode")
        logger.info(f"[GMS Patch] TorchMemorySaver initializing with hook_mode={hook_mode}")

        if hook_mode == "gpu_memory_service" or (
            hook_mode is None and _is_gpu_memory_service_mode()
        ):
            # Use our GPU Memory Service implementation
            from torch_memory_saver.entrypoint import _TorchMemorySaverImpl

            from gpu_memory_service.sglang_integration.torch_memory_saver_impl import (
                GPUMemoryServiceMemorySaverImpl,
                set_gpu_memory_service_impl,
            )

            # Get device from torch.cuda.current_device() (already set by SGLang)
            device_index = torch.cuda.current_device()

            # Resolve socket path from env or default
            socket_path = _resolve_socket_path(device_index)

            # Create underlying torch impl for non-weights tags (KV cache etc.)
            # Use "torch" hook mode which uses PyTorch's CUDAPluggableAllocator
            torch_impl = _TorchMemorySaverImpl(hook_mode="torch")

            # Create GPU Memory Service impl (owns allocator, uses auto mode)
            # Auto mode: first process gets RW and loads from disk, others get RO and import
            gpu_impl = GPUMemoryServiceMemorySaverImpl(
                torch_impl=torch_impl,
                socket_path=socket_path,
                device_index=device_index,
            )

            # Store reference for model loader to access
            set_gpu_memory_service_impl(gpu_impl)

            # Set _impl directly since all TorchMemorySaver methods access self._impl
            self._impl = gpu_impl
            logger.info(
                "[GMS] Using GPU Memory Service mode "
                "(device=%d, socket=%s, allocator_mode=%s)",
                device_index,
                socket_path,
                gpu_impl.get_mode(),
            )
            del self._impl_ctor_kwargs
        else:
            # Fall back to original implementation
            logger.info("[GMS Patch] Using default torch_memory_saver hook mode")
            original_ensure_initialized(self)

    # Patch the method
    entrypoint_module.TorchMemorySaver._ensure_initialized = patched_ensure_initialized

    _torch_memory_saver_patched = True
    logger.debug(
        "[GMS Patch] Successfully patched torch_memory_saver for GPU Memory Service mode"
    )
