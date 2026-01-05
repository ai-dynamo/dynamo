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
        os.environ.get("GPU_MEMORY_SERVICE_SGLANG_AUTO_REGISTER") == "1"
        or os.environ.get("GPU_MEMORY_SERVICE_VLLM_AUTO_REGISTER") == "1"
    )


def _get_gpu_memory_service_mode() -> str:
    """Get the GPU Memory Service load mode from environment variable.

    Returns:
        "write" (cold start from disk) or "read" (import only).
        Defaults to "write" if not specified.
    """
    mode = os.environ.get("GPU_MEMORY_SERVICE_LOAD_MODE", "write").strip().lower()
    if mode not in ("write", "read"):
        logger.warning(
            f"[GPU Memory Service Patch] Invalid GPU_MEMORY_SERVICE_LOAD_MODE={mode!r}, "
            "expected 'write' or 'read'. Defaulting to 'write'."
        )
        return "write"
    return mode


def _resolve_socket_path(device_index: int) -> str:
    """Resolve socket path from environment or use default."""
    socket_path = os.environ.get("GPU_MEMORY_SERVICE_SOCKET_PATH")
    if not socket_path:
        socket_path = f"/tmp/gpu_memory_service_{device_index}.sock"
    elif "{device}" in socket_path:
        socket_path = socket_path.format(device=device_index)
    elif "{local_rank}" in socket_path:
        socket_path = socket_path.format(local_rank=device_index)
    return socket_path


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
            logger.debug("[TorchMemorySaver Patch] Already initialized, skipping")
            return

        # Check if GPU Memory Service mode is enabled
        hook_mode = self._impl_ctor_kwargs.get("hook_mode")
        logger.info(f"[TorchMemorySaver Patch] Initializing with hook_mode={hook_mode}")
        logger.info(
            f"[TorchMemorySaver Patch] _is_gpu_memory_service_mode()={_is_gpu_memory_service_mode()}"
        )

        if hook_mode == "gpu_memory_service" or (
            hook_mode is None and _is_gpu_memory_service_mode()
        ):
            # Use our GPU Memory Service implementation
            import torch
            from torch_memory_saver.entrypoint import _TorchMemorySaverImpl

            from dynamo.sglang.gpu_memory_service_adapters.torch_memory_saver_impl import (
                GPUMemoryServiceMemorySaverImpl,
                set_gpu_memory_service_impl,
            )

            # Get device from torch.cuda.current_device() (already set by SGLang)
            device_index = torch.cuda.current_device()

            # Resolve socket path from env or default
            socket_path = _resolve_socket_path(device_index)

            # Get explicit mode from environment (defaults to "write")
            gms_mode = _get_gpu_memory_service_mode()

            # Create underlying torch impl for non-weights tags (KV cache etc.)
            # Use "torch" hook mode which uses PyTorch's CUDAPluggableAllocator
            torch_impl = _TorchMemorySaverImpl(hook_mode="torch")

            # Create GPU Memory Service impl (owns allocator, explicit mode)
            gpu_impl = GPUMemoryServiceMemorySaverImpl(
                torch_impl=torch_impl,
                socket_path=socket_path,
                device_index=device_index,
                mode=gms_mode,
            )

            # Store reference for model loader to access
            set_gpu_memory_service_impl(gpu_impl)

            # Set _impl directly since all TorchMemorySaver methods access self._impl
            self._impl = gpu_impl
            logger.info(
                "[TorchMemorySaver] Using GPU Memory Service mode "
                "(device=%d, socket=%s, allocator_mode=%s)",
                device_index,
                socket_path,
                gpu_impl.get_mode(),
            )
            del self._impl_ctor_kwargs
        else:
            # Fall back to original implementation
            logger.info("[TorchMemorySaver Patch] Using default hook mode")
            original_ensure_initialized(self)

    # Patch the method
    entrypoint_module.TorchMemorySaver._ensure_initialized = patched_ensure_initialized

    _patched = True
    logger.debug(
        "[GPU Memory Service Patch] Successfully patched torch_memory_saver for GPU Memory Service mode"
    )


# Auto-patch on import
patch_torch_memory_saver()
