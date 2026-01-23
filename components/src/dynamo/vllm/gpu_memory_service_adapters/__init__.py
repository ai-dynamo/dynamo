"""vLLM adapter for GPU Memory Service (Allocation Server + embedded metadata store).

This package provides GPU Memory Service integration for vLLM, enabling:
- VA-stable weight sharing across processes
- Sleep/wake memory management
- Efficient model weight loading via memory service

Usage:
    Set GPU_MEMORY_SERVICE_VLLM_AUTO_REGISTER=1 to enable auto-registration.
    The plugin will be initialized when vLLM loads.

Module structure:
    - config.py: Configuration constants and environment handling
    - utils.py: Shared utilities (Worker class detection, etc.)
    - memory_ops.py: Sleep/wake implementation logic
    - model_loader.py: Model loading implementation
    - patches.py: All monkey-patching logic
"""

from __future__ import annotations

import logging

from dynamo.vllm.gpu_memory_service_adapters.config import GMS_ENABLED
from dynamo.vllm.gpu_memory_service_adapters.model_loader import (
    get_imported_weights_bytes,
    register_gpu_memory_service_loader,
)
from dynamo.vllm.gpu_memory_service_adapters.patches import (
    apply_all_patches,
    patch_sleep_wake,
    patch_worker_for_gms,
)

logger = logging.getLogger(__name__)

__all__ = [
    "register_gpu_memory_service_loader",
    "apply_all_patches",
    "patch_sleep_wake",
    "patch_worker_for_gms",
    "get_imported_weights_bytes",
    "vllm_plugin_init",
]


def vllm_plugin_init():
    """vLLM plugin entry point for GPU Memory Service.

    This function is called by vLLM's plugin system in all processes (main process,
    engine core, and worker processes). It registers the GPU Memory Service loader
    and applies necessary patches if the GPU_MEMORY_SERVICE_VLLM_AUTO_REGISTER
    environment variable is set.
    """
    if GMS_ENABLED:
        register_gpu_memory_service_loader()
        apply_all_patches()
        logger.info("[GMS] vLLM plugin initialized")


# Auto-register when environment variable is set.
# This handles vLLM's multiprocessing with spawn mode, where child
# processes start fresh and don't inherit the parent's loader registration.
if GMS_ENABLED:
    vllm_plugin_init()
