# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM integration for GPU Memory Service.

This package provides GPU Memory Service integration for vLLM, enabling:
- VA-stable weight sharing across processes
- Sleep/wake memory management
- Efficient model weight loading via memory service

Usage:
    Use the GMSWorker class as a custom worker:

        --worker-cls=gpu_memory_service.client.vllm_integration.worker:GMSWorker

    This automatically:
    - Registers the 'gpu_memory_service' model loader
    - Applies necessary utility patches
    - Provides VA-stable sleep/wake functionality

Module structure:
    - config.py: Configuration constants and environment handling
    - memory_ops.py: Sleep/wake implementation and GMS utilities
    - model_loader.py: Model loading implementation
    - patches.py: Utility patching (empty_cache, MemorySnapshot)
    - worker.py: GMSWorker subclass with proper overrides
"""

from __future__ import annotations

import logging

from gpu_memory_service.client.vllm_integration.model_loader import (
    get_imported_weights_bytes,
    register_gms_loader,
)
from gpu_memory_service.client.vllm_integration.patches import (
    patch_empty_cache,
    patch_memory_snapshot,
)

logger = logging.getLogger(__name__)

__all__ = [
    # Model loader
    "register_gms_loader",
    "get_imported_weights_bytes",
    # Utility patches
    "patch_empty_cache",
    "patch_memory_snapshot",
    # Worker class (imported lazily to avoid circular imports)
    "GMSWorker",
]


def __getattr__(name):
    """Lazy import for GMSWorker to avoid circular imports."""
    if name == "GMSWorker":
        from gpu_memory_service.client.vllm_integration.worker import GMSWorker

        return GMSWorker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
