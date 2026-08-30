# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common patches shared across GMS integrations."""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch
from gpu_memory_service.client.torch.allocator import get_gms_client_memory_managers

logger = logging.getLogger(__name__)

_empty_cache_patched = False


def patch_empty_cache() -> None:
    """Patch the empty-cache entry points to prevent faults with VMM allocations.

    When weights are allocated through our VMM-based pluggable allocator, emptying
    the cache faults because the native caching allocator tries to release blocks
    that were allocated through VMM APIs.

    Both ``torch.cuda.empty_cache`` and ``torch.accelerator.empty_cache`` must be
    patched: the latter dispatches straight to ``torch._C._accelerator_emptyCache``
    rather than delegating to the former, so patching only the CUDA entry point
    leaves a live path into the native allocator. vLLM calls the accelerator entry
    point from inference-time paths (for example the sparse-MLA prefill workspace
    resize in ``vllm/v1/worker/workspace.py``), where the fault poisons the CUDA
    context and takes down the worker.

    This patch is idempotent - calling it multiple times has no effect.
    """
    global _empty_cache_patched

    if _empty_cache_patched:
        return

    def guard(original: Callable[[], None]) -> Callable[[], None]:
        def safe_empty_cache() -> None:
            managers = get_gms_client_memory_managers()
            # Allow empty_cache when all managers are unmapped (sleep/checkpoint)
            # or when there are no active VMM mappings with live handles.
            has_live_mappings = any(
                any(m.handle != 0 for m in manager.mappings.values())
                for manager in managers
            )
            if has_live_mappings:
                logger.debug(
                    "[GMS] Skipping empty_cache() - live VMM mappings active",
                )
                return
            original()

        return safe_empty_cache

    torch.cuda.empty_cache = guard(torch.cuda.empty_cache)
    # torch.accelerator exists from torch 2.5; guard so older builds still load.
    accelerator = getattr(torch, "accelerator", None)
    if accelerator is not None:
        accelerator.empty_cache = guard(accelerator.empty_cache)

    _empty_cache_patched = True
    logger.info("[GMS] Patched torch.cuda/torch.accelerator empty_cache")
