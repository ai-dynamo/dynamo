# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common patches shared across GMS integrations."""

from __future__ import annotations

import logging

from gpu_memory_service.client.torch.allocator import get_gms_client_memory_managers
from gpu_memory_service.integrations.common.utils import torch_device

logger = logging.getLogger(__name__)

_empty_cache_patched = False


def patch_empty_cache() -> None:
    """Patch torch.{cuda,xpu}.empty_cache to prevent segfaults with VMM allocations.

    When weights are allocated through our VMM-based pluggable allocator, calling
    empty_cache() causes segfaults because the native caching allocator
    tries to release blocks that were allocated through VMM APIs.

    This patch is idempotent - calling it multiple times has no effect.
    """
    global _empty_cache_patched

    if _empty_cache_patched:
        return

    dev_mod = torch_device()
    _original_empty_cache = dev_mod.empty_cache

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
        _original_empty_cache()

    dev_mod.empty_cache = safe_empty_cache
    _empty_cache_patched = True
    logger.info("[GMS] Patched %s.empty_cache", dev_mod.__name__)
