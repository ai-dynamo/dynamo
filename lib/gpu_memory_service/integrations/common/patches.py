# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common patches shared across GMS integrations."""

from __future__ import annotations

import logging

import torch
from gpu_memory_service import get_gms_client_memory_manager

logger = logging.getLogger(__name__)

_empty_cache_patched = False


def patch_empty_cache() -> None:
    """Patch torch.cuda.empty_cache to prevent segfaults with VMM allocations.

    When weights are allocated through our VMM-based pluggable allocator, calling
    torch.cuda.empty_cache() causes segfaults because the native caching allocator
    tries to release blocks that were allocated through VMM APIs.

    This patch is idempotent - calling it multiple times has no effect.
    """
    global _empty_cache_patched

    if _empty_cache_patched:
        return

    _original_empty_cache = torch.cuda.empty_cache

    def safe_empty_cache() -> None:
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
    logger.info("[GMS] Patched torch.cuda.empty_cache")
