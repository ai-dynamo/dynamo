# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common patches shared across GMS integrations."""

from __future__ import annotations

import logging

import torch
from gpu_memory_service.client.torch.allocator import get_gms_client_memory_managers

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
        active_mapping_count = sum(
            1
            for manager in get_gms_client_memory_managers()
            for mapping in manager.mappings.values()
            if mapping.handle != 0
        )
        if active_mapping_count:
            logger.warning(
                "[GMS] Skipping torch.cuda.empty_cache() - %d active GMS mappings",
                active_mapping_count,
            )
            return
        _original_empty_cache()

    torch.cuda.empty_cache = safe_empty_cache
    _empty_cache_patched = True
    logger.info("[GMS] Patched torch.cuda.empty_cache")
