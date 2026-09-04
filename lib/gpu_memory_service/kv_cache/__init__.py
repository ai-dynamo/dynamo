# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KV-cache management that outlives a single engine.

GMS makes the KV bytes survive a failover; this makes them findable. Engine-side
glue lives under ``integrations/<engine>/kv_cache``.
"""

from gpu_memory_service.kv_cache.backends.mmap import MmapBlockIndexStore
from gpu_memory_service.kv_cache.interface import BlockIndexStore, Refusal, store_path

__all__ = ["BlockIndexStore", "MmapBlockIndexStore", "Refusal", "store_path"]
