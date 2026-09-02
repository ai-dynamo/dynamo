# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Register Dynamo KVBM as an SGLang radix-cache backend."""

from __future__ import annotations

import importlib

from sglang.srt.mem_cache.registry import (
    TreeCacheBuildContext,
    register_radix_cache_backend,
)

# isort: split

from kvbm.sglang_integration.provider import (
    HostMemoryProvider,
    HostRegionRequest,
    register_host_memory_provider,
)


def build_dynamo_kvbm_cache(ctx: TreeCacheBuildContext):
    """Lazily load the native data plane after registering the backend name."""
    factory = importlib.import_module("kvbm.sglang_integration.factory")
    return factory.build_dynamo_kvbm_cache(ctx)


register_radix_cache_backend("dynamo_kvbm", build_dynamo_kvbm_cache)

__all__ = [
    "HostMemoryProvider",
    "HostRegionRequest",
    "build_dynamo_kvbm_cache",
    "register_host_memory_provider",
]
