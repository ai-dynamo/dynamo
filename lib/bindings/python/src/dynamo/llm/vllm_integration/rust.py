# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Loader for the Rust-based vLLM integration objects.
"""

from typing import TYPE_CHECKING

from dynamo._core import _vllm_integration

if TYPE_CHECKING:
    # Type stubs for static analysis - these are treated as proper types
    class KvbmCacheManager:
        ...

    class KvbmRequest:
        ...

    class KvbmBlockList:
        ...

    class BlockState:
        ...

    class BlockStates:
        ...

    class SlotUpdate:
        ...

else:
    # Runtime - dynamically loaded classes from Rust extension
    KvbmCacheManager = getattr(_vllm_integration, "KvbmCacheManager")
    KvbmRequest = getattr(_vllm_integration, "KvbmRequest")
    KvbmBlockList = getattr(_vllm_integration, "KvbmBlockList")
    BlockState = getattr(_vllm_integration, "BlockState")
    BlockStates = getattr(_vllm_integration, "BlockStates")
    SlotUpdate = getattr(_vllm_integration, "SlotUpdate")


__all__ = [
    "KvbmCacheManager",
    "KvbmRequest",
    "KvbmBlockList",
    "BlockState",
    "BlockStates",
    "SlotUpdate",
]
