# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Loader for the Rust-based vLLM integration objects.
"""

try:
    from dynamo._core import _vllm_integration

    # Runtime - dynamically loaded classes from Rust extension
    KvbmCacheManager = getattr(_vllm_integration, "KvbmCacheManager")
    KvbmRequest = getattr(_vllm_integration, "KvbmRequest")
    KvbmBlockList = getattr(_vllm_integration, "KvbmBlockList")
    BlockState = getattr(_vllm_integration, "BlockState")
    BlockStates = getattr(_vllm_integration, "BlockStates")
    SlotUpdate = getattr(_vllm_integration, "SlotUpdate")

    from dynamo.llm import BlockManager

except ImportError:
    print("Failed to import Dynamo KVBM. vLLM integration will not be available.")
    KvbmCacheManager = None
    KvbmRequest = None
    KvbmBlockList = None
    BlockState = None
    BlockStates = None
    SlotUpdate = None
    BlockManager = None

__all__ = [
    "KvbmCacheManager",
    "KvbmRequest",
    "KvbmBlockList",
    "BlockState",
    "BlockStates",
    "SlotUpdate",
    "BlockManager",
]
