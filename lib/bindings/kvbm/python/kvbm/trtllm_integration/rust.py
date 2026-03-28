# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Loader for the Rust-based TensorRT-LLM integration objects.
"""

try:
    from kvbm._core import BlockManager
    from kvbm._core import _trtllm_integration

    # Runtime - dynamically loaded classes from Rust extension
    KvbmRequest = getattr(_trtllm_integration, "KvbmRequest")
    KvbmBlockList = getattr(_trtllm_integration, "KvbmBlockList")
    BlockState = getattr(_trtllm_integration, "BlockState")
    BlockStates = getattr(_trtllm_integration, "BlockStates")
    SlotUpdate = getattr(_trtllm_integration, "SlotUpdate")

    KvConnectorWorker = getattr(_trtllm_integration, "PyTrtllmKvConnectorWorker")
    KvConnectorLeader = getattr(_trtllm_integration, "PyTrtllmKvConnectorLeader")
    SchedulerOutput = getattr(_trtllm_integration, "SchedulerOutput")

except ImportError:
    print(
        "Failed to import Dynamo KVBM. TensorRT-LLM integration will not be available."
    )
    BlockManager = None
    KvbmRequest = None
    KvbmBlockList = None
    BlockState = None
    BlockStates = None
    SlotUpdate = None
    KvConnectorWorker = None
    KvConnectorLeader = None
    SchedulerOutput = None

__all__ = [
    "BlockManager",
    "KvbmRequest",
    "KvbmBlockList",
    "BlockState",
    "BlockStates",
    "SlotUpdate",
    "KvConnectorWorker",
    "KvConnectorLeader",
    "SchedulerOutput",
]
