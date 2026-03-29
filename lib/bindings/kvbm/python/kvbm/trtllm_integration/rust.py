# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Loader for the Rust-based TensorRT-LLM integration objects.
"""

try:
    from kvbm._core import BlockManager
    from kvbm._core import _trtllm_integration

    # Runtime - dynamically loaded classes from Rust extension
    KvbmRequest = _trtllm_integration.KvbmRequest
    KvbmBlockList = _trtllm_integration.KvbmBlockList
    BlockState = _trtllm_integration.BlockState
    BlockStates = _trtllm_integration.BlockStates
    SlotUpdate = _trtllm_integration.SlotUpdate
    TrtllmStateManager = getattr(_trtllm_integration, "TrtllmStateManager", None)

    KvConnectorWorker = _trtllm_integration.PyTrtllmKvConnectorWorker
    KvConnectorLeader = _trtllm_integration.PyTrtllmKvConnectorLeader
    SchedulerOutput = _trtllm_integration.SchedulerOutput
    create_primary_pool = getattr(_trtllm_integration, "create_primary_pool", None)

except ImportError:
    BlockManager = None
    KvbmRequest = None
    KvbmBlockList = None
    BlockState = None
    BlockStates = None
    SlotUpdate = None
    TrtllmStateManager = None
    KvConnectorWorker = None
    KvConnectorLeader = None
    SchedulerOutput = None
    create_primary_pool = None

__all__ = [
    "BlockManager",
    "KvbmRequest",
    "KvbmBlockList",
    "BlockState",
    "BlockStates",
    "SlotUpdate",
    "TrtllmStateManager",
    "KvConnectorWorker",
    "KvConnectorLeader",
    "SchedulerOutput",
    "create_primary_pool",
]
