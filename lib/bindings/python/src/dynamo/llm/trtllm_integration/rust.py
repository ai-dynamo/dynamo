# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Loader for the Rust-based TensorRT-LLM integration objects, using objects from _vllm_integration for now
"""

try:
    # TODO: use TRTLLM own integration module
    from dynamo._core import _vllm_integration

    # Runtime - dynamically loaded classes from Rust extension
    KvbmRequest = getattr(_vllm_integration, "KvbmRequest")
    KvbmBlockList = getattr(_vllm_integration, "KvbmBlockList")
    BlockState = getattr(_vllm_integration, "BlockState")
    BlockStates = getattr(_vllm_integration, "BlockStates")
    SlotUpdate = getattr(_vllm_integration, "SlotUpdate")

    KvConnectorWorker = getattr(_vllm_integration, "PyTrtllmKvConnectorWorker")
    KvConnectorLeader = getattr(_vllm_integration, "PyTrtllmKvConnectorLeader")
    SchedulerOutput = getattr(_vllm_integration, "SchedulerOutput")

except ImportError:
    print(
        "Failed to import Dynamo KVBM. TensorRT-LLM integration will not be available."
    )
    KvbmRequest = None
    KvbmBlockList = None
    BlockState = None
    BlockStates = None
    SlotUpdate = None
    KvConnectorWorker = None
    KvConnectorLeader = None
    SchedulerOutput = None

__all__ = [
    "KvbmRequest",
    "KvbmBlockList",
    "BlockState",
    "BlockStates",
    "SlotUpdate",
    "KvConnectorWorker",
    "KvConnectorLeader",
    "SchedulerOutput",
]
