# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Loader for the Rust-based TensorRT-LLM integration objects, using objects from _vllm_integration for now
"""

try:
    # TODO: use TRTLLM own integration module
    from kvbm._core import _vllm_integration

    # Runtime - dynamically loaded classes from Rust extension
    KvbmRequest = getattr(_vllm_integration, "KvbmRequest")

    KvConnectorWorker = getattr(_vllm_integration, "PyTrtllmKvConnectorWorker")
    KvConnectorLeader = getattr(_vllm_integration, "PyTrtllmKvConnectorLeader")
    SchedulerOutput = getattr(_vllm_integration, "SchedulerOutput")

except ImportError:
    print(
        "Failed to import Dynamo KVBM. TensorRT-LLM integration will not be available."
    )
    KvbmRequest = None
    KvConnectorWorker = None
    KvConnectorLeader = None
    SchedulerOutput = None

__all__ = [
    "KvbmRequest",
    "KvConnectorWorker",
    "KvConnectorLeader",
    "SchedulerOutput",
]
