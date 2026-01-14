# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Loader for the Rust-based vLLM integration objects.
"""

try:
    from kvbm._core import _vllm_integration

    # Runtime - dynamically loaded classes from Rust extension
    KvbmRequest = getattr(_vllm_integration, "KvbmRequest")

    KvConnectorWorker = getattr(_vllm_integration, "PyKvConnectorWorker")
    KvConnectorLeader = getattr(_vllm_integration, "PyKvConnectorLeader")
    SchedulerOutput = getattr(_vllm_integration, "SchedulerOutput")

    from kvbm import BlockManager

except ImportError:
    print("Failed to import Dynamo KVBM. vLLM integration will not be available.")
    KvbmRequest = None
    BlockManager = None
    KvConnectorWorker = None
    KvConnectorLeader = None
    SchedulerOutput = None

__all__ = [
    "KvbmRequest",
    "BlockManager",
    "KvConnectorWorker",
    "KvConnectorLeader",
    "SchedulerOutput",
]
