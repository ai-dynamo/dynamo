# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Rust-based TensorRT-LLM integration loader.

Uses objects from _vllm_integration module. Type stubs in kvbm/_core.pyi.

KvConnectorWorker (PyTrtllmKvConnectorWorker) signature:
    (py_drt, trtllm_rank, rank=None, world_size=None, nccl_comm_ptr=None)

The rank, world_size, and nccl_comm_ptr parameters enable NCCL replicated mode
for MLA (Multi-head Latent Attention) support with broadcast-based KV cache transfers.
"""

try:
    # TODO: use TRTLLM own integration module
    from kvbm._core import _vllm_integration

    # Runtime - dynamically loaded classes from Rust extension
    KvbmRequest = getattr(_vllm_integration, "KvbmRequest")
    KvbmBlockList = getattr(_vllm_integration, "KvbmBlockList")
    BlockState = getattr(_vllm_integration, "BlockState")
    BlockStates = getattr(_vllm_integration, "BlockStates")
    SlotUpdate = getattr(_vllm_integration, "SlotUpdate")

    # TRT-LLM connector classes with NCCL replicated mode support
    # KvConnectorWorker: optional rank, world_size, nccl_comm_ptr for MLA support
    KvConnectorWorker = getattr(_vllm_integration, "PyTrtllmKvConnectorWorker")
    KvConnectorLeader = getattr(_vllm_integration, "PyTrtllmKvConnectorLeader")
    SchedulerOutput = getattr(_vllm_integration, "SchedulerOutput")

except ImportError:
    print(
        "Failed to import Dynamo KVBM. "
        "TensorRT-LLM integration will not be available."
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
