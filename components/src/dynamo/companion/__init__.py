# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dynamo companion server system for CUDA IPC weight sharing.

This module provides a companion server that loads model weights once and shares
them with vLLM workers via CUDA IPC using Dynamo's DistributedRuntime.
"""

from .handler import CompanionHandler, SuccessResponse, ErrorResponse, LoadModelResponse
from .client import CompanionClient
from .vllm import VllmModelLoader

__all__ = [
    "CompanionHandler",
    "CompanionClient",
    "VllmModelLoader",
    "SuccessResponse",
    "ErrorResponse",
    "LoadModelResponse",
]
