# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dynamo v2 TensorRT-LLM Connector package.

This package provides v2 connector implementations for TensorRT-LLM that use
the KVBM v2::integrations::connector module + Nova crate for distributed communication.
"""

from kvbm.v2.common import NovaPeerMetadata

from .leader import DynamoKVBMConnectorLeader
from .worker import DynamoKVBMConnectorWorker

__all__ = [
    "DynamoKVBMConnectorLeader",
    "DynamoKVBMConnectorWorker",
    "NovaPeerMetadata",
]
