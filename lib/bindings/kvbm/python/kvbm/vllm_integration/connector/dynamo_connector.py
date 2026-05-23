# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility import path for the v2 Dynamo vLLM connector."""

from kvbm.v2.vllm.schedulers.connector import (
    DynamoConnector,
    DynamoSchedulerConnectorMetadata,
)

DynamoConnectorMetadata = DynamoSchedulerConnectorMetadata

__all__ = [
    "DynamoConnector",
    "DynamoConnectorMetadata",
    "DynamoSchedulerConnectorMetadata",
]
