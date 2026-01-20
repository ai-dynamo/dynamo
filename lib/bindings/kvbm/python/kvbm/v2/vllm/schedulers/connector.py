# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Backward compatibility - re-export connector classes from canonical location.

The canonical location is: kvbm.v2.vllm.connector

This module exists to maintain backward compatibility with existing configs:
    "kv_connector_module_path": "kvbm.v2.vllm.schedulers.connector"
"""

from kvbm.v2.vllm.connector import (
    DynamoConnector,
    DynamoSchedulerConnectorMetadata,
    NovaPeerMetadata,
    SchedulerConnectorLeader,
    SchedulerConnectorWorker,
)

__all__ = [
    "DynamoConnector",
    "DynamoSchedulerConnectorMetadata",
    "SchedulerConnectorLeader",
    "SchedulerConnectorWorker",
    "NovaPeerMetadata",
]
