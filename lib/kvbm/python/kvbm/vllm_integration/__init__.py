# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Import connector classes to make them available at the expected paths for vLLM
from .connector.dynamo_connector import DynamoConnector, DynamoConnectorMetadata
from .connector.pd_connector import PdConnector, PdConnectorMetadata

# Create module-level alias for backward compatibility
dynamo_connector = DynamoConnector
pd_connector = PdConnector

__all__ = [
    "DynamoConnector",
    "DynamoConnectorMetadata",
    "dynamo_connector",
    "PdConnector",
    "PdConnectorMetadata",
]
