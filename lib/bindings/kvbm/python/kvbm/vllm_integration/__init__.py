# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Import connector classes to make them available at the expected paths for vLLM
from .connector.dynamo_connector import DynamoConnector, DynamoConnectorMetadata

# Create module-level alias for backward compatibility
dynamo_connector = DynamoConnector

__all__ = [
    "DynamoConnector",
    "DynamoConnectorMetadata",
    "dynamo_connector",
]

try:
    from .connector.pd_connector import PdConnector, PdConnectorMetadata

    pd_connector = PdConnector
    __all__ += ["PdConnector", "PdConnectorMetadata", "pd_connector"]
except ImportError:
    pass
