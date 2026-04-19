# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common backend interface for Dynamo LLM workers.

This module provides a unified interface for implementing LLM backend workers
(e.g., different LLM inference frameworks).

Key components:
- Backend: Abstract base class for backend workers with common lifecycle
- Handler: Abstract base class for request handlers
- DynamoRuntimeConfig: Base configuration class with common fields
- DynamoRuntimeArgGroup: Common CLI argument definitions
"""

from dynamo.backend.args import (
    DynamoBackendArgGroup,
    DynamoBackendConfig,
    DynamoRuntimeArgGroup,
    DynamoRuntimeConfig,
)
from dynamo.backend.base import DYNAMO_COMPONENT_REGISTRY, Backend
from dynamo.backend.handler import Handler

__all__ = [
    # Shared registry
    "DYNAMO_COMPONENT_REGISTRY",
    # Base classes
    "Backend",
    "Handler",
    # Configuration
    "DynamoBackendConfig",
    "DynamoBackendArgGroup",
    "DynamoRuntimeConfig",
    "DynamoRuntimeArgGroup",
]
