# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common backend interface for Dynamo LLM workers.

This module provides a unified interface for implementing LLM backend workers
(e.g., different LLM inference frameworks).

Key components:
- BackendConfig: Base configuration class with common fields
- BaseBackend: Abstract base class for backend workers with common lifecycle
- BaseHandler: Abstract base class for request handlers
- Argument groups: Common CLI argument definitions
"""

from dynamo.common.backend.args import (
    BackendCommonArgGroup,
    BackendCommonConfig,
    WorkerModeArgGroup,
    WorkerModeConfig,
)
from dynamo.common.backend.base import (
    DYNAMO_COMPONENT_REGISTRY,
    BackendConfig,
    BaseBackend,
)
from dynamo.common.backend.handler import BaseHandler

__all__ = [
    # Shared registry
    "DYNAMO_COMPONENT_REGISTRY",
    # Base classes
    "BackendConfig",
    "BaseBackend",
    "BaseHandler",
    # Configuration
    "BackendCommonConfig",
    "BackendCommonArgGroup",
    "WorkerModeConfig",
    "WorkerModeArgGroup",
]
