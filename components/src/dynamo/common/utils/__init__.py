# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dynamo Common Utils Module

This module contains shared utility functions used across multiple
Dynamo backends and components.

Submodules:
    - endpoint_types: Endpoint type parsing utilities
    - paths: Workspace directory detection and path utilities
    - prometheus: Prometheus metrics collection and logging utilities
"""

from dynamo.common.utils import endpoint_types, paths, prometheus

__all__ = ["endpoint_types", "paths", "prometheus"]
