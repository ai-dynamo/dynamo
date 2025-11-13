# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dynamo Common Utils Module

This module contains shared utility functions used across multiple
Dynamo backends and components.

Submodules:
    - paths: Workspace directory detection and path utilities
    - prometheus: Prometheus metrics collection and logging utilities
    - gpu: GPU device information and metadata utilities
"""

from dynamo.common.utils import gpu, paths, prometheus

__all__ = ["gpu", "paths", "prometheus"]
