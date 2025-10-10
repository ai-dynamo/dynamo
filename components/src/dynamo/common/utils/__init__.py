# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dynamo Common Utils Module

This module contains shared utility functions used across multiple
Dynamo backends and components.

Submodules:
    - env: Environment variable parsing utilities
    - prometheus: Prometheus metrics collection and logging utilities
"""

from dynamo.common.utils import env, prometheus

__all__ = ["env", "prometheus"]
