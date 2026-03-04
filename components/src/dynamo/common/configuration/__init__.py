# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ArgGroup-based configuration system for Dynamo.

This module provides a modular, domain-driven configuration architecture where:
- Each ArgGroup owns a specific domain of configuration parameters
- Components declare which ArgGroups they need
- Unrecognized arguments are captured for backend engines (passthrough)
"""

from .arg_group import ArgGroup
from .config_base import ConfigBase
from .utils import add_argument, add_negatable_bool_argument, env_or_default

__all__ = [
    # Base classes
    "ArgGroup",
    "ConfigBase",
    # Utilities
    "add_argument",
    "env_or_default",
    "add_negatable_bool_argument",
]
