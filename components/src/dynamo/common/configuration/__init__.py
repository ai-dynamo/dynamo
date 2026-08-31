# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ArgGroup-based configuration system for Dynamo.

This module provides a modular, domain-driven configuration architecture where:
- Each ArgGroup owns a specific domain of configuration parameters
- Components declare which ArgGroups they need
- Unrecognized arguments are captured for backend engines (passthrough)
"""

__all__ = [
    # Base classes
    "ArgGroup",
    "ConfigBase",
    # Utilities
    "add_argument",
    "env_or_default",
    "add_negatable_bool_argument",
]


def __getattr__(name: str):
    if name == "ArgGroup":
        from .arg_group import ArgGroup

        return ArgGroup
    if name == "ConfigBase":
        from .config_base import ConfigBase

        return ConfigBase
    if name in {"add_argument", "add_negatable_bool_argument", "env_or_default"}:
        from . import utils

        return getattr(utils, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
