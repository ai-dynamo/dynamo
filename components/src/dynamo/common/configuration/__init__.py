# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ArgGroup-based configuration system for Dynamo.

This module provides a modular, domain-driven configuration architecture where:
- Each ArgGroup owns a specific domain of configuration parameters
- Components declare which ArgGroups they need
- Unrecognized arguments are captured for backend engines (passthrough)
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Declared, not imported at runtime, so the type checker still resolves the
    # names `__getattr__` provides lazily below.
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


# Hidden from the type checker on purpose; see the note in dynamo/common/__init__.py.
if not TYPE_CHECKING:

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
