# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dynamo Common Module

This module contains shared utilities and components used across multiple
Dynamo backends and components.

Main submodules:
    - config_dump: Configuration dumping and system diagnostics utilities
    - snapshot: Snapshot checkpoint/restore lifecycle helpers
    - utils: Common utilities including environment and prometheus helpers
"""

from importlib import import_module
from types import ModuleType

try:
    from ._version import __version__
except Exception:
    try:
        from importlib.metadata import version as _pkg_version

        __version__ = _pkg_version("ai-dynamo")
    except Exception:
        __version__ = "0.0.0+unknown"

__all__ = ["__version__", "config_dump", "constants", "snapshot", "utils"]


def __getattr__(name: str) -> ModuleType:
    if name in {"config_dump", "constants", "snapshot", "utils"}:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
