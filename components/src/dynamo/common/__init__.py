# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dynamo Common Module

This module contains shared utilities and components used across multiple
Dynamo backends and components.

Main submodules:
    - config_dump: Configuration dumping and system diagnostics utilities
    - utils: Common utilities including environment and prometheus helpers
"""

from dynamo.common import config_dump, utils

try:
    from ._version import __version__
except Exception:
    try:
        from importlib.metadata import version as _pkg_version

        __version__ = _pkg_version("ai-dynamo")
    except Exception:
        __version__ = "0.0.0+unknown"

__all__ = ["__version__", "config_dump", "utils"]
