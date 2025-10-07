# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dynamo Common Module

This module contains shared utilities and components used across multiple
Dynamo backends and components.

Main submodules:
    - config_dump: Configuration dumping and system diagnostics utilities
"""

from dynamo.common import config_dump
from dynamo.common._version import __version__

__all__ = ["__version__", "config_dump"]
