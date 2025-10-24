# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Path utilities for Dynamo workspace detection.

This module provides utilities for determining the workspace directory
across different environments (development, container, CI).
"""

import os


def get_workspace_dir() -> str:
    """
    Determine the Dynamo workspace directory.

    Returns the workspace directory with the following precedence:
    1. Current working directory if it contains Cargo.toml (running from repo root)
    2. WORKSPACE_DIR environment variable (user-provided override)
    3. /workspace (default container path)
    4. Current working directory (fallback)

    Returns:
        str: Absolute path to the workspace directory

    Examples:
        >>> # Running from repo root
        >>> get_workspace_dir()
        '/home/ubuntu/dynamo'

        >>> # With environment variable override
        >>> os.environ['WORKSPACE_DIR'] = '/custom/path'
        >>> get_workspace_dir()
        '/custom/path'

        >>> # In container default
        >>> get_workspace_dir()
        '/workspace'
    """
    # Check if running from repo root (contains Cargo.toml)
    if os.path.exists(os.path.join(os.getcwd(), "Cargo.toml")):
        return os.getcwd()

    # Check environment variable
    workspace_dir = os.environ.get("WORKSPACE_DIR")
    if workspace_dir:
        return workspace_dir

    # Check container default
    if os.path.exists("/workspace"):
        return "/workspace"

    # Fallback to current directory
    return os.getcwd()


# Global constant for convenience
WORKSPACE_DIR = get_workspace_dir()

__all__ = ["get_workspace_dir", "WORKSPACE_DIR"]
