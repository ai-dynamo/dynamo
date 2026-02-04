# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for ArgGroup configuration."""

import argparse
import os
from typing import TypeVar

T = TypeVar("T")


def env_or_default(env_var: str, default: T) -> T:
    """
    Get value from environment variable or return default.

    Performs type conversion based on the default value's type.

    Args:
        env_var: Environment variable name (e.g., "DYN_NAMESPACE")
        default: Default value if env var not set

    Returns:
        Environment variable value (type-converted) or default

    Examples:
        >>> env_or_default("DYN_NAMESPACE", "test")
        "test"  # if DYN_NAMESPACE not set
        >>> env_or_default("DYN_MIGRATION_LIMIT", 0)
        5  # if DYN_MIGRATION_LIMIT="5"
    """
    value = os.environ.get(env_var)
    if value is None:
        return default

    # Type conversion based on default type
    if isinstance(default, bool):
        return value.lower() in ("true", "1", "yes", "on")  # type: ignore
    elif isinstance(default, int):
        return int(value)  # type: ignore
    elif isinstance(default, float):
        return float(value)  # type: ignore
    else:
        return value  # type: ignore


def add_negatable_bool(
    parser, flag_name: str, env_var: str, default: bool, help: str
) -> None:
    """
    Add negatable boolean flag (--foo / --no-foo).

    Args:
        parser: ArgumentParser or argument group
        flag_name: Flag name without dashes (e.g., "enable-feature")
        env_var: Environment variable name (e.g., "DYN_ENABLE_FEATURE")
        default: Default value
        help: Help text
    """
    dest = flag_name.replace("-", "_")
    default_with_env = env_or_default(env_var, default)

    parser.add_argument(
        f"--{flag_name}",
        dest=dest,
        action=argparse.BooleanOptionalAction,
        default=default_with_env,
        help=f"{help} (env: {env_var}, default: {default_with_env})",
    )
