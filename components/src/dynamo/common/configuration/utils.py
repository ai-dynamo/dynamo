# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for ArgGroup configuration."""

import argparse
import os
from typing import Any, Optional, TypeVar

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
    elif isinstance(default, list):
        # Env vars for list options (e.g. DYN_CONNECTOR) are space-separated; downstream expects a list.
        return [x.strip() for x in value.split() if x.strip()]  # type: ignore
    else:
        return value  # type: ignore


def add_argument(
    parser,
    *,
    flag_name: str,
    env_var: str,
    default: Any,
    help: str,
    obsolete_flag: Optional[str] = None,
    arg_type: Optional[type] = str,
    **kwargs: Any,
) -> None:
    """
    Add a CLI argument with env var default, optional alias and dest, and help message construction.

    Args:
        parser: ArgumentParser or argument group
        flag_name: Primary flag (must start with '--', e.g., "--foo")
        env_var: Environment variable name (e.g., "DYN_FOO")
        default: Default value
        help: Help text
        alias: Optional alias for the flag (must start with '--')
        obsolete_flag: Optional obsolete/legacy flag (for help msg only, must start with '--')
        dest: Optional destination name (defaults to flag_name with dashes replaced by underscores)
        choices: Optional list of valid values for the argument.
        arg_type: Type for the argument (default: str)
    """
    arg_dest = _get_dest_name(flag_name, kwargs.get("dest"))
    default_with_env = env_or_default(env_var, default)

    names = [flag_name]

    env_help = _build_help_message(help, env_var, default_with_env, obsolete_flag)

    add_arg_opts = {
        "dest": arg_dest,
        "default": default_with_env,
        "help": env_help,
        "type": arg_type,
    }
    kwargs.update(add_arg_opts)

    parser.add_argument(*names, **kwargs)


def add_negatable_bool_argument(
    parser,
    *,
    flag_name: str,
    env_var: str,
    default: bool,
    help: str,
    dest: Optional[str] = None,
) -> None:
    """
    Add negatable boolean flag (--foo / --no-foo).

    Args:
        parser: ArgumentParser or argument group
        flag_name: Primary flag (must start with '--', e.g. "--enable-feature")
        env_var: Environment variable name (e.g., "DYN_ENABLE_FEATURE")
        default: Default value
        help: Help text
    """
    arg_dest = _get_dest_name(flag_name, dest)
    default_with_env = env_or_default(env_var, default)

    parser.add_argument(
        flag_name,
        dest=arg_dest,
        action=argparse.BooleanOptionalAction,
        default=default_with_env,
        help=_build_help_message(help, env_var, default),
    )


def _build_help_message(
    help_text: str, env_var: str, default: Any, obsolete_flag: Optional[str] = None
) -> str:
    """
    Build help message with env var and default value.
    """
    if obsolete_flag:
        return f"{help_text}\nenv var: {env_var} | default: {default}\nobsolete flag: {obsolete_flag}"
    return f"{help_text}\nenv var: {env_var} | default: {default}"


def _get_dest_name(flag_name: str, dest: Optional[str] = None) -> str:
    """
    Get the destination name for the flag.
    """
    return dest if dest else flag_name.lstrip("-").replace("-", "_")
