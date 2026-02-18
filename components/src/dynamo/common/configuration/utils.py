# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for ArgGroup configuration."""

import argparse
import os
from typing import Any, Callable, Optional, TypeVar, Union

T = TypeVar("T")


def env_or_default(
    env_var: str,
    default: T,
    value_type: Optional[Union[type, Callable[..., Any]]] = None,
) -> T:
    """
    Get value from environment variable or return default.

    Performs type conversion based on the default value's type.

    Args:
        env_var: Environment variable name (e.g., "DYN_NAMESPACE")
        default: Default value if env var not set
        value_type: If provided, use this type to convert the env value. If None, the type
        is taken from type(default). Use value_type when default is None but you still
        want the env value coerced (e.g. env_or_default("DYN_FOO", None, value_type=int)).

    Returns:
        Environment variable value (type-converted) or default
    """
    value = os.environ.get(env_var)
    if value is None:
        return default

    # No type info available: default=None and no explicit value_type.
    if value_type is None and default is None:
        return value  # type: ignore[return-value]

    # Prefer the explicit type if provided; otherwise derive from default
    target_type = value_type if value_type is not None else type(default)

    if target_type is bool:
        return value.lower() in ("true", "1", "yes", "on")  # type: ignore
    if target_type is int:
        return int(value)  # type: ignore
    if target_type is float:
        return float(value)  # type: ignore
    if target_type is list:
        return [x.strip() for x in value.split() if x.strip()]  # type: ignore

    # Fall back to calling the type/callable for custom validators (e.g., pathlib.Path)
    return target_type(value) if callable(target_type) else value  # type: ignore


def add_argument(
    parser,
    *,
    flag_name: str,
    env_var: str,
    default: Any,
    help: str,
    obsolete_flag: Optional[str] = None,
    arg_type: Optional[Union[type, Callable[..., Any]]] = str,
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
    value_type_for_env: Optional[Union[type, Callable[..., Any]]] = None
    if arg_type is not None and isinstance(arg_type, type):
        value_type_for_env = arg_type
    if isinstance(default, list) and (arg_type is None or arg_type is str):
        value_type_for_env = None
    default_with_env = env_or_default(env_var, default, value_type=value_type_for_env)

    names = [flag_name]

    if obsolete_flag:
        # Accept obsolete flag as an alias (still show deprecation note in help)
        names.append(obsolete_flag)

    env_help = _build_help_message(help, env_var, default, obsolete_flag)

    add_arg_opts = {
        "dest": arg_dest,
        "default": default_with_env,
        "help": env_help,
    }
    if arg_type is not None:
        add_arg_opts["type"] = arg_type
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
    obsolete_flag: Optional[str] = None,
) -> None:
    """
    Add negatable boolean flag (--foo / --no-foo).

    Args:
        parser: ArgumentParser or argument group
        flag_name: Primary flag (must start with '--', e.g. "--enable-feature")
        env_var: Environment variable name (e.g., "DYN_ENABLE_FEATURE")
        default: Default value
        help: Help text
        dest: Optional destination name for the parsed value
        obsolete_flag: Optional obsolete/legacy flag (for help msg only, must start with '--')
    """
    add_argument(
        parser,
        flag_name=flag_name,
        env_var=env_var,
        default=default,
        help=help,
        dest=dest,
        obsolete_flag=obsolete_flag,
        arg_type=None,
        action=argparse.BooleanOptionalAction,
    )


def _build_help_message(
    help_text: str, env_var: str, default: Any, obsolete_flag: Optional[str] = None
) -> str:
    """
    Build help message with env var and default value.
    """
    if obsolete_flag:
        return f"{help_text}\nenv var: {env_var} | default: {default}\ndeprecating flag: {obsolete_flag}"
    return f"{help_text}\nenv var: {env_var} | default: {default}"


def _get_dest_name(flag_name: str, dest: Optional[str] = None) -> str:
    """
    Get the destination name for the flag.
    """
    return dest if dest else flag_name.lstrip("-").replace("-", "_")
