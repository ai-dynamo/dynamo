# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from typing import Any

from dynamo.planner.utils.planner_argparse import create_sla_planner_parser


def _get_action_type(action: argparse.Action) -> str | None:
    """
    Extract action type string from an argparse Action object.

    Args:
        action: The argparse Action object

    Returns:
        Action type string ('store_true', 'store_false', 'store_const',
        'boolean_optional') or None
    """
    action_class_name = type(action).__name__
    if action_class_name == "_StoreTrueAction":
        return "store_true"
    elif action_class_name == "_StoreFalseAction":
        return "store_false"
    elif action_class_name == "_StoreConstAction":
        return "store_const"
    elif action_class_name == "BooleanOptionalAction":
        return "boolean_optional"
    return None


def _build_action_kwargs(
    action: argparse.Action, action_type: str | None, prefix: str
) -> dict:
    """
    Build kwargs dictionary for add_argument based on action type.

    Args:
        action: The argparse Action object
        action_type: The action type string ('store_true', 'store_false', etc.)
        prefix: Prefix for the destination name (empty string for no prefix)

    Returns:
        Dictionary of kwargs for add_argument
    """
    dest = f"{prefix.replace('-', '_')}{action.dest}" if prefix else action.dest
    kwargs = {
        "dest": dest,
        "default": action.default,
        "help": action.help,
    }

    # Add action type if specified
    if action_type is not None:
        if action_type == "boolean_optional":
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            kwargs["action"] = action_type

    # For store_true/store_false/boolean_optional, don't add type, nargs, metavar, const
    # For other actions, add them if they're set
    if action_type not in ["store_true", "store_false", "boolean_optional"]:
        if action.type is not None:
            kwargs["type"] = action.type
        if action.nargs is not None:
            kwargs["nargs"] = action.nargs
        if action.metavar is not None:
            kwargs["metavar"] = action.metavar
        if action.choices is not None:
            kwargs["choices"] = action.choices
        if action_type == "store_const" and action.const is not None:
            kwargs["const"] = action.const

    return kwargs


def _get_planner_defaults() -> dict[str, Any]:
    """
    Get default values for all planner arguments from the planner parser.

    Returns:
        Dictionary mapping argument names (with dashes) to their default values
    """
    planner_parser = create_sla_planner_parser()
    defaults = {}
    for action in planner_parser._actions:
        if action.dest == "help" or not action.option_strings:
            continue
        # Convert dest (underscores) to arg name (dashes)
        arg_name = action.dest.replace("_", "-")
        defaults[arg_name] = action.default
    return defaults


def _format_arg_for_command_line(
    arg_name: str, value, defaults: dict[str, Any] | None = None
) -> list[str]:
    """
    Format an argument name and value for command line usage.

    Args:
        arg_name: The argument name (without dashes)
        value: The argument value
        defaults: Optional dict of default values. If provided and value matches
                  the default, the arg is skipped (allows operator env vars to take effect)

    Returns:
        List of command-line argument strings (empty list if value is None or False bool)
    """
    if value is None:
        return []

    # Skip args that match their default values
    # This allows the operator's injected env vars to take effect
    # (e.g., PLANNER_PROMETHEUS_PORT=9085 won't be overridden by --prometheus-port=0)
    if defaults is not None and arg_name in defaults:
        if value == defaults[arg_name]:
            return []

    if isinstance(value, bool):
        # For boolean flags, only add if True
        if value:
            return [f"--{arg_name}"]
        return []
    else:
        # For valued arguments
        return [f"--{arg_name}={value}"]


def _collect_args_from_namespace(
    args: argparse.Namespace,
    arg_names: list[str],
    prefix_to_strip: str = "",
    defaults: dict[str, Any] | None = None,
) -> list[str]:
    """
    Collect and format command-line arguments from a namespace for given attribute names.

    Args:
        args: The argparse Namespace containing parsed arguments
        arg_names: List of attribute names to collect from the namespace
        prefix_to_strip: Optional prefix to remove from attribute names before formatting
        defaults: Optional dict of default values. Args matching defaults are skipped.

    Returns:
        List of formatted command-line argument strings
    """
    result = []
    for attr_name in sorted(arg_names):  # sorted for consistent ordering
        value = getattr(args, attr_name)
        # Strip prefix and convert to command-line argument name
        if prefix_to_strip and attr_name.startswith(prefix_to_strip):
            arg_name = attr_name[len(prefix_to_strip) :].replace("_", "-")
        else:
            arg_name = attr_name.replace("_", "-")
        result.extend(_format_arg_for_command_line(arg_name, value, defaults))
    return result


def add_planner_arguments_to_parser(
    parser: argparse.ArgumentParser, prefix: str = "planner-"
):
    """
    Dynamically add planner arguments from create_sla_planner_parser() to the given parser.
    Only adds arguments that don't already exist in the parser.

    Planner flags are added without any prefix so that BooleanOptionalAction
    (e.g. --load-predictor-log1p / --no-load-predictor-log1p) works correctly.

    Args:
        parser: The ArgumentParser to add arguments to
        prefix: Optional prefix for dest only (default ""). Do not use a flag prefix.
    """
    # Create a temporary planner parser to extract its arguments
    planner_parser = create_sla_planner_parser()

    # Get existing argument names in the parser
    existing_dests = {action.dest for action in parser._actions}

    # Add a group for planner arguments
    planner_group = parser.add_argument_group(
        "planner arguments",
        "Arguments that will be passed to the planner service (only showing args not already in profile_sla)",
    )

    # Use planner's option strings as-is (no flag prefix) so BooleanOptionalAction works
    for action in planner_parser._actions:
        # Skip help and positional arguments
        if action.dest in ["help"] or not action.option_strings:
            continue

        # Skip if this argument already exists in the main parser
        if action.dest in existing_dests:
            continue

        option_strings = action.option_strings

        # Determine the action type and build kwargs
        action_type = _get_action_type(action)
        kwargs = _build_action_kwargs(action, action_type, prefix)

        planner_group.add_argument(*option_strings, **kwargs)


def build_planner_args_from_namespace(
    args: argparse.Namespace, prefix: str = ""
) -> list[str]:
    """
    Build planner command-line arguments from parsed args namespace.
    Collects all planner argument dests that exist in the namespace (no flag prefix).

    Args that match their default values are skipped, allowing the operator's
    injected environment variables to take effect (e.g., PLANNER_PROMETHEUS_PORT).

    Args:
        args: Parsed arguments namespace
        prefix: Optional dest prefix to strip when collecting (default ""). Unused when no prefix.

    Returns:
        List of planner command-line arguments
    """
    defaults = _get_planner_defaults()

    planner_parser = create_sla_planner_parser()
    planner_arg_dests = {
        action.dest
        for action in planner_parser._actions
        if action.dest != "help" and action.option_strings
    }

    shared_arg_dests = {dest for dest in planner_arg_dests if hasattr(args, dest)}
    prefix_to_strip = prefix.replace("-", "_") if prefix else ""

    planner_args = _collect_args_from_namespace(
        args, list(shared_arg_dests), prefix_to_strip=prefix_to_strip, defaults=defaults
    )

    if prefix:
        prefixed_attrs = [attr for attr in dir(args) if attr.startswith(prefix)]
        planner_args.extend(
            _collect_args_from_namespace(
                args, prefixed_attrs, prefix_to_strip=prefix, defaults=defaults
            )
        )

    return planner_args
