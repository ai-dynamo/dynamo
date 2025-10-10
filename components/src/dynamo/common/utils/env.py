# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Environment variable utilities for Dynamo components.

This module provides common utilities for parsing and handling environment variables.

Note: The function names in this module match their Rust counterparts in lib/runtime/src/config.rs
to maintain consistency across the codebase.
"""

import os


def is_truthy(val: str) -> bool:
    """
    Check if a string is truthy.

    This matches the Rust function `is_truthy` in lib/runtime/src/config.rs.
    This will be used to evaluate environment variables or any other subjective
    configuration parameters that can be set by the user that should be evaluated
    as a boolean value.

    Args:
        val: String value to check

    Returns:
        True if the value is "1", "true", "on", or "yes" (case-insensitive).
        False otherwise.

    Example:
        is_truthy("1")     # True
        is_truthy("true")  # True
        is_truthy("TRUE")  # True
        is_truthy("on")    # True
        is_truthy("yes")   # True
        is_truthy("0")     # False
        is_truthy("false") # False
    """
    return val.lower() in ("1", "true", "on", "yes")


def is_falsey(val: str) -> bool:
    """
    Check if a string is falsey.

    This matches the Rust function `is_falsey` in lib/runtime/src/config.rs.
    This will be used to evaluate environment variables or any other subjective
    configuration parameters that can be set by the user that should be evaluated
    as a boolean value (opposite of is_truthy).

    Args:
        val: String value to check

    Returns:
        True if the value is "0", "false", "off", or "no" (case-insensitive).
        False otherwise.

    Example:
        is_falsey("0")     # True
        is_falsey("false") # True
        is_falsey("FALSE") # True
        is_falsey("off")   # True
        is_falsey("no")    # True
        is_falsey("1")     # False
        is_falsey("true")  # False
    """
    return val.lower() in ("0", "false", "off", "no")


def env_is_truthy(name: str) -> bool:
    """
    Check if an environment variable is truthy.

    This matches the Rust function `env_is_truthy` in lib/runtime/src/config.rs.

    Args:
        name: Environment variable name

    Returns:
        True if the environment variable is set to a truthy value (1, true, on, yes).
        False if not set or set to any other value.

    Example:
        export DYN_ENGINE_METRICS_ENABLED=1     # True
        export DYN_ENGINE_METRICS_ENABLED=true  # True
        export DYN_ENGINE_METRICS_ENABLED=on    # True
        export DYN_ENGINE_METRICS_ENABLED=yes   # True
        export DYN_ENGINE_METRICS_ENABLED=0     # False
        export DYN_ENGINE_METRICS_ENABLED=false # False
    """
    value = os.getenv(name)
    if value is None:
        return False

    return is_truthy(value)


def env_is_falsey(name: str) -> bool:
    """
    Check if an environment variable is falsey.

    This matches the Rust function `env_is_falsey` in lib/runtime/src/config.rs.

    Args:
        name: Environment variable name

    Returns:
        True if the environment variable is set to a falsey value (0, false, off, no).
        False if not set or set to any other value.

    Example:
        export DYN_ENGINE_METRICS_ENABLED=0     # True
        export DYN_ENGINE_METRICS_ENABLED=false # True
        export DYN_ENGINE_METRICS_ENABLED=off   # True
        export DYN_ENGINE_METRICS_ENABLED=no    # True
        export DYN_ENGINE_METRICS_ENABLED=1     # False
        export DYN_ENGINE_METRICS_ENABLED=true  # False
    """
    value = os.getenv(name)
    if value is None:
        return False

    return is_falsey(value)
