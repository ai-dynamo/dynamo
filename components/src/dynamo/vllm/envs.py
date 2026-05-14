# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Environment variable configuration for Dynamo vLLM integration.

This module provides a centralized location for managing environment variables
used by Dynamo's vLLM backend, following vLLM's pattern.
"""

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

# TODO: move this to configuration system.

# Port range constants
REGISTERED_PORT_MIN = 1024
REGISTERED_PORT_MAX = 49151

if TYPE_CHECKING:
    DYN_FORWARDPASS_METRIC_PORT: int = 20380
    DYN_VLLM_KV_EVENT_BLOCK_SIZE: int


def _resolve_port(env_var: str, default_port: int) -> int:
    """
    Resolve port from environment variable with validation.

    Args:
        env_var: Environment variable name
        default_port: Default port if env var not set

    Returns:
        Validated port number

    Raises:
        ValueError: If port is invalid or out of range
    """
    env_value = os.getenv(env_var)
    if env_value is None:
        port = default_port
    else:
        try:
            port = int(env_value)
        except ValueError as exc:
            raise ValueError(
                f"{env_var} must be an integer port number, got {env_value!r}."
            ) from exc

    if not (REGISTERED_PORT_MIN <= port <= REGISTERED_PORT_MAX):
        raise ValueError(
            f"{env_var} port {port} is outside of the registered port range "
            f"({REGISTERED_PORT_MIN}-{REGISTERED_PORT_MAX})."
        )

    return port


def _resolve_positive_int(env_var: str) -> int:
    """
    Resolve a positive integer from an environment variable.

    Args:
        env_var: Environment variable name

    Returns:
        Parsed positive integer value

    Raises:
        ValueError: If the env var is set to a non-integer or non-positive value
    """
    env_value = os.getenv(env_var)
    if env_value is None:
        raise ValueError(f"{env_var} is not set.")

    try:
        value = int(env_value)
    except ValueError as exc:
        raise ValueError(
            f"{env_var} must be a positive integer, got {env_value!r}."
        ) from exc

    if value <= 0:
        raise ValueError(f"{env_var} must be a positive integer, got {value}.")

    return value


# Environment variables configuration
environment_variables: dict[str, Callable[[], Any]] = {
    "DYN_FORWARDPASS_METRIC_PORT": lambda: _resolve_port(
        "DYN_FORWARDPASS_METRIC_PORT", 20380
    ),
    "DYN_VLLM_KV_EVENT_BLOCK_SIZE": lambda: _resolve_positive_int(
        "DYN_VLLM_KV_EVENT_BLOCK_SIZE"
    ),
}


def __getattr__(name: str):
    """
    Gets environment variables lazily.
    """
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())


def is_set(name: str) -> bool:
    """Check if an environment variable is explicitly set."""
    if name in environment_variables:
        return name in os.environ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
