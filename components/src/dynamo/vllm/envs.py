# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Environment variable configuration for Dynamo vLLM integration.

This module provides a centralized location for managing environment variables
used by Dynamo's vLLM backend, following vLLM's pattern.
"""

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

# Port range constants
REGISTERED_PORT_MIN = 1024
REGISTERED_PORT_MAX = 49151

if TYPE_CHECKING:
    DYN_VLLM_KV_EVENT_PORT: int = 20080


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


# Environment variables configuration
environment_variables: dict[str, Callable[[], Any]] = {
    # Port used for KV events publishing to the frontend
    # Note: This env variable is ignored if explicitly using --kv-events-config ''
    "DYN_VLLM_KV_EVENT_PORT": lambda: _resolve_port("DYN_VLLM_KV_EVENT_PORT", 20080),
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
