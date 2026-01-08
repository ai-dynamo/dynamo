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

"""Factory module for selecting client implementation (legacy or AI-Perf)."""

from typing import Callable


def get_client_function(client_type: str) -> Callable:
    """Get the appropriate client function based on client type.

    This factory function returns the correct client implementation without
    requiring the caller to know the internal module structure.

    Args:
        client_type: Type of client to use. Valid options:
            - "aiperf": Use AI-Perf for load generation (default)
            - "legacy": Use legacy custom HTTP client

    Returns:
        Client function that matches the signature:
        client(
            deployment_spec,
            namespace,
            model,
            log_dir,
            index,
            requests_per_client,
            input_token_length,
            output_token_length,
            max_retries,
            retry_delay_or_rate,  # Differs between implementations
            continuous_load,
        )

    Raises:
        ValueError: If client_type is not recognized

    Example:
        >>> client_func = get_client_function("aiperf")
        >>> client_func(deployment_spec, namespace, model, ...)

        >>> legacy_func = get_client_function("legacy")
        >>> legacy_func(deployment_spec, namespace, model, ...)
    """
    if client_type == "aiperf":
        from tests.fault_tolerance.deploy.client import client as aiperf_client

        return aiperf_client
    elif client_type == "legacy":
        from tests.fault_tolerance.deploy.legacy_client import client as legacy_client

        return legacy_client
    else:
        raise ValueError(
            f"Unknown client type: '{client_type}'. "
            f"Valid options are: 'aiperf', 'legacy'"
        )


def get_supported_client_types() -> list[str]:
    """Get list of all supported client types.

    Returns:
        List of valid client type strings
    """
    return ["aiperf", "legacy"]


def validate_client_type(client_type: str) -> bool:
    """Validate that a client type is supported.

    Args:
        client_type: Client type string to validate

    Returns:
        True if valid, False otherwise
    """
    return client_type in get_supported_client_types()


def get_client_description(client_type: str) -> str:
    """Get a human-readable description of a client type.

    Args:
        client_type: Client type to describe

    Returns:
        Description string

    Raises:
        ValueError: If client_type is not recognized
    """
    descriptions = {
        "aiperf": (
            "AI-Perf client: Uses the AI-Perf CLI tool for load generation. "
            "Provides comprehensive metrics including P50/P90/P99 latencies, "
            "TTFT (Time to First Token), ITL (Inter-Token Latency), and throughput. "
            "Outputs results in JSON/CSV format with retry support at the test level."
        ),
        "legacy": (
            "Legacy custom client: Direct HTTP request loop with per-request retry logic. "
            "Logs results in JSONL format with basic latency and status tracking. "
            "Includes rate limiting and round-robin pod selection."
        ),
    }

    if client_type not in descriptions:
        raise ValueError(
            f"Unknown client type: '{client_type}'. "
            f"Valid options are: {', '.join(get_supported_client_types())}"
        )

    return descriptions[client_type]
