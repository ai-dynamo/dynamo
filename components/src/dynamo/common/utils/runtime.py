# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Common runtime utilities shared across Dynamo engine backends.

Provides:
    - parse_endpoint: Parse 'dyn://namespace.component.endpoint' strings
    - graceful_shutdown: Shutdown DistributedRuntime with optional event signaling
    - create_runtime: Create DistributedRuntime.
"""

import asyncio
import os
from typing import Tuple

from dynamo.runtime import DistributedRuntime


def parse_endpoint(endpoint: str) -> Tuple[str, str, str]:
    """Parse a Dynamo endpoint string into its components.

    Args:
        endpoint: Endpoint string in format 'namespace.component.endpoint'
            or 'dyn://namespace.component.endpoint'.

    Returns:
        Tuple of (namespace, component, endpoint_name).

    Raises:
        ValueError: If endpoint format is invalid.
    """
    endpoint_str = endpoint.replace("dyn://", "", 1)
    endpoint_parts = endpoint_str.split(".")
    if len(endpoint_parts) != 3:
        raise ValueError(
            f"Invalid endpoint format: '{endpoint}'. "
            "Expected 'dyn://namespace.component.endpoint' or 'namespace.component.endpoint'."
        )
    namespace, component, endpoint_name = endpoint_parts
    return namespace, component, endpoint_name


def create_runtime(
    discovery_backend: str,
    request_plane: str,
    event_plane: str,
    use_kv_events: bool,
) -> Tuple[DistributedRuntime, asyncio.AbstractEventLoop]:
    """Create a DistributedRuntime.

    Sets DYN_EVENT_PLANE in the environment, computes whether NATS is needed,
    and creates the runtime.

    Args:
        discovery_backend: Discovery backend type (kubernetes, etcd, file, mem).
        request_plane: Request distribution method (nats, http, tcp).
        event_plane: Event publishing method (nats, zmq).
        use_kv_events: Whether KV events are enabled.

    Returns:
        Tuple of (runtime, event_loop).
    """
    loop = asyncio.get_running_loop()

    os.environ["DYN_EVENT_PLANE"] = event_plane

    enable_nats = request_plane == "nats" or (event_plane == "nats" and use_kv_events)

    runtime = DistributedRuntime(loop, discovery_backend, request_plane, enable_nats)

    return runtime, loop
