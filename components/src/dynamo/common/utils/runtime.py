# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Common runtime utilities shared across Dynamo engine backends.

Provides:
    - parse_endpoint: Parse 'dyn://namespace.component.endpoint' strings
    - create_runtime: Create DistributedRuntime.
    - run_async: Helper to run async functions in non-async functions that
                 may be run in either sync or async context.
"""

import asyncio
import os
import warnings
from typing import Optional, Tuple

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
    use_kv_events: Optional[bool] = None,
) -> Tuple[DistributedRuntime, asyncio.AbstractEventLoop]:
    """Create a DistributedRuntime.

    Args:
        discovery_backend: Discovery backend type (kubernetes, etcd, file, mem).
        request_plane: Request distribution method (nats, http, tcp).
        event_plane: Event publishing method (nats, zmq).
        use_kv_events: Deprecated. NATS enablement is now determined automatically
            from the event-plane configuration. This parameter is accepted for
            backwards compatibility but will be removed in a future release.

    Returns:
        Tuple of (runtime, event_loop).
    """
    if use_kv_events is not None:
        warnings.warn(
            "The 'use_kv_events' parameter is deprecated and will be removed in a "
            "future release. NATS enablement is now determined automatically from "
            "the event-plane configuration.",
            DeprecationWarning,
            stacklevel=2,
        )

    loop = asyncio.get_running_loop()

    os.environ["DYN_EVENT_PLANE"] = event_plane

    runtime = DistributedRuntime(loop, discovery_backend, request_plane)

    return runtime, loop


def run_async(func, *args, **kwargs):
    """Run an async function as if it is synchronous, handling both sync and async contexts.

    Args:
        func: An async function to execute.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        The result of the async function.
    """
    # Check if we're in async context, exception is raised if not and we can safely
    # run 'func' with asyncio.run()
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(func(*args, **kwargs))

    # In an async context, we want to run 'func' in a separate thread to avoid blocking the event loop
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor() as pool:
        return pool.submit(asyncio.run, func(*args, **kwargs)).result()
