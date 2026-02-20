# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Common runtime utilities shared across Dynamo engine backends.

Provides:
    - parse_endpoint: Parse 'dyn://namespace.component.endpoint' strings
    - graceful_shutdown: Shutdown DistributedRuntime with optional event signaling
    - create_runtime: Create DistributedRuntime with signal handlers
"""

import asyncio
import logging
import os
import signal
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


async def graceful_shutdown(
    runtime: DistributedRuntime,
    shutdown_event: Optional[asyncio.Event] = None,
) -> None:
    """Shutdown DistributedRuntime with optional event signaling.

    Args:
        runtime: The DistributedRuntime instance to shut down.
        shutdown_event: Optional event to set before shutting down,
            signaling in-flight handlers to finish.
    """
    logging.info("Received shutdown signal, shutting down DistributedRuntime")
    if shutdown_event is not None:
        shutdown_event.set()
    runtime.shutdown()
    logging.info("DistributedRuntime shutdown complete")


def create_runtime(
    discovery_backend: str,
    request_plane: str,
    event_plane: str,
    use_kv_events: bool,
    shutdown_event: Optional[asyncio.Event] = None,
) -> Tuple[DistributedRuntime, asyncio.AbstractEventLoop]:
    """Create a DistributedRuntime and register signal handlers for graceful shutdown.

    Sets DYN_EVENT_PLANE in the environment, computes whether NATS is needed,
    creates the runtime, and installs SIGTERM/SIGINT handlers.

    Args:
        discovery_backend: Discovery backend type (kubernetes, etcd, file, mem).
        request_plane: Request distribution method (nats, http, tcp).
        event_plane: Event publishing method (nats, zmq).
        use_kv_events: Whether KV events are enabled.
        shutdown_event: Optional event to set on shutdown signal.

    Returns:
        Tuple of (runtime, event_loop).
    """
    loop = asyncio.get_running_loop()

    os.environ["DYN_EVENT_PLANE"] = event_plane

    enable_nats = request_plane == "nats" or (event_plane == "nats" and use_kv_events)

    runtime = DistributedRuntime(loop, discovery_backend, request_plane, enable_nats)

    def signal_handler():
        asyncio.create_task(graceful_shutdown(runtime, shutdown_event))

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    logging.debug("Signal handlers set up for graceful shutdown")

    return runtime, loop
