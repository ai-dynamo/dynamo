# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Common runtime utilities shared across Dynamo engine backends.

Provides:
    - parse_endpoint: Parse 'dyn://namespace.component.endpoint' strings
    - slugify_model_name: Convert model name to URL-safe endpoint suffix with hash
    - graceful_shutdown: Shutdown DistributedRuntime with optional event signaling
    - create_runtime: Create DistributedRuntime with signal handlers
"""

import asyncio
import hashlib
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


def slugify_model_name(model_name: str) -> str:
    """Convert model name to URL-safe endpoint suffix with hash for uniqueness.

    Creates endpoint names in format: generate_{slug}_{hash8}
    - slug: First 40 chars of slugified model name (lowercase, alphanumeric/-/_)
    - hash8: First 8 chars of sha256 hash of original model name

    If slug is empty (model name was all special chars), uses: generate_model_{hash8}

    This ensures:
    - Human-readable: "Qwen/Qwen2.5-7B-Instruct" → "generate_qwen_qwen2_5-7b-instruct_a1b2c3d4"
    - Collision-resistant: Hash suffix prevents collisions
    - Never empty: Always produces valid endpoint name

    Args:
        model_name: Model name or path (e.g., "Qwen/Qwen2.5-7B-Instruct")

    Returns:
        URL-safe endpoint name with hash suffix

    Examples:
        >>> slugify_model_name("Qwen/Qwen2.5-7B-Instruct")
        "generate_qwen_qwen2_5-7b-instruct_1a2b3c4d"
        >>> slugify_model_name("$%@#!")
        "generate_model_1a2b3c4d"
    """
    # Compute hash first (on original name) - using sha256 from stdlib
    hash_digest = hashlib.sha256(model_name.encode()).hexdigest()[:8]

    # Slugify: lowercase, replace non-alphanumeric (except -_) with _
    slug = model_name.lower()
    slug = "".join(
        c if (c.isascii() and (c.isalnum() or c in ("-", "_"))) else "_" for c in slug
    )
    # Remove leading underscores
    slug = slug.lstrip("_")

    # Truncate to 40 chars for readability
    slug = slug[:40] if slug else "model"

    # Combine: generate_{slug}_{hash}
    return f"generate_{slug}_{hash_digest}"


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
    store_kv: str,
    request_plane: str,
    event_plane: str,
    use_kv_events: bool,
    shutdown_event: Optional[asyncio.Event] = None,
) -> Tuple[DistributedRuntime, asyncio.AbstractEventLoop]:
    """Create a DistributedRuntime and register signal handlers for graceful shutdown.

    Sets DYN_EVENT_PLANE in the environment, computes whether NATS is needed,
    creates the runtime, and installs SIGTERM/SIGINT handlers.

    Args:
        store_kv: Key-value backend type (etcd, file, mem).
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

    runtime = DistributedRuntime(loop, store_kv, request_plane, enable_nats)

    def signal_handler():
        asyncio.create_task(graceful_shutdown(runtime, shutdown_event))

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    logging.debug("Signal handlers set up for graceful shutdown")

    return runtime, loop
