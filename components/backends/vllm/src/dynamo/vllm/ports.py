# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Port allocation and management utilities for Dynamo services."""

import json
import logging
import os
import socket
import time
from dataclasses import dataclass

from dynamo.runtime import DistributedRuntime

logger = logging.getLogger(__name__)

# Default port range in the registered ports section
DEFAULT_DYNAMO_PORT_MIN = 20000
DEFAULT_DYNAMO_PORT_MAX = 30000


@dataclass
class DynamoPortRange:
    """Port range configuration for Dynamo services"""

    min: int
    max: int

    def __post_init__(self):
        if self.min < 1024 or self.max > 49151:
            raise ValueError(
                f"Port range {self.min}-{self.max} is outside registered ports range (1024-49151)"
            )
        if self.min >= self.max:
            raise ValueError(
                f"Invalid port range: min ({self.min}) must be less than max ({self.max})"
            )


@dataclass
class PortMetadata:
    """Metadata to store with port reservations"""

    worker_id: str  # Worker identifier (e.g., "vllm-backend-dp0")
    reason: str  # Purpose of the port (e.g., "nixl_side_channel_port")


@dataclass
class PortAllocationRequest:
    """Parameters for port allocation"""

    metadata: PortMetadata
    port_range: DynamoPortRange
    block_size: int = 1

    def __post_init__(self):
        if self.block_size < 1:
            raise ValueError("block_size must be >= 1")
        range_len = self.port_range.max - self.port_range.min + 1
        if self.block_size > range_len:
            raise ValueError(
                f"block_size {self.block_size} exceeds range length {range_len} "
                f"({self.port_range.min}-{self.port_range.max})"
            )


def check_port_available(port: int) -> bool:
    """Check if a specific port is available for binding."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", port))
            return True
    except OSError:
        return False


async def allocate_and_reserve_port_block(
    runtime: DistributedRuntime, namespace: str, request: PortAllocationRequest
) -> list[int]:
    """
    Allocate a contiguous block of ports from the specified range and atomically reserve them.
    Returns a list of all allocated ports in order.

    Args:
        request: PortAllocationRequest containing all allocation parameters

    Returns:
        list[int]: List of all allocated ports in ascending order
    """
    # Create a list of valid starting ports (must have room for the entire block)

    context_json = {
        "worker_id": str(request.metadata.worker_id),
        "reason": request.metadata.reason,
        "reserved_at": time.time(),
        "pid": os.getpid(),
        "block_size": request.block_size,
    }

    return await runtime.allocate_port_block(
        namespace,
        request.port_range.min,
        request.port_range.max,
        request.block_size,
        json.dumps(context_json),
    )


async def allocate_and_reserve_port(
    runtime: DistributedRuntime,
    namespace: str,
    metadata: PortMetadata,
    port_range: DynamoPortRange,
) -> int:
    """
    Allocate a port from the specified range and atomically reserve it.
    This is a convenience wrapper around allocate_and_reserve_port_block with block_size=1.

    Args:
        metadata: Port metadata / context
        port_range: DynamoPortRange object specifying min and max ports to try

    Returns:
        int: The allocated port number
    """
    request = PortAllocationRequest(
        metadata=metadata,
        port_range=port_range,
        block_size=1,
    )
    allocated_ports = await allocate_and_reserve_port_block(runtime, namespace, request)
    if not allocated_ports:
        raise RuntimeError("Failed to allocate required ports")
    return allocated_ports[0]  # Return the single allocated port


def get_host_ip() -> str:
    """Get the IP address of the host.
    This is needed for the side channel to work in multi-node deployments.
    """
    try:
        host_name = socket.gethostname()
    except socket.error as e:
        logger.warning(f"Failed to get hostname: {e}, falling back to '127.0.0.1'")
        return "127.0.0.1"
    else:
        try:
            # Get the IP address of the hostname - this is needed for the side channel to work in multi-node deployments
            host_ip = socket.gethostbyname(host_name)
            # Test if the IP is actually usable by binding to it
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as test_socket:
                test_socket.bind((host_ip, 0))
            return host_ip
        except socket.gaierror as e:
            logger.warning(
                f"Hostname '{host_name}' cannot be resolved: {e}, falling back to '127.0.0.1'"
            )
            return "127.0.0.1"
        except socket.error as e:
            # If hostname is not usable for binding, fall back to localhost
            logger.warning(
                f"Hostname '{host_name}' is not usable for binding: {e}, falling back to '127.0.0.1'"
            )
            return "127.0.0.1"
