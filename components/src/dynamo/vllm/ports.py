# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Port allocation and management utilities for Dynamo services."""

import asyncio
import json
import logging
import os
import random
import socket
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from dynamo.runtime import EtcdKvCache

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
class EtcdContext:
    """Context for ETCD operations"""

    client: EtcdKvCache  # etcd client instance
    namespace: str  # Namespace for keys (used in key prefix)

    def make_port_key(self, port: int) -> str:
        """Generate ETCD key for a port reservation"""
        node_ip = get_host_ip()
        return f"dyn://{self.namespace}/ports/{node_ip}/{port}"


@dataclass
class PortMetadata:
    """Metadata to store with port reservations in ETCD"""

    worker_id: str  # Worker identifier (e.g., "vllm-backend-dp0")
    reason: str  # Purpose of the port (e.g., "nixl_side_channel_port")
    block_info: dict = field(default_factory=dict)  # Optional block allocation info

    def to_etcd_value(self) -> dict:
        """Convert to dictionary for ETCD storage"""
        value = {
            "worker_id": self.worker_id,
            "reason": self.reason,
            "reserved_at": time.time(),
            "pid": os.getpid(),
        }
        if self.block_info:
            value.update(self.block_info)
        return value


@dataclass
class PortAllocationRequest:
    """Parameters for port allocation"""

    etcd_context: EtcdContext
    metadata: PortMetadata
    port_range: DynamoPortRange
    block_size: int = 1
    max_attempts: int = 100


@contextmanager
def hold_ports(ports: int | list[int]):
    """Context manager to hold port binding(s).

    Holds socket bindings to ensure exclusive access to ports during reservation.
    Can handle a single port or multiple ports.

    Args:
        ports: Single port number or list of port numbers to hold
    """
    if isinstance(ports, int):
        ports = [ports]

    sockets = []
    try:
        for port in ports:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("", port))
            sockets.append(sock)

        yield

    finally:
        for sock in sockets:
            sock.close()


def check_port_available(port: int) -> bool:
    """Check if a specific port is available for binding."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", port))
            return True
    except OSError:
        return False


async def reserve_port_in_etcd(
    etcd_context: EtcdContext,
    port: int,
    metadata: PortMetadata,
) -> None:
    """Reserve a single port in ETCD."""
    key = etcd_context.make_port_key(port)
    value = metadata.to_etcd_value()

    await etcd_context.client.kv_create(
        key=key,
        value=json.dumps(value).encode(),
        lease_id=etcd_context.client.primary_lease_id(),
    )


async def allocate_and_reserve_port_block(request: PortAllocationRequest) -> list[int]:
    """
    Allocate a contiguous block of ports from the specified range and atomically reserve them in ETCD.
    Returns a list of all allocated ports in order.

    This function uses a context manager to hold port bindings while reserving in ETCD,
    preventing race conditions between multiple processes.

    Args:
        request: PortAllocationRequest containing all allocation parameters

    Returns:
        list[int]: List of all allocated ports in ascending order

    Raises:
        RuntimeError: If unable to reserve a port block within max_attempts
        OSError: If unable to create sockets (system resource issues)
    """
    # Create a list of valid starting ports (must have room for the entire block)
    max_start_port = request.port_range.max - request.block_size + 1
    if max_start_port < request.port_range.min:
        raise ValueError(
            f"Port range {request.port_range.min}-{request.port_range.max} is too small for block size {request.block_size}"
        )

    available_start_ports = list(range(request.port_range.min, max_start_port + 1))
    random.shuffle(available_start_ports)

    actual_max_attempts = min(len(available_start_ports), request.max_attempts)

    for attempt in range(1, actual_max_attempts + 1):
        start_port = available_start_ports[attempt - 1]
        ports_to_reserve = list(range(start_port, start_port + request.block_size))

        try:
            # Try to bind to all ports in the block atomically
            with hold_ports(ports_to_reserve):
                logger.debug(
                    f"Successfully bound to ports {ports_to_reserve}, now reserving in ETCD"
                )

                # We have exclusive access to these ports, now reserve them in ETCD
                for i, port in enumerate(ports_to_reserve):
                    port_metadata = PortMetadata(
                        worker_id=f"{request.metadata.worker_id}-{i}"
                        if request.block_size > 1
                        else request.metadata.worker_id,
                        reason=request.metadata.reason,
                        block_info={
                            "block_index": i,
                            "block_size": request.block_size,
                            "block_start": start_port,
                        }
                        if request.block_size > 1
                        else {},
                    )

                    await reserve_port_in_etcd(
                        etcd_context=request.etcd_context,
                        port=port,
                        metadata=port_metadata,
                    )

                logger.debug(
                    f"Reserved port block {ports_to_reserve} from range {request.port_range.min}-{request.port_range.max} "
                    f"for {request.metadata.worker_id} (block_size={request.block_size})"
                )
                return ports_to_reserve

        except OSError as e:
            logger.debug(
                f"Failed to bind to port block starting at {start_port} (attempt {attempt}): {e}"
            )
        except Exception as e:
            logger.debug(
                f"Failed to reserve port block starting at {start_port} in ETCD (attempt {attempt}): {e}"
            )

        if attempt < actual_max_attempts:
            await asyncio.sleep(0.01)

    raise RuntimeError(
        f"Failed to allocate and reserve a port block of size {request.block_size} from range "
        f"{request.port_range.min}-{request.port_range.max} after {actual_max_attempts} attempts"
    )


async def allocate_and_reserve_port(
    etcd_context: EtcdContext,
    metadata: PortMetadata,
    port_range: DynamoPortRange,
    max_attempts: int = 100,
) -> int:
    """
    Allocate a port from the specified range and atomically reserve it in ETCD.
    This is a convenience wrapper around allocate_and_reserve_port_block with block_size=1.

    Args:
        etcd_context: ETCD context for operations
        metadata: Port metadata for ETCD storage
        port_range: DynamoPortRange object specifying min and max ports to try
        max_attempts: Maximum number of ports to try (default: 100)

    Returns:
        int: The allocated port number

    Raises:
        RuntimeError: If unable to reserve a port within max_attempts
        OSError: If unable to create sockets (system resource issues)
    """
    request = PortAllocationRequest(
        etcd_context=etcd_context,
        metadata=metadata,
        port_range=port_range,
        block_size=1,
        max_attempts=max_attempts,
    )
    allocated_ports = await allocate_and_reserve_port_block(request)
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
