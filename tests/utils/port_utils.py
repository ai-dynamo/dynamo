# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Port allocation utilities for tests.

Port allocation with flock-based locking to prevent race conditions in parallel tests.
"""

import fcntl
import json
import os
import socket
import tempfile
import time
from pathlib import Path

# Port allocation lock file
_PORT_LOCK_FILE = Path(tempfile.gettempdir()) / "pytest_port_allocations.lock"
_PORT_REGISTRY_FILE = Path(tempfile.gettempdir()) / "pytest_port_allocations.json"

# Port range for allocation (i16 range for Rust compatibility)
_PORT_MIN = 30000
_PORT_MAX = 32767


def _load_port_registry() -> dict:
    """Load the port registry from disk.

    Returns:
        dict: Port registry mapping port numbers (as strings) to timestamps.
              Example: {"30001": 1732647123.456, "30002": 1732647123.789}
    """
    if not _PORT_REGISTRY_FILE.exists():
        return {}
    try:
        with open(_PORT_REGISTRY_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_port_registry(registry: dict) -> None:
    """Save the port registry to disk."""
    with open(_PORT_REGISTRY_FILE, "w") as f:
        json.dump(registry, f)


def _cleanup_stale_allocations(registry: dict, max_age: float = 900.0) -> dict:
    """Remove port allocations older than max_age seconds."""
    current_time = time.time()
    return {
        str(port): timestamp
        for port, timestamp in registry.items()
        if current_time - timestamp < max_age
    }


def allocate_free_ports(count: int, start_port: int) -> list[int]:
    """Find and return available ports in i16 range with flock-based locking.

    Uses file locking (flock) to prevent race conditions when multiple test processes
    allocate ports simultaneously.

    Port range is limited to i16 (30000-32767) due to Rust backend expecting i16.

    Args:
        count: Number of unique ports to allocate
        start_port: Starting port number for allocation (required)

    Returns:
        list[int]: List of available port numbers between start_port and 32767
    """
    # Validate start_port is in valid i16 range
    if start_port < 1024 or start_port > _PORT_MAX:
        raise ValueError(
            f"start_port must be between 1024 and {_PORT_MAX}, got {start_port}"
        )

    # Ensure lock file exists and is writable
    _PORT_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    _PORT_LOCK_FILE.touch(exist_ok=True)

    if not os.access(_PORT_LOCK_FILE, os.W_OK):
        raise PermissionError(
            f"Port allocation lock file is not writable: {_PORT_LOCK_FILE}"
        )

    with open(_PORT_LOCK_FILE, "r+") as lock_file:
        # Acquire exclusive lock
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

        try:
            # Load registry and clean up stale allocations
            registry = _load_port_registry()
            registry = _cleanup_stale_allocations(registry)

            allocated_ports = set(int(p) for p in registry.keys())
            ports: list[int] = []

            # Start searching from start_port
            current_port = start_port

            # Try to find free ports
            attempts = 0
            max_attempts = count * 50

            while len(ports) < count and attempts < max_attempts:
                attempts += 1

                # Try sequential allocation from our range
                port = current_port
                current_port += 1
                if current_port > _PORT_MAX:
                    current_port = start_port

                # Skip if already allocated or in our current list
                if port in allocated_ports or port in ports:
                    continue

                # Try to bind to verify it's actually free
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.bind(("", port))
                    sock.close()
                    ports.append(port)
                    registry[str(port)] = time.time()
                except OSError:
                    continue

            if len(ports) < count:
                raise RuntimeError(
                    f"Could not find {count} available ports in range {start_port}-{_PORT_MAX}"
                )

            # Save updated registry
            _save_port_registry(registry)

            return ports

        finally:
            # Release lock
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def allocate_free_port(start_port: int) -> int:
    """Find and return a single available port in i16 range.

    Args:
        start_port: Starting port number for allocation (required)

    Returns:
        int: An available port number between start_port and 32767 (i16 max)
    """
    return allocate_free_ports(1, start_port)[0]


def free_ports(ports: list[int]) -> None:
    """Release previously allocated ports back to the pool.

    Args:
        ports: List of port numbers to release
    """
    if not ports:
        return

    # Ensure lock file exists
    _PORT_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    _PORT_LOCK_FILE.touch(exist_ok=True)

    with open(_PORT_LOCK_FILE, "r+") as lock_file:
        # Acquire exclusive lock
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

        try:
            # Load registry
            registry = _load_port_registry()

            # Remove the specified ports
            for port in ports:
                registry.pop(str(port), None)

            # Save updated registry
            _save_port_registry(registry)

        finally:
            # Release lock
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def free_port(port: int) -> None:
    """Release a previously allocated port back to the pool.

    Args:
        port: Port number to release
    """
    free_ports([port])
