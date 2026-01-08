# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Port allocation utilities for tests.

Port allocation with flock-based locking to prevent race conditions in parallel tests.
"""

import fcntl
import inspect
import json
import os
import random
import socket
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

# Port allocation lock file
_PORT_LOCK_FILE = Path(tempfile.gettempdir()) / "pytest_port_allocations.lock"
_PORT_REGISTRY_FILE = Path(tempfile.gettempdir()) / "pytest_port_allocations.json"

# Port range for allocation (i16 range for Rust compatibility)
# TODO: Get Rust backend to use u16 instead of i16 so we can use full 1024-65535 range
_PORT_MIN = 1024
_PORT_MAX = 32767


@dataclass(frozen=True)
class ServicePorts:
    """Port allocation for Dynamo service deployments.

    Used by tests that need to pass a cohesive set of ports around (frontend + one or
    more worker/system ports).
    """

    frontend_port: int
    system_ports: list[int]


def _load_port_registry() -> dict:
    """Load the port registry from disk.

    Returns:
        dict: Port registry mapping port numbers (as strings) to allocation info.
              Example: {
                  "30001": {
                      "timestamp": 1732647123.456,
                      "caller_file": "/workspace/tests/test_foo.py",
                      "caller_function": "test_bar",
                      "caller_line": 42
                  }
              }
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
    cleaned = {}
    for port, info in registry.items():
        # Handle both old format (timestamp only) and new format (dict with timestamp)
        if isinstance(info, dict):
            timestamp = info.get("timestamp", 0)
        else:
            timestamp = info

        if current_time - timestamp < max_age:
            cleaned[str(port)] = info

    return cleaned


def allocate_ports(count: int, start_port: int) -> list[int]:
    """Find and return available ports in i16 range with flock-based locking.

    Uses file locking (flock) to prevent race conditions when multiple test processes
    allocate ports simultaneously.

    Port range is limited to i16 (1024-32767) due to Rust backend expecting i16.

    Searches from a random offset (start_port + random(100)) and walks up incrementally.
    Wraps around to _PORT_MIN (1024) when exceeding _PORT_MAX. Retries up to 100 times.

    Args:
        count: Number of unique ports to allocate
        start_port: Starting port number for allocation (required)

    Returns:
        list[int]: List of available port numbers
    """
    # Get caller information for debugging
    caller_file = "unknown"
    caller_function = "unknown"
    caller_line = 0

    frame = inspect.currentframe()
    if frame and frame.f_back:
        caller_frame = frame.f_back
        caller_info = inspect.getframeinfo(caller_frame)
        caller_function = caller_frame.f_code.co_name
        caller_file = caller_info.filename
        caller_line = caller_info.lineno

    # Validate start_port is in valid i16 range. Note that <1024 is reserved for system services (root only)
    if start_port < _PORT_MIN or start_port > _PORT_MAX:
        raise ValueError(
            f"start_port must be between {_PORT_MIN} and {_PORT_MAX}, got {start_port}"
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

            # Start searching from desired port + random offset
            current_port = start_port + random.randint(0, 100)
            if current_port > _PORT_MAX:
                current_port = _PORT_MIN + (current_port - _PORT_MAX - 1)

            # Retry limit
            max_retries = 100
            attempts = 0

            while len(ports) < count and attempts < max_retries:
                attempts += 1

                # Try current port
                port = current_port

                # Increment and wrap around to _PORT_MIN
                current_port += 1
                if current_port > _PORT_MAX:
                    current_port = _PORT_MIN

                # Skip if already allocated or in our current list
                if port in allocated_ports or port in ports:
                    continue

                # Try to bind to verify it's actually free
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.bind(("", port))
                    sock.close()
                    ports.append(port)
                    registry[str(port)] = {
                        "timestamp": time.time(),
                        "caller_file": caller_file,
                        "caller_function": caller_function,
                        "caller_line": caller_line,
                    }
                except OSError:
                    continue

            if len(ports) < count:
                raise RuntimeError(
                    f"Could not find {count} available ports after {max_retries} retries"
                )

            # Save updated registry
            _save_port_registry(registry)

            return ports

        finally:
            # Release lock
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def allocate_port(start_port: int) -> int:
    """Find and return a single available port in i16 range.

    Args:
        start_port: Starting port number for allocation (required)

    Returns:
        int: An available port number between start_port and 32767 (i16 max)
    """
    return allocate_ports(1, start_port)[0]


def deallocate_ports(ports: list[int]) -> None:
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


def deallocate_port(port: int) -> None:
    """Release a previously allocated port back to the pool.

    Args:
        port: Port number to release
    """
    deallocate_ports([port])
