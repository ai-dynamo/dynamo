# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Port allocation utilities for tests.

Simple socket-based port allocation for test isolation.
"""

import socket

# Sequential port counter for fallback allocation
_next_sequential_port = 30000


def get_free_ports(count: int = 1) -> list[int]:
    """Find and return available ports in i16 range.

    Note: There is a small race condition between finding a free port and another
    process binding to it. However, this probability is much smaller than using
    hard-coded ports (8000, 8081, 8082) which always conflict when running tests in parallel.

    Port range is limited to i16 (1024-32767) due to Rust backend expecting i16.
    See TODO in Rust code to change type to u16.

    Args:
        count: Number of unique ports to allocate (default: 1)

    Returns:
        list[int]: List of available port numbers between 1024 and 32767 (i16 max)
    """
    global _next_sequential_port

    ports: list[int] = []
    attempts = 0
    max_attempts = count * 100

    while len(ports) < count and attempts < max_attempts:
        attempts += 1
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("", 0))
        port = sock.getsockname()[1]
        sock.close()
        # Add if port is in i16 range and not already in our list
        if 1024 <= port <= 32767 and port not in ports:
            ports.append(port)

    # Fallback: try sequential allocation if we didn't get enough ports
    if len(ports) < count:
        checked = 0
        while len(ports) < count and checked < 2768:  # 32767 - 30000 + 1
            port = _next_sequential_port
            _next_sequential_port += 1
            # Wrap around to start of range
            if _next_sequential_port > 32767:
                _next_sequential_port = 30000
            checked += 1

            if port in ports:
                continue
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(("", port))
                sock.close()
                ports.append(port)
            except OSError:
                continue

    if len(ports) < count:
        raise RuntimeError(f"Could not find {count} available ports in i16 range")

    return ports


def get_free_port() -> int:
    """Find and return a single available port in i16 range.

    Returns:
        int: An available port number between 1024 and 32767 (i16 max)
    """
    return get_free_ports(1)[0]
