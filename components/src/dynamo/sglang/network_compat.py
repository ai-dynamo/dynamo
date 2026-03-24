# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backwards-compatible SGLang network utilities.

Supports both the old API (sglang.srt.utils, SGLang <= 0.5.9) and the new API
(sglang.srt.utils.network, SGLang > 0.5.9) with a unified interface.
"""

import logging
import socket
from typing import Tuple
from urllib.parse import urlparse

# --- Compat imports: try new API first, fall back to old ---

try:
    from sglang.srt.utils.network import (  # noqa: F401
        NetworkAddress,
        get_local_ip_auto,
        get_zmq_socket,
    )

    HAS_NETWORK_ADDRESS = True
except ImportError:
    from sglang.srt.utils import (  # type: ignore[no-redef]  # noqa: F401
        get_local_ip_auto,
        get_zmq_socket,
    )

    NetworkAddress = None  # type: ignore[assignment,misc]
    HAS_NETWORK_ADDRESS = False


def wrap_ipv6(host: str) -> str:
    """Wrap an IPv6 address in brackets for use in host:port strings.

    IPv4 addresses and already-bracketed IPv6 addresses are returned unchanged.
    """
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def format_zmq_endpoint(endpoint_template: str, ip_address: str) -> str:
    """Format ZMQ endpoint by replacing wildcard with IP address.

    Properly handles IPv6 addresses by wrapping them in square brackets.

    Args:
        endpoint_template: ZMQ endpoint template with wildcard (e.g., "tcp://*:5557")
        ip_address: IP address to use (can be IPv4 or IPv6)

    Returns:
        Formatted ZMQ endpoint string

    Example:
        >>> format_zmq_endpoint("tcp://*:5557", "192.168.1.1")
        'tcp://192.168.1.1:5557'
        >>> format_zmq_endpoint("tcp://*:5557", "2a02:6b8:c46:2b4:0:74c1:75b0:0")
        'tcp://[2a02:6b8:c46:2b4:0:74c1:75b0:0]:5557'
    """
    if HAS_NETWORK_ADDRESS:
        parsed = urlparse(endpoint_template)
        if parsed.scheme != "tcp" or parsed.port is None:
            raise ValueError(
                f"Expected tcp://host:port endpoint, got {endpoint_template!r}"
            )
        return NetworkAddress(ip_address, parsed.port).to_tcp()

    # Legacy path: manual bracket-wrap + string replace
    formatted_ip = wrap_ipv6(ip_address)
    return endpoint_template.replace("*", formatted_ip)


def _parse_host_from_addr(addr: str) -> str:
    """Extract the host portion from a dist_init_addr string.

    Handles formats: "host:port", "[IPv6]:port", "[IPv6]", bare IPv6, FQDN.
    """
    addr = addr.strip()
    if addr.startswith("["):
        end = addr.find("]")
        return addr[1:end] if end != -1 else addr.strip("[]")
    # Only treat single ':' with numeric suffix as host:port
    if addr.count(":") == 1:
        host_candidate, maybe_port = addr.rsplit(":", 1)
        return host_candidate if maybe_port.isdigit() else addr
    return addr


def _resolve_host(host: str) -> Tuple[str, bool]:
    """Resolve a hostname via DNS, returning (resolved_ip, is_ipv6).

    Falls back to the original host string on resolution failure.
    """
    try:
        infos = socket.getaddrinfo(
            host,
            None,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
        )
        resolved = infos[0][4][0]
        is_ipv6 = infos[0][0] == socket.AF_INET6
        return resolved, is_ipv6
    except socket.gaierror as e:
        logging.warning(f"Failed to resolve host '{host}': {e}, using as-is")
        is_ipv6 = ":" in host
        return host, is_ipv6


def resolve_bootstrap_host(
    server_args, bootstrap_port: int, *, log: bool = True
) -> Tuple[str, int]:
    """Resolve bootstrap host and port from SGLang server args.

    Uses NetworkAddress on SGLang > 0.5.9, falls back to manual resolution
    with socket.getaddrinfo on older versions. Always preserves a DNS failure
    fallback (use the literal host as-is).

    Args:
        server_args: SGLang ServerArgs with dist_init_addr and
            disaggregation_bootstrap_port attributes.
        bootstrap_port: The bootstrap port to use.
        log: Whether to emit info/warning logs during resolution.

    Returns:
        Tuple of (bootstrap_host, bootstrap_port) where bootstrap_host is
        bracket-wrapped for IPv6.
    """
    if server_args.dist_init_addr:
        if HAS_NETWORK_ADDRESS:
            dist_init = NetworkAddress.parse(server_args.dist_init_addr)
            try:
                resolved = dist_init.resolved()
                bootstrap_host = wrap_ipv6(resolved.host)
                is_ipv6 = resolved.is_ipv6
            except Exception as e:
                # Fallback: use the parsed host as-is
                bootstrap_host = wrap_ipv6(dist_init.host)
                is_ipv6 = ":" in dist_init.host
                if log:
                    logging.warning(
                        f"Failed to resolve bootstrap host "
                        f"'{dist_init.host}': {e}, using as-is"
                    )
            else:
                if log:
                    logging.info(
                        f"Resolved bootstrap host '{dist_init.host}' -> '{bootstrap_host}' "
                        f"({'IPv6' if is_ipv6 else 'IPv4'})"
                    )
        else:
            # Legacy path: manual parsing + socket.getaddrinfo
            host_core = _parse_host_from_addr(server_args.dist_init_addr)
            resolved_ip, is_ipv6 = _resolve_host(host_core)
            bootstrap_host = wrap_ipv6(resolved_ip)
            if log:
                logging.info(
                    f"Resolved bootstrap host '{host_core}' -> '{bootstrap_host}' "
                    f"({'IPv6' if is_ipv6 else 'IPv4'})"
                )
    else:
        local_ip = get_local_ip_auto()
        is_ipv6 = ":" in local_ip
        bootstrap_host = wrap_ipv6(local_ip)
        if log:
            logging.info(
                f"Using auto-detected local IP: {bootstrap_host} "
                f"({'IPv6' if is_ipv6 else 'IPv4'})"
            )

    return bootstrap_host, bootstrap_port
