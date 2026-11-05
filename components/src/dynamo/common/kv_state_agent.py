# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attachment helpers shared by KV state-agent workers."""

import asyncio
import hashlib
import ipaddress
import logging
import re
import socket
from collections.abc import Awaitable, Iterable

logger = logging.getLogger(__name__)

CACHE_OWNER_PATTERN = re.compile(
    r"^(?P<domain>[0-9a-f]{32}:[0-9a-f]{32})/[0-9a-f]{16}/[0-9a-f]{32}$"
)
_STATE_AGENT_CLOSE_TIMEOUT_SECS = 30.0

# Fixed width, so no worker's topic is a prefix of another's: SUB subscriptions
# are prefix filters, and the state agent routes pooled raw streams by topic.
_RAW_TOPIC_HEX_DIGITS = 16


async def bounded_close(cleanup: Awaitable[None]) -> None:
    """Bound cleanup without replacing a worker's error or cancellation on timeout."""
    try:
        await asyncio.wait_for(cleanup, timeout=_STATE_AGENT_CLOSE_TIMEOUT_SECS)
    except asyncio.TimeoutError:
        logger.warning(
            "KV state attachment cleanup timed out after %.0fs; continuing worker shutdown",
            _STATE_AGENT_CLOSE_TIMEOUT_SECS,
        )


def validate_advertise_host(host: str) -> None:
    """Require a connectable hostname, IPv4 address, or bracketed IPv6 literal."""
    if host.rstrip(".").lower() in {"*", "localhost", "0.0.0.0", "::", "[::]"}:
        raise ValueError("raw_advertise_host must be a non-loopback connectable host")
    if "://" in host or "/" in host:
        raise ValueError("raw_advertise_host must not include a scheme, port, or path")
    if host.startswith("["):
        if not host.endswith("]"):
            raise ValueError("raw_advertise_host has mismatched IPv6 brackets")
        literal = host[1:-1]
        try:
            address = ipaddress.ip_address(literal)
        except ValueError as error:
            raise ValueError(
                "bracketed raw_advertise_host must be an IPv6 literal"
            ) from error
        if address.version != 6:
            raise ValueError("only IPv6 literals may use brackets")
    else:
        if "[" in host or "]" in host or ":" in host or any(c.isspace() for c in host):
            raise ValueError(
                "raw_advertise_host must be a hostname or bracketed IPv6 literal without a port"
            )
        literal = host
        try:
            address = ipaddress.ip_address(literal)
        except ValueError:
            # POSIX accepts abbreviated/integer IPv4 literals (e.g.127.1).
            # Check them without resolving arbitrary DNS names at startup.
            try:
                address = ipaddress.IPv4Address(socket.inet_aton(literal))
            except OSError:
                return

    if address.version == 6 and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    if address.is_loopback or address.is_unspecified:
        raise ValueError("raw_advertise_host must be a non-loopback connectable host")


def worker_raw_topic(cache_owner_ids: Iterable[str]) -> str:
    """Raw KV event topic of one worker, published by every DP rank it owns.

    Derived from the worker's CacheOwner IDs, so it is distinct per worker and
    stable across restarts.
    """
    digest = hashlib.sha256("\n".join(sorted(cache_owner_ids)).encode()).hexdigest()
    return digest[:_RAW_TOPIC_HEX_DIGITS]
