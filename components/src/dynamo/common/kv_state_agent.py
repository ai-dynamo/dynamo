# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attachment helpers shared by KV state-agent workers (vLLM and the mocker)."""

import hashlib
import ipaddress
from collections.abc import Iterable

# Fixed width, so no worker's topic is a prefix of another's: SUB subscriptions
# are prefix filters, and the state agent routes pooled raw streams by topic.
_RAW_TOPIC_HEX_DIGITS = 16


def validate_advertise_host(host: str) -> None:
    """Reject a raw advertise host the state agent could not connect to."""
    if host in {"*", "localhost", "0.0.0.0", "::", "[::]"}:
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
            return

    if address.is_loopback or address.is_unspecified:
        raise ValueError("raw_advertise_host must be a non-loopback connectable host")


def worker_raw_topic(cache_owner_ids: Iterable[str]) -> str:
    """Raw KV event topic of one worker, published by every DP rank it owns.

    Derived from the worker's CacheOwner IDs, so it is distinct per worker and
    stable across restarts.
    """
    digest = hashlib.sha256("\n".join(sorted(cache_owner_ids)).encode()).hexdigest()
    return digest[:_RAW_TOPIC_HEX_DIGITS]
