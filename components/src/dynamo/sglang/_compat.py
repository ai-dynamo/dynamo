# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Compatibility shim for SGLang internal APIs.

SGLang is pre-1.0 and routinely moves, renames, or introduces APIs between
releases. This module is the single place where we handle those differences
so the rest of the component can import from here without version-specific
try/except blocks.

Policy: support current SGLang release + 1 version back (N and N-1). Each
fallback branch must document which version it covers and when it can be
removed. When the old version falls outside the support window, delete the
fallback and any associated polyfills.
"""

import ipaddress
import logging
import socket
from typing import Any

logger = logging.getLogger(__name__)

try:
    from sglang.srt.utils.network import (  # noqa: F401
        NetworkAddress,
        get_local_ip_auto,
        get_zmq_socket,
    )

    _SGLANG_HAS_NETWORK_MODULE = True
except ImportError:
    # Fallback for sglang <= 0.5.9. Remove when min supported version is 0.6.0+
    from sglang.srt.utils import (  # type: ignore[no-redef]  # noqa: F401
        get_local_ip_auto,
        get_zmq_socket,
    )

    _SGLANG_HAS_NETWORK_MODULE = False
    logger.info(
        "sglang.srt.utils.network not found (sglang <= 0.5.9); "
        "using compatibility shim for NetworkAddress"
    )

    class NetworkAddress:  # type: ignore[no-redef]
        """Minimal polyfill for sglang.srt.utils.network.NetworkAddress."""

        def __init__(self, host: str, port: int) -> None:
            self.host = host
            self.port = port

        @property
        def is_ipv6(self) -> bool:
            try:
                ipaddress.IPv6Address(self.host)
                return True
            except ValueError:
                return False

        @classmethod
        def parse(cls, addr: str) -> "NetworkAddress":
            """Parse 'host:port', '[IPv6]:port', or bare host."""
            addr = addr.strip()
            if addr.startswith("["):
                end = addr.find("]")
                host = addr[1:end] if end != -1 else addr.strip("[]")
                rest = addr[end + 1 :] if end != -1 else ""
                if rest.startswith(":") and rest[1:].isdigit():
                    return cls(host, int(rest[1:]))
                return cls(host, 0)
            if addr.count(":") == 1:
                host_part, port_part = addr.rsplit(":", 1)
                if port_part.isdigit():
                    return cls(host_part, int(port_part))
            return cls(addr, 0)

        def resolved(self) -> "NetworkAddress":
            """DNS-resolve the host, preserving port."""
            try:
                infos = socket.getaddrinfo(
                    self.host, None, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM
                )
                resolved_ip = infos[0][4][0]
                return NetworkAddress(resolved_ip, self.port)
            except socket.gaierror:
                return self

        def to_host_port_str(self) -> str:
            """Return '[IPv6]:port' or 'host:port'."""
            if self.is_ipv6:
                return f"[{self.host}]:{self.port}"
            return f"{self.host}:{self.port}"

        def to_tcp(self) -> str:
            """Return 'tcp://[IPv6]:port' or 'tcp://host:port'."""
            if self.is_ipv6:
                return f"tcp://[{self.host}]:{self.port}"
            return f"tcp://{self.host}:{self.port}"


def enable_disjoint_streaming_output(server_args: Any) -> None:
    """
    Enable SGLang's disjoint streaming output across ServerArgs field renames.

    Covers sglang <= 0.5.x (`stream_output`) and newer releases
    (`incremental_streaming_output`).
    """
    fields = getattr(type(server_args), "__dataclass_fields__", None)
    if isinstance(fields, dict):
        if "incremental_streaming_output" in fields:
            server_args.incremental_streaming_output = True
            return
        if "stream_output" in fields:
            server_args.stream_output = True
            return
        raise AttributeError(
            "SGLang ServerArgs has neither 'incremental_streaming_output' nor "
            "'stream_output'"
        )

    if hasattr(server_args, "incremental_streaming_output"):
        server_args.incremental_streaming_output = True
        return
    if hasattr(server_args, "stream_output"):
        server_args.stream_output = True
        return

    logger.debug(
        "Skipping streaming output compatibility for non-ServerArgs object: %s",
        type(server_args).__name__,
    )


__all__ = [
    "NetworkAddress",
    "enable_disjoint_streaming_output",
    "get_local_ip_auto",
    "get_zmq_socket",
    "_SGLANG_HAS_NETWORK_MODULE",
]
