# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TokenSpeed's Mooncake bootstrap protocol for Dynamo P/D workers."""

from __future__ import annotations

import ipaddress
import os
import secrets
import socket
from collections.abc import Mapping
from typing import Any

from dynamo.common.constants import DisaggregationMode
from dynamo.llm.exceptions import InvalidArgument

BOOTSTRAP_HOST_ENV = "DYN_TOKENSPEED_BOOTSTRAP_HOST"


def resolve_disaggregation_mode(server_args: Any) -> DisaggregationMode:
    value = getattr(server_args, "disaggregation_mode", None)
    if value in (None, "null", "aggregated"):
        return DisaggregationMode.AGGREGATED
    if value not in ("prefill", "decode"):
        raise ValueError(f"Unsupported TokenSpeed disaggregation mode: {value!r}")
    return DisaggregationMode(value)


def attention_dp_size(server_args: Any) -> int:
    mapping = getattr(server_args, "mapping", None)
    attention = getattr(mapping, "attn", None)
    return int(
        getattr(attention, "dp_size", None)
        or getattr(server_args, "data_parallel_size", None)
        or getattr(server_args, "dp_size", None)
        or 1
    )


def validate_disagg_compatibility(mode: DisaggregationMode, server_args: Any) -> None:
    if mode == DisaggregationMode.AGGREGATED:
        return
    if (
        getattr(server_args, "disaggregation_transfer_backend", "mooncake")
        != "mooncake"
    ):
        raise ValueError("Dynamo TokenSpeed disaggregation requires Mooncake")


def runtime_disaggregated_endpoint(server_args: Any) -> tuple[str, int]:
    """Advertise a peer-reachable host; the environment opts into single-host use."""
    port = getattr(server_args, "disaggregation_bootstrap_port", None)
    if port is None or not 0 < int(port) < 65536:
        raise ValueError("TokenSpeed prefill requires a valid bootstrap port")
    explicit_host = os.environ.get(BOOTSTRAP_HOST_ENV)
    host = explicit_host or getattr(server_args, "host", None)
    try:
        address = ipaddress.ip_address(host.strip("[]")) if host else None
    except ValueError:
        address = None  # Hostnames are valid advertised addresses too.
    if (
        not host
        or (address is not None and address.is_unspecified)
        or (
            not explicit_host
            and ((address is not None and address.is_loopback) or host == "localhost")
        )
    ):
        # ServerArgs.host defaults to loopback. Mooncake's bootstrap server
        # binds all interfaces, so derive the advertisement independently.
        try:
            host = socket.gethostbyname(socket.gethostname())
        except socket.gaierror as exc:
            raise ValueError(
                f"Cannot determine TokenSpeed bootstrap host; set {BOOTSTRAP_HOST_ENV} "
                "to an address reachable by decode workers"
            ) from exc
        address = ipaddress.ip_address(host)
        if address.is_loopback or address.is_unspecified:
            raise ValueError(
                "TokenSpeed bootstrap host resolves to a local-only address; "
                f"set {BOOTSTRAP_HOST_ENV} to an address reachable by decode workers"
            )
    return str(host), int(port)


def bootstrap_kwargs(
    request: Mapping[str, Any],
    mode: DisaggregationMode,
    prefill_endpoint: tuple[str, int] | None = None,
    dp_rank: int | None = None,
    dp_size: int = 1,
) -> dict[str, Any]:
    """Validate the handoff, creating a room for synchronous prefill if needed."""
    if mode == DisaggregationMode.AGGREGATED:
        return {}
    info = request.get("bootstrap_info")
    if info is None and mode == DisaggregationMode.PREFILL and prefill_endpoint:
        prefill_host, prefill_port = prefill_endpoint
        info = {
            "bootstrap_host": prefill_host,
            "bootstrap_port": prefill_port,
            "bootstrap_room": bootstrap_room(dp_rank, dp_size),
        }
    if not isinstance(info, dict):
        raise InvalidArgument(
            f"TokenSpeed {mode.value} worker requires bootstrap_info from "
            "Dynamo's prefill router"
        )
    host, port, room = (
        info.get(f"bootstrap_{key}") for key in ("host", "port", "room")
    )
    if (
        not isinstance(host, str)
        or not host
        or isinstance(port, bool)
        or not isinstance(port, int)
        or not 0 < port < 65536
        or isinstance(room, bool)
        or not isinstance(room, int)
        or not 0 <= room < 2**64
    ):
        raise InvalidArgument("Invalid TokenSpeed bootstrap host, port, or room")
    return {"bootstrap_host": host, "bootstrap_port": port, "bootstrap_room": room}


def bootstrap_room(dp_rank: int | None = None, dp_size: int = 1) -> int:
    """A random room in the router's signed-64-bit range. With a DP rank,
    ``room % dp_size == dp_rank``: TokenSpeed runs a prefill request on that
    residue, and decode finds its KV cache there."""
    max_room = 2**63 - 1
    if dp_rank is None or dp_size <= 1:
        return secrets.randbits(63)
    return secrets.randbelow((max_room - dp_rank) // dp_size + 1) * dp_size + dp_rank
