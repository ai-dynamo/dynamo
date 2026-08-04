# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ipaddress
import json
from typing import Any

from dynamo.common.constants import (
    ROUTER_HINT_RUNTIME_CAPABILITY_KEY,
    ROUTER_HINT_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY,
    ROUTER_HINT_WORKER_TYPE_RUNTIME_KEY,
)


def _set_engine_specific(runtime_config: Any, key: str, value: Any) -> None:
    """Publish an engine-specific runtime value.

    The PyO3 binding for ``set_engine_specific`` parses its argument with
    ``serde_json::from_str``, so every value must be JSON text. Routing all
    writes through here keeps that contract explicit — a bare ``"prefill"``
    is not valid JSON and would raise at registration.
    """
    runtime_config.set_engine_specific(key, json.dumps(value))


def _get(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _secondary_tiers(engine_args: Any) -> list[Any]:
    kv_config = _get(engine_args, "kv_transfer_config")
    extra_config = _get(kv_config, "kv_connector_extra_config")
    secondary_tiers = _get(extra_config, "secondary_tiers")
    if not isinstance(secondary_tiers, list):
        return []
    return secondary_tiers


def _supports_router_hint(tier: Any) -> bool:
    capabilities = _get(tier, "router_capabilities")
    if not isinstance(capabilities, list):
        return False
    return ROUTER_HINT_RUNTIME_CAPABILITY_KEY in capabilities


def _router_hint_tiers(engine_args: Any) -> list[Any]:
    return [
        tier for tier in _secondary_tiers(engine_args) if _supports_router_hint(tier)
    ]


def _router_hint_source_host(host: Any) -> str | None:
    if not isinstance(host, str) or not host:
        return None
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return host
    if address.is_unspecified:
        return None
    if address.version == 6:
        return f"[{address.compressed}]"
    return address.compressed


def _router_hint_source_control_endpoint(tier: Any, port_offset: int = 0) -> str | None:
    try:
        control_port = int(_get(tier, "control_port")) + port_offset
    except (TypeError, ValueError):
        return None
    if not 0 < control_port <= 65535:
        return None
    host = _router_hint_source_host(
        _get(tier, "control_advertise_host") or _get(tier, "control_host")
    )
    if host is None:
        return None
    return f"tcp://{host}:{control_port}"


def _router_hint_source_control_endpoints(
    tier: Any, dp_range: tuple[int, int]
) -> dict[str, str] | None:
    dp_start, dp_size = dp_range
    endpoints: dict[str, str] = {}
    for local_dp_rank in range(dp_size):
        endpoint = _router_hint_source_control_endpoint(tier, local_dp_rank)
        if endpoint is None:
            return None
        endpoints[str(dp_start + local_dp_rank)] = endpoint
    return endpoints


def _router_hint_worker_type(worker_type: Any) -> str | None:
    role = getattr(worker_type, "value", None)
    if not isinstance(role, str):
        role = str(worker_type)
    if role == "agg":
        role = "aggregated"
    if role not in {"aggregated", "prefill", "decode"}:
        return None
    return role


def enable_router_hint_support(
    runtime_config: Any,
    engine_args: Any,
    worker_type: Any,
    dp_range: tuple[int, int] = (0, 1),
) -> None:
    router_hint_worker_type = _router_hint_worker_type(worker_type)
    if router_hint_worker_type is None:
        return

    router_hint_tiers = _router_hint_tiers(engine_args)
    if not router_hint_tiers:
        return
    if len(router_hint_tiers) > 1:
        raise ValueError(
            "router_hint support requires exactly one router-hint-capable "
            "secondary tier; found multiple tiers advertising router_hint"
        )

    endpoints = _router_hint_source_control_endpoints(router_hint_tiers[0], dp_range)
    if endpoints is None:
        raise ValueError(
            "router_hint support requires advertisable source control endpoints "
            "for all managed DP ranks"
        )

    _set_engine_specific(runtime_config, ROUTER_HINT_RUNTIME_CAPABILITY_KEY, True)
    _set_engine_specific(
        runtime_config, ROUTER_HINT_WORKER_TYPE_RUNTIME_KEY, router_hint_worker_type
    )
    _set_engine_specific(
        runtime_config, ROUTER_HINT_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY, endpoints
    )
