# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import Any

from dynamo.common.constants import (
    ROUTER_HINT_RUNTIME_CAPABILITY_KEY,
    ROUTER_HINT_SOURCE_CONTROL_ENDPOINT_RUNTIME_KEY,
)


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


def _router_hint_source_control_endpoint(tier: Any) -> str | None:
    try:
        control_port = int(_get(tier, "control_port"))
    except (TypeError, ValueError):
        return None
    if control_port <= 0:
        return None
    host = _get(tier, "control_advertise_host") or _get(tier, "control_host")
    if not isinstance(host, str) or not host or host in {"0.0.0.0", "::"}:
        return None
    return f"tcp://{host}:{control_port}"


def enable_router_hint_support(runtime_config: Any, engine_args: Any) -> None:
    router_hint_tiers = _router_hint_tiers(engine_args)
    if not router_hint_tiers:
        return
    if len(router_hint_tiers) > 1:
        raise ValueError(
            "router_hint support requires exactly one router-hint-capable "
            "secondary tier; found multiple tiers advertising router_hint"
        )

    control_endpoint = _router_hint_source_control_endpoint(router_hint_tiers[0])
    if control_endpoint is None:
        raise ValueError(
            "router_hint support requires an advertisable source control endpoint; "
            "set control_advertise_host and a positive control_port on the "
            "router_hint secondary tier"
        )

    runtime_config.set_engine_specific(ROUTER_HINT_RUNTIME_CAPABILITY_KEY, "true")
    runtime_config.set_engine_specific(
        ROUTER_HINT_SOURCE_CONTROL_ENDPOINT_RUNTIME_KEY,
        json.dumps(control_endpoint),
    )
