# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ipaddress
import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.engine.arg_utils import AsyncEngineArgs

from dynamo.common.constants import (
    ROUTER_HINT_RUNTIME_CAPABILITY_KEY,
    ROUTER_HINT_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY,
    ROUTER_HINT_WORKER_TYPE_RUNTIME_KEY,
)
from dynamo.llm import ModelRuntimeConfig, WorkerType


def _secondary_tiers(engine_args: AsyncEngineArgs) -> list[Mapping[str, Any]]:
    """Return mapping-shaped secondary-tier configs from KVTransferConfig."""
    kv_config = getattr(engine_args, "kv_transfer_config", None)
    extra_config = getattr(kv_config, "kv_connector_extra_config", None)
    if not isinstance(extra_config, Mapping):
        return []

    secondary_tiers = extra_config.get("secondary_tiers")
    if not isinstance(secondary_tiers, list):
        return []

    tiers: list[Mapping[str, Any]] = []
    for tier in secondary_tiers:
        if isinstance(tier, Mapping):
            tiers.append(tier)
    return tiers


def _supports_router_hint(tier: Mapping[str, Any]) -> bool:
    """Return whether a secondary tier advertises router_hint capability."""
    capabilities = tier.get("router_capabilities")
    if not isinstance(capabilities, list):
        return False
    return ROUTER_HINT_RUNTIME_CAPABILITY_KEY in capabilities


def _router_hint_tiers(engine_args: AsyncEngineArgs) -> list[Mapping[str, Any]]:
    """Return secondary tiers that opt in to router-hint support."""
    router_hint_tiers: list[Mapping[str, Any]] = []
    for tier in _secondary_tiers(engine_args):
        if _supports_router_hint(tier):
            router_hint_tiers.append(tier)
    return router_hint_tiers


def _router_hint_source_host(host: str | None) -> str | None:
    """Normalize an advertisable host for a tcp:// endpoint."""
    if not host:
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


def _router_hint_source_control_endpoint(
    tier: Mapping[str, Any], port_offset: int = 0
) -> str | None:
    """Build one advertisable source-control endpoint for a tier and DP offset."""
    configured_port = tier.get("control_port")
    if not isinstance(configured_port, (int, str)):
        return None
    try:
        control_port = int(configured_port) + port_offset
    except ValueError:
        return None
    if not 0 < control_port <= 65535:
        return None
    configured_host = tier.get("control_advertise_host") or tier.get("control_host")
    host = _router_hint_source_host(
        configured_host if isinstance(configured_host, str) else None
    )
    if host is None:
        return None
    return f"tcp://{host}:{control_port}"


def _router_hint_source_control_endpoints(
    tier: Mapping[str, Any], dp_range: tuple[int, int]
) -> dict[str, str] | None:
    """Build source-control endpoints keyed by global DP rank."""
    dp_start, dp_size = dp_range
    if dp_start < 0 or dp_size <= 0:
        return None
    endpoints: dict[str, str] = {}
    for local_dp_rank in range(dp_size):
        endpoint = _router_hint_source_control_endpoint(tier, local_dp_rank)
        if endpoint is None:
            return None
        endpoints[str(dp_start + local_dp_rank)] = endpoint
    return endpoints


def _router_hint_worker_type(worker_type: WorkerType) -> str | None:
    """Normalize a Dynamo WorkerType value into router-hint runtime metadata."""
    role = getattr(worker_type, "value", None)
    if not isinstance(role, str):
        role = str(worker_type)
    if role == "agg":
        role = "aggregated"
    if role not in {"aggregated", "prefill", "decode"}:
        return None
    return role


def enable_router_hint_support(
    runtime_config: ModelRuntimeConfig,
    engine_args: AsyncEngineArgs,
    worker_type: WorkerType,
    dp_range: tuple[int, int] = (0, 1),
) -> None:
    """Publish router-hint runtime metadata when vLLM config supports it."""
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

    # set_engine_specific expects JSON text; the PyO3 binding parses each value
    # with serde_json::from_str, so Python strings must be json.dumps'ed first.
    # Publish capability last so partial metadata is not capability-only if this
    # setup is ever observed before model registration completes.
    runtime_config.set_engine_specific(
        ROUTER_HINT_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY, json.dumps(endpoints)
    )
    runtime_config.set_engine_specific(
        ROUTER_HINT_WORKER_TYPE_RUNTIME_KEY, json.dumps(router_hint_worker_type)
    )
    runtime_config.set_engine_specific(
        ROUTER_HINT_RUNTIME_CAPABILITY_KEY, json.dumps(True)
    )
