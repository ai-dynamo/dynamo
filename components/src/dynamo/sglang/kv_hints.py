# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Forward router KV hints and advertise SGLang KVCR sources to Dynamo."""

from __future__ import annotations

import ipaddress
import json
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from dynamo.common.constants import (
    KV_HINT_TRANSFER_CAPABILITY_KEY,
    KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY,
    KV_HINT_TRANSFER_WORKER_TYPE_RUNTIME_KEY,
)
from dynamo.common.kv_state_agent import validate_advertise_host
from dynamo.sglang._compat import (
    filter_supported_async_generate_kwargs,
    resolved_server_args,
)
from dynamo.sglang.capacity import model_card_dp_rank_bounds


def kv_hint_kwargs(engine: Any, request: Mapping[str, Any]) -> dict[str, Any]:
    """Pass the private router hint only when the engine accepts ``kv_hints``."""
    hint = request.get("kv_hint")
    if not isinstance(hint, Mapping):
        return {}
    return filter_supported_async_generate_kwargs(engine, {"kv_hints": dict(hint)})


def is_supported_kvcr_linker_topology(server_args) -> bool:
    return (
        getattr(server_args, "enable_unified_cache_external_linker", False)
        and getattr(server_args, "unified_cache_external_linker_backend", None)
        == "kvcr"
        and all(
            getattr(server_args, name, 1) in (None, 1)
            for name in (
                "nnodes",
                "tp_size",
                "dp_size",
                "attn_dp_size",
                "attn_cp_size",
                "dcp_size",
                "pp_size",
                "tokenizer_worker_num",
            )
        )
    )


def kvcr_hint_endpoints(
    scheduler_infos, rank_bounds, *, require=None
) -> dict[str, str]:
    """Validate SGLang-owned coordinates; never infer backend defaults or offsets."""
    endpoints: dict[str, str] = {}
    expected_ranks = set(range(*rank_bounds))
    for info in scheduler_infos:
        sources = info.get("kv_hint_sources", [])
        if not isinstance(sources, list):
            raise ValueError("SGLang KV hint sources must be a list")
        for source in sources:
            if not isinstance(source, Mapping):
                raise ValueError("SGLang KV hint source must be an object")
            capabilities = source.get("capabilities", [])
            if not isinstance(capabilities, list):
                raise ValueError("SGLang KV hint capabilities must be a list")
            if "router_hint" not in capabilities:
                continue
            if require == "inventory" and "kvcr_inventory" not in capabilities:
                continue
            if require == "guard" and source.get("inventory_lifetime") != "guard":
                continue
            rank = source.get("dp_rank")
            endpoint = source.get("source_control_endpoint")
            if type(rank) is not int or rank not in expected_ranks:
                raise ValueError("SGLang KV hint source has an invalid DP rank")
            if not isinstance(endpoint, str):
                raise ValueError("SGLang KV hint source requires a TCP endpoint")
            parsed = urlsplit(endpoint)
            if (
                parsed.scheme != "tcp"
                or not parsed.hostname
                or parsed.port is None
                or not 0 < parsed.port <= 65535
                or parsed.username is not None
                or parsed.password is not None
                or parsed.path
                or parsed.query
                or parsed.fragment
                or any(char.isspace() for char in parsed.hostname)
            ):
                raise ValueError("SGLang KV hint source has an invalid TCP endpoint")
            try:
                address = ipaddress.ip_address(parsed.hostname)
            except ValueError:
                if parsed.hostname == "*":
                    raise ValueError(
                        "SGLang KV hint source cannot advertise a wildcard"
                    )
            else:
                if address.is_unspecified:
                    raise ValueError(
                        "SGLang KV hint source cannot advertise a wildcard"
                    )
            if require:
                host = parsed.hostname
                try:
                    validate_advertise_host(f"[{host}]" if ":" in host else host)
                except ValueError as exc:
                    raise ValueError(
                        "KVCR advertise_host in --hicache-storage-backend-extra-config "
                        "must be a non-loopback connectable host"
                    ) from exc
            key = str(rank)
            if key in endpoints and endpoints[key] != endpoint:
                raise ValueError(
                    "SGLang reported conflicting KV hint sources for a rank"
                )
            endpoints[key] = endpoint
    if endpoints and set(map(int, endpoints)) != expected_ranks:
        raise ValueError("SGLang KV hint sources must cover every managed DP rank")
    return endpoints


def publish_kvcr_hint_capabilities(
    runtime_config, engine, server_args, worker_type
) -> None:
    """Publish KVCR sources explicitly reported by a supported SGLang worker."""
    server_args = resolved_server_args(server_args)
    linker = is_supported_kvcr_linker_topology(server_args)
    if (
        not (getattr(server_args, "enable_hierarchical_cache", False) or linker)
        or getattr(server_args, "nnodes", 1) != 1
        or getattr(server_args, "pp_size", 1) != 1
        or not filter_supported_async_generate_kwargs(engine, {"kv_hints": None})
    ):
        return
    init_result = getattr(engine, "_scheduler_init_result", None)
    if init_result is None:
        return
    bounds = model_card_dp_rank_bounds(server_args)
    endpoints = {}
    for require in ("inventory", "guard") if linker else (None,):
        endpoints = kvcr_hint_endpoints(
            init_result.scheduler_infos, bounds, require=require
        )
        if endpoints:
            break
    if not endpoints:
        return
    for key, value in (
        (KV_HINT_TRANSFER_WORKER_TYPE_RUNTIME_KEY, str(worker_type)),
        (KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY, endpoints),
        (KV_HINT_TRANSFER_CAPABILITY_KEY, True),
    ):
        runtime_config.set_engine_specific(key, json.dumps(value))
