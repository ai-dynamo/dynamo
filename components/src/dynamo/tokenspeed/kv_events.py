# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Describe TokenSpeed's native KV event stream to the common Worker."""

from __future__ import annotations

import json
from typing import Any

from dynamo.common.backend.publisher import ZmqSource
from dynamo.tokenspeed.disagg import attention_dp_size


def kv_events_config_dict(raw: Any) -> dict[str, Any]:
    if raw is None or raw == "":
        return {}
    config = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(config, dict):
        raise ValueError("TokenSpeed --kv-events-config must be a JSON object")
    return dict(config)


def kv_events_enabled(config: dict[str, Any]) -> bool:
    enabled = config.get("enable_kv_cache_events", False)
    if not isinstance(enabled, bool):
        raise ValueError("TokenSpeed enable_kv_cache_events must be a JSON boolean")
    return enabled and config.get("publisher", "zmq") != "null"


def validate_kv_events(server_args: Any, kv_block_size: int | None) -> None:
    """Reject, at startup on every node, a KV-event setup the worker cannot
    serve."""
    config = kv_events_config_dict(getattr(server_args, "kv_events_config", None))
    if not kv_events_enabled(config):
        return
    if not server_args.enable_prefix_caching:
        raise ValueError(
            "TokenSpeed KV events need prefix caching; remove "
            "--disable-prefix-caching or disable KV events"
        )
    if attention_dp_size(server_args) != 1:
        # Each rank publishes its own KV events (on its own node); the worker
        # subscribes to one publisher and tags no DP rank.
        raise ValueError(
            "TokenSpeed KV events currently require attention DP=1; with "
            "attention DP > 1, route without --router-kv-events"
        )
    if kv_block_size is None:
        raise ValueError(
            "TokenSpeed KV events require --kv-block-size, the engine's prefix "
            "granularity (TokenSpeed logs prefix_granularity=N at startup)"
        )


def kv_event_source(config: dict[str, Any]) -> ZmqSource:
    if config.get("replay_endpoint") is not None:
        raise ValueError(
            "Dynamo TokenSpeed KV events do not support replay_endpoint; "
            "remove it from --kv-events-config"
        )
    if (config.get("publisher") or "zmq") != "zmq":
        raise ValueError("Dynamo TokenSpeed KV events require the zmq publisher")
    endpoint = config["endpoint"]
    if not isinstance(endpoint, str):
        raise ValueError("TokenSpeed KV event endpoint must be a string")
    # TokenSpeed binds wildcard TCP endpoints and IPC endpoints. Other TCP
    # addresses make its publisher connect, which cannot pair with Worker's SUB.
    if endpoint.startswith("tcp://*:"):
        endpoint = endpoint.replace("tcp://*:", "tcp://127.0.0.1:", 1)
    elif endpoint.startswith("tcp://[::]:"):
        endpoint = endpoint.replace("tcp://[::]:", "tcp://[::1]:", 1)
    elif not endpoint.startswith("ipc://"):
        raise ValueError(
            "TokenSpeed KV events require a binding endpoint: ipc://... or tcp://*:PORT"
        )
    return ZmqSource(endpoint=endpoint, topic=config.get("topic", ""), dp_rank=0)
