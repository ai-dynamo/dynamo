# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Describe TokenSpeed's native KV event stream to the common Worker."""

from __future__ import annotations

import json
from typing import Any

from dynamo.common.backend.publisher import ZmqSource


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
