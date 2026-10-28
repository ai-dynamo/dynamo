# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from dynamo.common.constants import DisaggregationMode
from dynamo.sglang.state_agent import state_agent_descriptors

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.router,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]


def fixture():
    source = {
        "dp_rank": 0,
        "capabilities": ["router_hint", "kvcr_inventory"],
        "source_control_endpoint": "tcp://worker:24100",
        "cache_owner_id": f"{'01' * 16}:{'02' * 16}/{'00' * 8}/{'03' * 16}",
    }
    info = {
        "kv_hint_sources": [source],
        "kv_event_sources": [
            {
                "dp_rank": 0,
                "endpoint": "tcp://worker:24020",
                "topic": "raw",
                "block_size": 64,
                "format": "dynamo",
            }
        ],
    }
    engine = SimpleNamespace(
        _scheduler_init_result=SimpleNamespace(scheduler_infos=[info])
    )
    config = SimpleNamespace(
        server_args=SimpleNamespace(enable_hierarchical_cache=True, dp_size=1),
        dynamo_args=SimpleNamespace(
            use_kv_events=True,
            kv_state_endpoint=None,
            namespace="ns",
            component="backend",
            endpoint="generate",
        ),
        serving_mode=DisaggregationMode.AGGREGATED,
    )
    return engine, config, source, info


def test_descriptors_use_only_reported_sources():
    engine, config, source, _ = fixture()
    (result,) = state_agent_descriptors(engine, config)
    assert result["cache_owner_id"] == source["cache_owner_id"]
    assert (
        result["router_hint_source"]["source_control_endpoint"] == "tcp://worker:24100"
    )
    assert result["raw_zmq_endpoint"] == "tcp://worker:24020"
    assert result["ingress_protocol"] == "sglang_hicache_v1"
    assert result["kv_state_endpoint"] == "ns/backend/generate"
    source["capabilities"] = ["router_hint"]
    assert state_agent_descriptors(engine, config) == []


@pytest.mark.parametrize(
    "change,match",
    [
        ("missing_events", "raw event source per rank"),
        ("wildcard", "wildcard"),
        ("missing_rank", "every managed DP rank"),
        ("events_disabled", "KV event emission"),
        ("wrong_format", "Dynamo-format"),
        ("empty_topic", "engine incarnation"),
    ],
)
def test_inventory_fails_closed_for_incomplete_startup_metadata(change, match):
    engine, config, _, info = fixture()
    if change == "missing_events":
        info["kv_event_sources"] = []
    elif change == "wildcard":
        info["kv_event_sources"][0]["endpoint"] = "tcp://0.0.0.0:24020"
    elif change == "missing_rank":
        config.server_args.dp_size = 2
    elif change == "wrong_format":
        info["kv_event_sources"][0]["format"] = "default"
    elif change == "empty_topic":
        info["kv_event_sources"][0]["topic"] = ""
    else:
        config.dynamo_args.use_kv_events = False
    with pytest.raises(ValueError, match=match):
        state_agent_descriptors(engine, config)
