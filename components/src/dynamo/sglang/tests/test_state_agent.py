# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dynamo.common.constants import DisaggregationMode
from dynamo.sglang import state_agent
from dynamo.sglang.state_agent import StateAgentLifecycle, state_agent_descriptors

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
        server_args=SimpleNamespace(
            enable_hierarchical_cache=True,
            dp_size=1,
            nnodes=1,
            pp_size=1,
            tokenizer_worker_num=1,
        ),
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
        ("missing_router_hint", "router_hint source for every managed DP rank"),
        ("events_disabled", "KV event emission"),
        ("wrong_format", "Dynamo-format"),
        ("empty_topic", "engine incarnation"),
    ],
)
def test_inventory_fails_closed_for_incomplete_startup_metadata(change, match):
    engine, config, source, info = fixture()
    if change == "missing_events":
        info["kv_event_sources"] = []
    elif change == "wildcard":
        info["kv_event_sources"][0]["endpoint"] = "tcp://0.0.0.0:24020"
    elif change == "missing_rank":
        config.server_args.dp_size = 2
    elif change == "missing_router_hint":
        source["capabilities"] = ["kvcr_inventory"]
    elif change == "wrong_format":
        info["kv_event_sources"][0]["format"] = "default"
    elif change == "empty_topic":
        info["kv_event_sources"][0]["topic"] = ""
    else:
        config.dynamo_args.use_kv_events = False
    with pytest.raises(ValueError, match=match):
        state_agent_descriptors(engine, config)


@pytest.mark.parametrize("native_finished_first", [False, True])
def test_cancelled_start_withdraws_registered_owner(monkeypatch, native_finished_first):
    async def exercise():
        engine, config, _, _ = fixture()
        started = asyncio.Event()
        native_result = asyncio.get_running_loop().create_future()

        def start():
            started.set()
            return native_result

        async def close():
            assert native_result.done() and not native_result.cancelled()

        owner = SimpleNamespace(start=start, close=AsyncMock(side_effect=close))
        monkeypatch.setattr(state_agent, "KvStateAttachmentOwner", lambda *args: owner)
        lifecycle = StateAgentLifecycle()
        endpoint = SimpleNamespace(connection_id=lambda: 17)
        task = asyncio.create_task(lifecycle.start(engine, config, endpoint))
        await asyncio.wait_for(started.wait(), timeout=1)
        if native_finished_first:
            native_result.set_result(None)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not native_result.cancelled()
        if not native_finished_first:
            owner.close.assert_not_awaited()
            native_result.set_result(None)
        await asyncio.wait_for(lifecycle.close(), timeout=1)
        owner.close.assert_awaited_once()
        assert lifecycle._owner is None

    asyncio.run(exercise())
