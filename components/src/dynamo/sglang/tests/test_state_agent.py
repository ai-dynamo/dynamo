# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("sglang")

from dynamo.common import kv_state_agent
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
    "target,updates,match",
    [
        ("info", {"kv_event_sources": []}, "raw event source per rank"),
        ("raw", {"endpoint": "tcp://0.0.0.0:24020"}, "non-loopback"),
        ("server_args", {"dp_size": 2}, "every managed DP rank"),
        (
            "source",
            {"capabilities": ["kvcr_inventory"]},
            "router_hint source for every managed DP rank",
        ),
        ("dynamo_args", {"use_kv_events": False}, "KV event emission"),
        ("raw", {"format": "default"}, "Dynamo-format"),
        ("raw", {"topic": ""}, "engine incarnation"),
    ],
)
def test_inventory_fails_closed_for_incomplete_startup_metadata(target, updates, match):
    engine, config, source, info = fixture()
    targets = {
        "info": info,
        "source": source,
        "raw": info["kv_event_sources"][0],
        "server_args": vars(config.server_args),
        "dynamo_args": vars(config.dynamo_args),
    }
    targets[target].update(updates)
    with pytest.raises(ValueError, match=match):
        state_agent_descriptors(engine, config)


@pytest.mark.parametrize(
    "host",
    [
        "localhost",
        "127.0.0.1",
        "127.1",
        "2130706433",
        "0",
        "[::1]",
        "[::ffff:127.0.0.1]",
    ],
)
def test_raw_event_endpoint_rejects_loopback_and_unspecified_hosts(host):
    engine, config, _, info = fixture()
    info["kv_event_sources"][0]["endpoint"] = f"tcp://{host}:24020"
    with pytest.raises(ValueError, match="non-loopback"):
        state_agent_descriptors(engine, config)


def test_raw_event_endpoint_accepts_connectable_ipv6():
    engine, config, _, info = fixture()
    endpoint = "tcp://[2001:db8::1]:24020"
    info["kv_event_sources"][0]["endpoint"] = endpoint
    (descriptor,) = state_agent_descriptors(engine, config)
    assert descriptor["raw_zmq_endpoint"] == endpoint


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
async def test_cleanup_timeout_preserves_body_exception(
    monkeypatch, caplog, error_type
):
    engine, config, _, _ = fixture()
    owner = SimpleNamespace(
        start=AsyncMock(), close=AsyncMock(side_effect=asyncio.Event().wait)
    )
    monkeypatch.setattr(state_agent, "KvStateAttachmentOwner", lambda *args: owner)
    monkeypatch.setattr(kv_state_agent, "_STATE_AGENT_CLOSE_TIMEOUT_SECS", 0.01)
    lifecycle = StateAgentLifecycle()
    await lifecycle.start(engine, config, SimpleNamespace(connection_id=lambda: 17))
    original = error_type("worker body failed")

    with pytest.raises(error_type) as caught:
        try:
            raise original
        finally:
            await lifecycle.close()

    assert caught.value is original
    assert "cleanup timed out" in caplog.text
    await lifecycle.close()
    owner.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancelled_start_cleanup_timeout_preserves_withdrawal(
    monkeypatch, caplog
):
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
    monkeypatch.setattr(kv_state_agent, "_STATE_AGENT_CLOSE_TIMEOUT_SECS", 0.01)
    lifecycle = StateAgentLifecycle()
    task = asyncio.create_task(
        lifecycle.start(engine, config, SimpleNamespace(connection_id=lambda: 17))
    )
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        try:
            await task
        finally:
            await lifecycle.close()

    assert "cleanup timed out" in caplog.text
    assert not native_result.done()
    owner.close.assert_not_awaited()
    native_result.set_result(None)
    await asyncio.wait_for(lifecycle.close(), timeout=1)
    owner.close.assert_awaited_once()
    assert lifecycle._owner is None


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
