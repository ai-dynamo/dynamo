# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dynamo.vllm import state_agent

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.usefixtures("vllm_cpu_platform_when_no_accelerator"),
]


def _owner(slot: str) -> str:
    return f"{'01' * 16}:{'02' * 16}/{'03' * 8}/{slot * 32}"


def _config(mapping: dict[str, str], host: str = "worker-a.example"):
    return SimpleNamespace(
        namespace="ns",
        component="backend",
        endpoint="generate",
        kv_state_endpoint=None,
        engine_args=SimpleNamespace(
            kv_transfer_config=SimpleNamespace(
                kv_connector_extra_config={
                    "dynamo_state_agent": {
                        "raw_advertise_host": host,
                        "cache_owner_ids": mapping,
                    }
                }
            ),
            kv_events_config=SimpleNamespace(
                endpoint="tcp://*:5557", enable_kv_cache_events=True
            ),
            enable_prefix_caching=True,
        ),
    )


def _vllm_config(start: int, size: int):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_external_lb=False,
            data_parallel_hybrid_lb=True,
            data_parallel_rank=start,
            data_parallel_size_local=size,
        ),
        cache_config=SimpleNamespace(block_size=64),
        additional_config={},
    )


def test_state_agent_config_rejects_ambiguous_identity_and_host_before_start():
    with pytest.raises(ValueError, match="unique"):
        state_agent.state_agent_settings(_config({"4": _owner("a"), "7": _owner("a")}))
    with pytest.raises(ValueError, match="non-loopback"):
        state_agent.state_agent_settings(_config({"4": _owner("a")}, "127.0.0.1"))
    for invalid in (
        "worker.example:1234",
        "2001:db8::1",
        "[2001:db8::1",
        "[127.0.0.2]",
    ):
        with pytest.raises(ValueError):
            state_agent.state_agent_settings(_config({"4": _owner("a")}, invalid))

    assert (
        state_agent.state_agent_settings(_config({"4": _owner("a")}, "[2001:db8::1]"))
        is not None
    )


@pytest.mark.asyncio
async def test_closed_lifecycle_rejects_and_closes_late_owner():
    lifecycle = state_agent.StateAgentLifecycle()
    owner = SimpleNamespace(close=AsyncMock())

    await lifecycle.close()
    with pytest.raises(RuntimeError, match="closed"):
        await lifecycle.install(owner)

    owner.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_closed_lifecycle_bounds_late_owner_cleanup(monkeypatch):
    never_finishes = asyncio.Event()
    lifecycle = state_agent.StateAgentLifecycle()
    owner = SimpleNamespace(close=AsyncMock(side_effect=never_finishes.wait))
    monkeypatch.setattr(state_agent, "_STATE_AGENT_CLOSE_TIMEOUT_SECS", 0.01)

    await lifecycle.close()
    with pytest.raises(RuntimeError, match="closed"):
        await lifecycle.install(owner)

    owner.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_lifecycle_close_has_a_bounded_wait(monkeypatch):
    never_finishes = asyncio.Event()
    lifecycle = state_agent.StateAgentLifecycle()
    owner = SimpleNamespace(close=AsyncMock(side_effect=never_finishes.wait))
    monkeypatch.setattr(state_agent, "_STATE_AGENT_CLOSE_TIMEOUT_SECS", 0.01)
    await lifecycle.install(owner)

    await lifecycle.close()

    owner.close.assert_awaited_once()


def test_cli_validates_state_agent_after_vllm_engine_config():
    from dynamo.vllm.args import parse_args

    transfer = {
        "kv_connector": "OffloadingConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "dynamo_state_agent": {
                "raw_advertise_host": "worker-a.example",
                "cache_owner_ids": {"0": _owner("a")},
            }
        },
    }
    events = {
        "publisher": "zmq",
        "topic": "kv-events",
        "endpoint": "tcp://*:5557",
        "enable_kv_cache_events": True,
    }
    config = parse_args(
        [
            "--model",
            "Qwen/Qwen3-0.6B",
            "--kv-transfer-config",
            json.dumps(transfer),
            "--kv-events-config",
            json.dumps(events),
        ]
    )

    assert state_agent.state_agent_settings(config) is not None


@pytest.mark.asyncio
async def test_attachment_owner_preserves_global_rank_and_resolved_endpoint(
    monkeypatch,
):
    captured = {}

    class FakeOwner:
        def __init__(self, endpoint, worker_id, descriptors):
            captured.update(
                endpoint=endpoint, worker_id=worker_id, descriptors=descriptors
            )

        async def start(self):
            captured["started"] = True

    monkeypatch.setattr(state_agent, "KvStateAttachmentOwner", FakeOwner)
    endpoint = SimpleNamespace(connection_id=lambda: 17)
    config = _config(
        {
            "7": _owner("d"),
            "4": _owner("a"),
            "6": _owner("c"),
            "5": _owner("b"),
        }
    )
    config.disaggregation_mode = state_agent.DisaggregationMode.PREFILL
    config.engine_args.kv_transfer_config.kv_connector_extra_config[
        "secondary_tiers"
    ] = [
        {
            "router_capabilities": ["router_hint"],
            "control_advertise_host": "cache-owner.example",
            "control_ports": ["23284", "23285", "23286", "23287"],
        }
    ]
    owner = await state_agent.start_attachment_owner(
        config,
        endpoint,
        _vllm_config(4, 4),
        image_token_id=99,
    )

    assert owner is not None
    assert captured["worker_id"] == 17
    assert [item["global_dp_rank"] for item in captured["descriptors"]] == [4, 5, 6, 7]
    assert [item["raw_zmq_endpoint"] for item in captured["descriptors"]] == [
        "tcp://worker-a.example:5561",
        "tcp://worker-a.example:5562",
        "tcp://worker-a.example:5563",
        "tcp://worker-a.example:5564",
    ]
    assert [item["router_hint_source"] for item in captured["descriptors"]] == [
        {
            "source_control_endpoint": "tcp://cache-owner.example:23284",
            "worker_type": "prefill",
        },
        {
            "source_control_endpoint": "tcp://cache-owner.example:23285",
            "worker_type": "prefill",
        },
        {
            "source_control_endpoint": "tcp://cache-owner.example:23286",
            "worker_type": "prefill",
        },
        {
            "source_control_endpoint": "tcp://cache-owner.example:23287",
            "worker_type": "prefill",
        },
    ]
    assert captured["started"] is True


@pytest.mark.asyncio
async def test_partial_rank_mapping_is_rejected_before_owner_construction(monkeypatch):
    constructed = False

    class UnexpectedOwner:
        def __init__(self, *_args, **_kwargs):
            nonlocal constructed
            constructed = True

    monkeypatch.setattr(state_agent, "KvStateAttachmentOwner", UnexpectedOwner)
    with pytest.raises(ValueError, match="exactly cover"):
        await state_agent.start_attachment_owner(
            _config({"4": _owner("a")}),
            SimpleNamespace(connection_id=lambda: 17),
            _vllm_config(4, 2),
            image_token_id=None,
        )
    assert constructed is False
