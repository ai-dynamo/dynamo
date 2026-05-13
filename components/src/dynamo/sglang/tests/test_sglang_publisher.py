# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

import dynamo.sglang.publisher as publisher_mod
from dynamo.sglang.publisher import (
    DynamoSglangPublisher,
    get_local_dp_rank_range,
    resolve_multinode_leader_worker_id,
    set_forward_pass_metrics_worker_id,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]


def test_get_local_dp_rank_range_defaults_to_rank_zero():
    server_args = SimpleNamespace(
        dp_size=1,
        enable_dp_attention=False,
        nnodes=1,
        node_rank=0,
    )

    assert list(get_local_dp_rank_range(server_args)) == [0]


def test_get_local_dp_rank_range_respects_multinode_dp_attention():
    server_args = SimpleNamespace(
        dp_size=8,
        enable_dp_attention=True,
        nnodes=2,
        node_rank=1,
    )

    assert list(get_local_dp_rank_range(server_args)) == [4, 5, 6, 7]


def test_set_forward_pass_metrics_worker_id_uses_endpoint_identity():
    server_args = SimpleNamespace(enable_forward_pass_metrics=True)
    endpoint = SimpleNamespace(connection_id=lambda: "endpoint-9")

    set_forward_pass_metrics_worker_id(server_args, endpoint)

    assert server_args.forward_pass_metrics_worker_id == "endpoint-9"
    assert server_args.forward_pass_metrics_ipc_name.startswith("ipc://")


def test_set_forward_pass_metrics_worker_id_is_noop_when_disabled():
    server_args = SimpleNamespace(enable_forward_pass_metrics=False)
    endpoint = SimpleNamespace(connection_id=lambda: "endpoint-9")

    set_forward_pass_metrics_worker_id(server_args, endpoint)

    assert not hasattr(server_args, "forward_pass_metrics_worker_id")


@pytest.mark.asyncio
async def test_resolve_multinode_leader_worker_id_uses_single_instance():
    class FakeClient:
        async def wait_for_instances(self):
            return [1234]

    class FakeEndpoint:
        async def client(self):
            return FakeClient()

    server_args = SimpleNamespace(nnodes=2, node_rank=1)

    worker_id = await resolve_multinode_leader_worker_id(FakeEndpoint(), server_args)

    assert worker_id == 1234


@pytest.mark.asyncio
async def test_resolve_multinode_leader_worker_id_ignores_ambiguous_instances():
    class FakeClient:
        async def wait_for_instances(self):
            return [1234, 5678]

    class FakeEndpoint:
        async def client(self):
            return FakeClient()

    server_args = SimpleNamespace(nnodes=2, node_rank=1)

    worker_id = await resolve_multinode_leader_worker_id(FakeEndpoint(), server_args)

    assert worker_id is None


def test_init_kv_event_publish_uses_worker_id_override(monkeypatch):
    calls = []

    class FakeKvEventPublisher:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(publisher_mod, "KvEventPublisher", FakeKvEventPublisher)
    monkeypatch.setattr(publisher_mod, "get_local_ip_auto", lambda: "127.0.0.1")
    monkeypatch.setattr(
        publisher_mod,
        "ZmqEventPublisher",
        SimpleNamespace(
            offset_endpoint_port=staticmethod(
                lambda base_ep, dp_rank: f"tcp://*:{5557 + dp_rank}"
            )
        ),
    )
    monkeypatch.setattr(
        publisher_mod,
        "format_zmq_endpoint",
        lambda endpoint, ip_address: endpoint.replace("*", ip_address),
    )

    server_args = SimpleNamespace(
        kv_events_config='{"endpoint": "tcp://*:5557"}',
        page_size=16,
        dp_size=8,
        enable_dp_attention=True,
        nnodes=2,
        node_rank=1,
    )
    config = SimpleNamespace(
        server_args=server_args,
        dynamo_args=SimpleNamespace(enable_local_indexer=True),
    )
    publisher = DynamoSglangPublisher(
        engine=SimpleNamespace(),
        config=config,
        generate_endpoint=SimpleNamespace(),
        component_gauges=SimpleNamespace(),
        kv_worker_id=1234,
    )

    publishers = publisher.init_kv_event_publish()

    assert len(publishers) == 4
    assert [call["dp_rank"] for call in calls] == [4, 5, 6, 7]
    assert {call["worker_id"] for call in calls} == {1234}
