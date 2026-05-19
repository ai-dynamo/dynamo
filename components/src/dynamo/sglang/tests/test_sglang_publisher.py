# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace

import pytest

import dynamo.sglang._disagg as disagg_mod
import dynamo.sglang.publisher as publisher_mod
from dynamo.sglang._disagg import SGLANG_WORKER_GROUP_ID_KEY, get_sglang_worker_group_id
from dynamo.sglang.publisher import (
    DynamoSglangPublisher,
    _resolve_multinode_leader_worker_id,
    get_local_dp_rank_range,
    handle_non_leader_node,
    set_forward_pass_metrics_worker_id,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
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


class FakeNetworkAddress:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

    @staticmethod
    def parse(value: str):
        host, port = value.rsplit(":", 1)
        return FakeNetworkAddress(host, int(port))

    def resolved(self):
        return self

    def to_tcp(self):
        return f"tcp://{self.host}:{self.port}"


def test_sglang_worker_group_id_matches_across_node_ranks(monkeypatch):
    monkeypatch.setattr(disagg_mod, "_network_address_cls", lambda: FakeNetworkAddress)
    node0_args = SimpleNamespace(nnodes=2, node_rank=0, dist_init_addr="10.0.0.1:2345")
    node1_args = SimpleNamespace(nnodes=2, node_rank=1, dist_init_addr="10.0.0.1:2345")

    assert get_sglang_worker_group_id(node0_args) == get_sglang_worker_group_id(
        node1_args
    )


def test_sglang_worker_group_id_differs_by_dist_init_addr(monkeypatch):
    monkeypatch.setattr(disagg_mod, "_network_address_cls", lambda: FakeNetworkAddress)
    group_a = get_sglang_worker_group_id(
        SimpleNamespace(nnodes=2, dist_init_addr="10.0.0.1:2345")
    )
    group_b = get_sglang_worker_group_id(
        SimpleNamespace(nnodes=2, dist_init_addr="10.0.0.2:2345")
    )

    assert group_a != group_b


def test_sglang_worker_group_id_returns_none_without_dist_init_addr():
    server_args = SimpleNamespace(nnodes=2, node_rank=1)

    assert get_sglang_worker_group_id(server_args) is None


def test_sglang_worker_group_id_returns_none_when_unparseable(monkeypatch):
    class BrokenNetworkAddress:
        @staticmethod
        def parse(value: str):
            raise ValueError(f"bad address: {value}")

    monkeypatch.setattr(
        disagg_mod, "_network_address_cls", lambda: BrokenNetworkAddress
    )
    server_args = SimpleNamespace(nnodes=2, dist_init_addr="not an address")

    assert get_sglang_worker_group_id(server_args) is None


@pytest.mark.asyncio
async def test_resolve_multinode_leader_worker_id_uses_single_instance():
    class FakeClient:
        async def wait_for_instances(self):
            return [1234]

    class FakeEndpoint:
        async def client(self):
            return FakeClient()

    server_args = SimpleNamespace(nnodes=2, node_rank=1)

    worker_id = await _resolve_multinode_leader_worker_id(FakeEndpoint(), server_args)

    assert worker_id == 1234


@pytest.mark.asyncio
async def test_resolve_multinode_leader_worker_id_uses_worker_group(monkeypatch):
    calls = []

    class FakeClient:
        async def wait_for_instance_by_runtime_data(self, key, value, timeout_s=None):
            calls.append((key, value, timeout_s))
            return 1234

    class FakeEndpoint:
        async def client(self):
            return FakeClient()

    monkeypatch.setattr(
        publisher_mod,
        "get_sglang_worker_group_id",
        lambda server_args: "dist_init:tcp://10.0.0.1:2345",
    )
    server_args = SimpleNamespace(
        nnodes=2,
        node_rank=1,
        dist_timeout=5,
    )

    worker_id = await _resolve_multinode_leader_worker_id(FakeEndpoint(), server_args)

    assert worker_id == 1234
    assert calls == [
        (
            SGLANG_WORKER_GROUP_ID_KEY,
            "dist_init:tcp://10.0.0.1:2345",
            5.0,
        )
    ]


@pytest.mark.asyncio
async def test_resolve_multinode_leader_worker_id_has_no_default_timeout(monkeypatch):
    calls = []

    class FakeClient:
        async def wait_for_instance_by_runtime_data(self, key, value, timeout_s=None):
            calls.append((key, value, timeout_s))
            return 1234

    class FakeEndpoint:
        async def client(self):
            return FakeClient()

    monkeypatch.setattr(
        publisher_mod,
        "get_sglang_worker_group_id",
        lambda server_args: "dist_init:tcp://10.0.0.1:2345",
    )
    server_args = SimpleNamespace(
        nnodes=2,
        node_rank=1,
    )

    worker_id = await _resolve_multinode_leader_worker_id(FakeEndpoint(), server_args)

    assert worker_id == 1234
    assert calls == [
        (
            SGLANG_WORKER_GROUP_ID_KEY,
            "dist_init:tcp://10.0.0.1:2345",
            None,
        )
    ]


@pytest.mark.asyncio
async def test_resolve_multinode_leader_worker_id_ignores_ambiguous_instances():
    class FakeClient:
        async def wait_for_instances(self):
            return [1234, 5678]

    class FakeEndpoint:
        async def client(self):
            return FakeClient()

    server_args = SimpleNamespace(nnodes=2, node_rank=1)

    worker_id = await _resolve_multinode_leader_worker_id(FakeEndpoint(), server_args)

    assert worker_id is None


@pytest.mark.asyncio
async def test_handle_non_leader_node_resolves_worker_before_kv_publish(monkeypatch):
    calls = []
    init_called = asyncio.Event()

    class FakeClient:
        async def wait_for_instance_by_runtime_data(self, key, value, timeout_s=None):
            return 1234

    class FakeEndpoint:
        async def client(self):
            return FakeClient()

    server_args = SimpleNamespace(
        nnodes=2,
        node_rank=1,
        dist_timeout=5,
        kv_events_config='{"endpoint": "tcp://*:5557"}',
    )

    class FakePublisher:
        def __init__(self):
            self.generate_endpoint = FakeEndpoint()
            self.server_args = server_args
            self.kv_worker_id = None

        def init_kv_event_publish(self):
            calls.append(self.kv_worker_id)
            init_called.set()

        def cleanup(self):
            pass

    monkeypatch.setattr(
        publisher_mod,
        "get_sglang_worker_group_id",
        lambda server_args: "dist_init:tcp://10.0.0.1:2345",
    )
    metrics_task = asyncio.create_task(asyncio.Event().wait())
    task = asyncio.create_task(
        handle_non_leader_node(
            SimpleNamespace(server_args=server_args),
            FakePublisher(),
            metrics_task,
        )
    )

    await asyncio.wait_for(init_called.wait(), timeout=1)
    assert calls == [1234]

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert metrics_task.cancelled()


@pytest.mark.asyncio
async def test_handle_non_leader_node_skips_kv_publish_without_resolved_worker(
    monkeypatch,
):
    resolution_done = asyncio.Event()
    init_called = asyncio.Event()
    cleanup_called = asyncio.Event()

    async def missing_resolution(generate_endpoint, server_args):
        resolution_done.set()
        return None

    class FakePublisher:
        generate_endpoint = object()
        server_args = SimpleNamespace(kv_events_config='{"endpoint": "tcp://*:5557"}')
        kv_worker_id = None

        def init_kv_event_publish(self):
            init_called.set()

        def cleanup(self):
            cleanup_called.set()

    monkeypatch.setattr(
        publisher_mod,
        "_resolve_multinode_leader_worker_id",
        missing_resolution,
    )
    metrics_task = asyncio.create_task(asyncio.Event().wait())
    task = asyncio.create_task(
        handle_non_leader_node(
            SimpleNamespace(server_args=SimpleNamespace(node_rank=1)),
            FakePublisher(),
            metrics_task,
        )
    )

    await asyncio.wait_for(resolution_done.wait(), timeout=1)
    await asyncio.sleep(0)

    assert not init_called.is_set()
    assert not task.done()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleanup_called.is_set()
    assert metrics_task.cancelled()


@pytest.mark.asyncio
async def test_handle_non_leader_node_cleans_up_when_resolution_fails(monkeypatch):
    cleanup_called = asyncio.Event()

    async def fail_resolution(generate_endpoint, server_args):
        raise RuntimeError("resolution failed")

    class FakePublisher:
        generate_endpoint = object()
        server_args = SimpleNamespace(kv_events_config='{"endpoint": "tcp://*:5557"}')

        def cleanup(self):
            cleanup_called.set()

    monkeypatch.setattr(
        publisher_mod,
        "_resolve_multinode_leader_worker_id",
        fail_resolution,
    )
    metrics_task = asyncio.create_task(asyncio.Event().wait())

    with pytest.raises(RuntimeError, match="resolution failed"):
        await handle_non_leader_node(
            SimpleNamespace(server_args=SimpleNamespace(node_rank=1)),
            FakePublisher(),
            metrics_task,
        )

    assert cleanup_called.is_set()
    assert metrics_task.cancelled()


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


def test_init_kv_event_publish_allows_zero_worker_id_override(monkeypatch):
    calls = []

    class FakeKvEventPublisher:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def shutdown(self):
            pass

    monkeypatch.setattr(publisher_mod, "KvEventPublisher", FakeKvEventPublisher)
    monkeypatch.setattr(
        publisher_mod,
        "get_zmq_socket",
        lambda *args, **kwargs: SimpleNamespace(close=lambda linger=0: None),
    )
    monkeypatch.setattr(publisher_mod, "get_local_ip_auto", lambda: "127.0.0.1")
    monkeypatch.setattr(
        publisher_mod,
        "ZmqEventPublisher",
        SimpleNamespace(
            offset_endpoint_port=staticmethod(lambda base_ep, dp_rank: "tcp://*:5557")
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
        dp_size=1,
        enable_dp_attention=False,
        nnodes=1,
        node_rank=0,
    )
    config = SimpleNamespace(
        server_args=server_args,
        dynamo_args=SimpleNamespace(enable_local_indexer=True),
    )
    publisher = DynamoSglangPublisher(
        engine=SimpleNamespace(
            port_args=SimpleNamespace(metrics_ipc_name="ipc://metrics")
        ),
        config=config,
        generate_endpoint=SimpleNamespace(),
        component_gauges=SimpleNamespace(),
        kv_worker_id=0,
    )

    publisher.init_kv_event_publish()

    assert calls[0]["worker_id"] == 0
    publisher.cleanup()
