# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the multi-process SGLang gateway (no engine, no GPU)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("sglang", reason="sglang not installed in this container")

from dynamo.sglang import gateway  # noqa: E402

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _server_args(tokenizer_worker_num=1, node_rank=0, **kw):
    return SimpleNamespace(
        tokenizer_worker_num=tokenizer_worker_num, node_rank=node_rank, **kw
    )


def _dyn(gateway_workers=None, **kw):
    return SimpleNamespace(gateway_workers=gateway_workers, **kw)


@pytest.fixture
def not_a_child(monkeypatch):
    monkeypatch.delenv(gateway.ENV_PARENT_PID, raising=False)
    monkeypatch.delenv(gateway.ENV_CHILD_INDEX, raising=False)


def test_effective_count_from_either_flag(not_a_child):
    assert gateway.effective_gateway_workers(_server_args(1), _dyn()) == 1
    assert gateway.effective_gateway_workers(_server_args(8), _dyn()) == 8
    assert gateway.effective_gateway_workers(_server_args(1), _dyn(3)) == 3
    assert gateway.effective_gateway_workers(_server_args(3), _dyn(3)) == 3


@pytest.mark.parametrize("tokenizer_workers,requested", [(8, 3), (2, 1)])
def test_conflicting_explicit_counts_are_rejected(
    not_a_child, tokenizer_workers, requested
):
    with pytest.raises(ValueError, match="conflicts"):
        gateway.effective_gateway_workers(
            _server_args(tokenizer_workers), _dyn(requested)
        )


def test_gateway_worker_count_only_on_leader_and_not_in_children(
    not_a_child, monkeypatch
):
    assert gateway.gateway_worker_count(_server_args(8), _dyn()) == 8
    assert gateway.gateway_worker_count(_server_args(8, node_rank=1), _dyn()) == 1
    monkeypatch.setenv(gateway.ENV_PARENT_PID, "123")
    assert gateway.is_gateway_child()
    assert gateway.gateway_worker_count(_server_args(8), _dyn()) == 1


def test_validate_rejects_unsupported_modes(not_a_child, monkeypatch):
    monkeypatch.delenv("DYN_SNAPSHOT_CONTROL_DIR", raising=False)
    gateway.validate_gateway_mode(_server_args(4), _dyn(), 4)
    gateway.validate_gateway_mode(_server_args(1), _dyn(embedding_worker=True), 1)
    with pytest.raises(ValueError, match="embedding-worker"):
        gateway.validate_gateway_mode(_server_args(4), _dyn(embedding_worker=True), 4)
    with pytest.raises(ValueError, match="enable-lora"):
        gateway.validate_gateway_mode(_server_args(4, enable_lora=True), _dyn(), 4)
    with pytest.raises(ValueError, match="forward-pass-metrics"):
        gateway.validate_gateway_mode(
            _server_args(4, enable_forward_pass_metrics=True), _dyn(), 4
        )
    monkeypatch.setenv("DYN_SNAPSHOT_CONTROL_DIR", "/snapshot-control")
    with pytest.raises(ValueError, match="snapshot"):
        gateway.validate_gateway_mode(_server_args(4), _dyn(), 4)


def test_child_index_decides_metrics_ownership_and_fanout(monkeypatch):
    monkeypatch.delenv(gateway.ENV_PARENT_PID, raising=False)
    monkeypatch.delenv(gateway.ENV_CHILD_INDEX, raising=False)
    assert gateway.gateway_child_index() == 0
    assert gateway.owns_engine_metrics()
    assert gateway.metrics_fanout_endpoint() is None
    monkeypatch.setenv(gateway.ENV_PARENT_PID, "4242")
    monkeypatch.setenv(gateway.ENV_CHILD_INDEX, "2")
    assert not gateway.owns_engine_metrics()
    assert gateway.metrics_fanout_endpoint().endswith("_4242")


def test_system_port_is_handed_to_children(not_a_child, monkeypatch):
    monkeypatch.setenv(gateway.ENV_SYSTEM_PORT, "8081")
    monkeypatch.delenv(gateway.ENV_SYSTEM_PORT_BASE, raising=False)
    gateway.reserve_system_port_for_children()
    import os

    assert os.environ[gateway.ENV_SYSTEM_PORT] == "-1"
    env0, env2 = gateway.child_environment(0), gateway.child_environment(2)
    assert env0[gateway.ENV_SYSTEM_PORT] == "8081"
    assert env2[gateway.ENV_SYSTEM_PORT] == "8083"
    assert gateway.ENV_SYSTEM_PORT_BASE not in env2
    assert env2[gateway.ENV_CHILD_INDEX] == "2"
    assert env2[gateway.ENV_PARENT_PID] == str(os.getpid())


@pytest.mark.parametrize("value", [None, "-1", "0", "not-a-port"])
def test_system_port_untouched_when_disabled(not_a_child, monkeypatch, value):
    import os

    monkeypatch.delenv(gateway.ENV_SYSTEM_PORT_BASE, raising=False)
    if value is None:
        monkeypatch.delenv(gateway.ENV_SYSTEM_PORT, raising=False)
    else:
        monkeypatch.setenv(gateway.ENV_SYSTEM_PORT, value)
    gateway.reserve_system_port_for_children()
    assert os.environ.get(gateway.ENV_SYSTEM_PORT) == value
    assert gateway.ENV_SYSTEM_PORT not in gateway.child_environment(1) or value
    assert gateway.ENV_SYSTEM_PORT_BASE not in os.environ


def test_gateway_engine_facade_generates_through_tokenizer_manager():
    seen = {}

    class FakeTokenizerManager:
        async def generate_request(self, obj, request):
            seen["obj"] = obj
            yield {"text": "ok"}

    facade = gateway.GatewayEngine(
        FakeTokenizerManager(),
        server_args=SimpleNamespace(),
        port_args=SimpleNamespace(metrics_ipc_name="ipc:///tmp/x"),
        scheduler_info={"max_req_input_len": 4},
    )
    assert facade._scheduler_init_result.scheduler_infos[0]["max_req_input_len"] == 4

    async def run():
        gen = await facade.async_generate(input_ids=[1, 2, 3], stream=True)
        return [chunk async for chunk in gen]

    assert asyncio.run(run()) == [{"text": "ok"}]
    assert seen["obj"].input_ids == [1, 2, 3]
    facade.shutdown()  # no-op: the parent owns the engine


class FakeProc:
    def __init__(self, pid):
        self.pid = pid
        self.returncode = None
        self.terminated = False
        self.waited = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def wait(self, timeout=None):
        self.waited = True
        return self.returncode


class FakeShm:
    unlinked = False

    def unlink(self):
        self.unlinked = True


def _engine():
    return SimpleNamespace(
        port_args=SimpleNamespace(),
        server_args=SimpleNamespace(),
        tokenizer_manager=SimpleNamespace(startup_time={"t": 0}),
        _scheduler_init_result=SimpleNamespace(
            scheduler_infos=[{"max_req_input_len": 8}]
        ),
    )


def test_serve_via_gateway_children_spawns_and_fails_on_dead_child(
    not_a_child, monkeypatch
):
    spawned = []

    def fake_popen(cmd, env):
        p = FakeProc(4000 + len(spawned))
        p.cmd, p.env = cmd, env
        spawned.append(p)
        return p

    shm = FakeShm()
    monkeypatch.setattr(gateway.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        "sglang.srt.managers.multi_tokenizer_mixin.write_data_for_multi_tokenizer",
        lambda port_args, server_args, info: shm,
    )
    monkeypatch.setattr(gateway.sys, "argv", ["dynamo.sglang", "--model-path", "/m"])

    async def fast_sleep(_):
        # let the loop observe the dead child on its first check
        spawned[1].returncode = 1

    monkeypatch.setattr(gateway.asyncio, "sleep", fast_sleep)

    with pytest.raises(RuntimeError, match="exited rc=1"):
        asyncio.run(gateway.serve_via_gateway_children(_engine(), 3, asyncio.Event()))

    assert len(spawned) == 3
    assert all(p.cmd[1:3] == ["-m", "dynamo.sglang"] for p in spawned)
    assert all(p.cmd[3:] == ["--model-path", "/m"] for p in spawned)
    assert [p.env[gateway.ENV_CHILD_INDEX] for p in spawned] == ["0", "1", "2"]
    assert all(gateway.ENV_PARENT_PID in p.env for p in spawned)
    assert all(p.terminated for p in spawned if p.pid != spawned[1].pid)
    assert all(p.waited for p in spawned)
    assert shm.unlinked


def test_serve_via_gateway_children_tolerates_child_exit_during_shutdown(
    not_a_child, monkeypatch
):
    spawned = []
    stop = asyncio.Event()

    def fake_popen(cmd, env):
        p = FakeProc(4100 + len(spawned))
        spawned.append(p)
        return p

    monkeypatch.setattr(gateway.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        "sglang.srt.managers.multi_tokenizer_mixin.write_data_for_multi_tokenizer",
        lambda port_args, server_args, info: FakeShm(),
    )

    async def fast_sleep(_):
        stop.set()
        spawned[0].returncode = 0

    monkeypatch.setattr(gateway.asyncio, "sleep", fast_sleep)
    asyncio.run(gateway.serve_via_gateway_children(_engine(), 2, stop))
    assert spawned[1].terminated and all(p.waited for p in spawned)


def test_serve_via_gateway_children_cleans_up_when_spawn_fails(
    not_a_child, monkeypatch
):
    spawned = []

    def fake_popen(cmd, env):
        if len(spawned) == 1:
            raise OSError("no more pids")
        p = FakeProc(5000 + len(spawned))
        spawned.append(p)
        return p

    shm = FakeShm()
    monkeypatch.setattr(gateway.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        "sglang.srt.managers.multi_tokenizer_mixin.write_data_for_multi_tokenizer",
        lambda port_args, server_args, info: shm,
    )
    with pytest.raises(OSError, match="no more pids"):
        asyncio.run(gateway.serve_via_gateway_children(_engine(), 3, asyncio.Event()))
    assert len(spawned) == 1 and spawned[0].terminated and spawned[0].waited
    assert shm.unlinked


def test_serve_via_gateway_children_reuses_engine_published_shm(
    not_a_child, monkeypatch
):
    """An SGLang Engine that already published its args (tokenizer_router set) owns
    the shared memory; the gateway must not publish or unlink it."""
    calls = []
    monkeypatch.setattr(
        "sglang.srt.managers.multi_tokenizer_mixin.write_data_for_multi_tokenizer",
        lambda *a: calls.append(a),
    )
    monkeypatch.setattr(gateway.subprocess, "Popen", lambda cmd, env: FakeProc(1))

    class OwnedShm:
        def unlink(self):
            raise AssertionError("gateway must not unlink the engine's shm")

    engine = _engine()
    engine._multi_tokenizer_shm = OwnedShm()
    stop = asyncio.Event()

    async def run():
        task = asyncio.create_task(gateway.serve_via_gateway_children(engine, 1, stop))
        await asyncio.sleep(0)
        stop.set()
        await task

    asyncio.run(run())
    assert calls == []
