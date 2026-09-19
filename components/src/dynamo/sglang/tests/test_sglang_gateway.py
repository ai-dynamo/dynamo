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


def test_validate_engine_routes_need_attach_api(monkeypatch):
    monkeypatch.delenv("DYN_SNAPSHOT_CONTROL_DIR", raising=False)
    monkeypatch.setattr(
        gateway.sgl.Engine, "attach_tokenizer_worker", None, raising=False
    )
    routes = ["flush_cache", "abort=abort_request:tm"]
    gateway.validate_gateway_mode(_server_args(4), _dyn(engine_routes=routes[1:]), 4)
    with pytest.raises(ValueError, match="attach_tokenizer_worker"):
        gateway.validate_gateway_mode(_server_args(4), _dyn(engine_routes=routes), 4)
    monkeypatch.setattr(gateway.sgl.Engine, "attach_tokenizer_worker", lambda pid: None)
    gateway.validate_gateway_mode(_server_args(4), _dyn(engine_routes=routes), 4)


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
    assert env2[gateway.ENV_SYSTEM_PORT] == "0"
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
    assert gateway.child_environment(1).get(gateway.ENV_SYSTEM_PORT) == value
    assert gateway.ENV_SYSTEM_PORT_BASE not in os.environ


def test_parent_watchdog_fires_when_parent_is_gone():
    import threading

    died = threading.Event()
    dead_pid = 2**22 - 7  # above pid_max on every default Linux, so never alive
    thread = gateway.start_parent_watchdog(
        dead_pid, on_parent_death=died.set, poll_seconds=0.01
    )
    assert died.wait(5)
    thread.join(5)
    assert not thread.is_alive()


def test_parent_watchdog_stays_quiet_while_parent_lives():
    import os
    import threading
    import time

    died = threading.Event()
    gateway.start_parent_watchdog(
        os.getppid(), on_parent_death=died.set, poll_seconds=0.01
    )
    time.sleep(0.1)
    assert not died.is_set()


def test_gateway_engine_id_and_shutdown_budget(monkeypatch):
    monkeypatch.delenv(gateway.ENV_PARENT_PID, raising=False)
    assert gateway.gateway_engine_id() is None
    monkeypatch.setenv(gateway.ENV_PARENT_PID, "777")
    assert gateway.gateway_engine_id().endswith(":777")
    monkeypatch.setenv("DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS", "12")
    assert gateway.child_shutdown_timeout() == 12 + gateway.CHILD_DRAIN_AND_CLEANUP_SECS


def _child_env(monkeypatch, parent_pid="4321"):
    monkeypatch.setenv(gateway.ENV_PARENT_PID, parent_pid)
    monkeypatch.setenv(gateway.ENV_CHILD_INDEX, "1")
    monkeypatch.setattr(gateway, "start_parent_watchdog", lambda pid: None)


def test_build_gateway_engine_prefers_attach_api(monkeypatch):
    _child_env(monkeypatch)
    seen = {}

    def attach(pid):
        seen["pid"] = pid
        return SimpleNamespace(port_args=SimpleNamespace(metrics_ipc_name="ipc:///p"))

    monkeypatch.setattr(
        gateway.sgl.Engine, "attach_tokenizer_worker", attach, raising=False
    )
    engine = gateway.build_gateway_engine()
    assert seen["pid"] == 4321
    assert engine.port_args.metrics_ipc_name == "ipc:///p"


def test_build_gateway_engine_falls_back_to_facade(monkeypatch):
    import sglang.srt.managers.multi_tokenizer_mixin as mixin
    import sglang.srt.runtime_context as runtime_context

    _child_env(monkeypatch)
    monkeypatch.setattr(
        gateway.sgl.Engine, "attach_tokenizer_worker", None, raising=False
    )
    port_args = SimpleNamespace(
        metrics_ipc_name="ipc:///p", tokenizer_ipc_name="ipc:///t"
    )
    server_args = SimpleNamespace(tokenizer_worker_num=2)
    info = {"max_req_input_len": 64, "startup_time": {"t": 1}}
    created = {}

    class FakeWorker:
        def __init__(self, sa, pa):
            created["port_args"] = pa

        def set_startup_time(self, t):
            created["startup_time"] = t

    monkeypatch.setattr(
        mixin, "read_from_shared_memory", lambda name: (port_args, server_args, info)
    )
    monkeypatch.setattr(mixin, "get_tokenizer_worker_class", lambda sa: FakeWorker)
    monkeypatch.setattr(
        runtime_context, "publish", lambda sa, role: created.setdefault("role", role)
    )

    engine = gateway.build_gateway_engine()
    assert isinstance(engine, gateway.GatewayEngine)
    assert created["role"] == "tokenizer"
    assert created["startup_time"] == {"t": 1}
    assert created["port_args"].tokenizer_ipc_name != "ipc:///t"
    assert engine.port_args.metrics_ipc_name == "ipc:///p"
    assert engine.tokenizer_manager.max_req_input_len == 64


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
    facade.shutdown()


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


def test_child_zero_receives_the_leader_load_time(not_a_child, monkeypatch):
    monkeypatch.delenv(gateway.ENV_LOAD_TIME, raising=False)
    env0 = gateway.child_environment(0, load_time=12.5)
    env1 = gateway.child_environment(1, load_time=12.5)
    assert env0[gateway.ENV_LOAD_TIME] == "12.5"
    assert gateway.ENV_LOAD_TIME not in env1
    assert gateway.ENV_LOAD_TIME not in gateway.child_environment(0)
    assert gateway.attached_engine_load_time() is None
    monkeypatch.setenv(gateway.ENV_LOAD_TIME, env0[gateway.ENV_LOAD_TIME])
    assert gateway.attached_engine_load_time() == 12.5


def test_follow_pause_broadcasts_resyncs_after_each_broadcast():
    seen = []

    class FakeWorker:
        is_pause = False

        async def _apply_pause_continue_broadcast(self, obj):
            self.is_pause = obj.is_pause

    worker = FakeWorker()

    async def on_change():
        seen.append(worker.is_pause)

    assert gateway.follow_pause_broadcasts(worker, on_change)
    assert not gateway.follow_pause_broadcasts(SimpleNamespace(), on_change)

    async def run():
        await worker._apply_pause_continue_broadcast(SimpleNamespace(is_pause=True))
        await worker._apply_pause_continue_broadcast(SimpleNamespace(is_pause=False))

    asyncio.run(run())
    assert seen == [True, False]
