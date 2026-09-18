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


def _server_args(tokenizer_worker_num=1, node_rank=0):
    return SimpleNamespace(
        tokenizer_worker_num=tokenizer_worker_num, node_rank=node_rank
    )


def test_gateway_worker_count_defaults_to_tokenizer_worker_num(monkeypatch):
    monkeypatch.delenv(gateway.ENV_PARENT_PID, raising=False)
    dyn = SimpleNamespace(gateway_workers=None)
    assert gateway.gateway_worker_count(_server_args(1), dyn) == 1
    assert gateway.gateway_worker_count(_server_args(8), dyn) == 8


def test_gateway_worker_count_knob_overrides(monkeypatch):
    monkeypatch.delenv(gateway.ENV_PARENT_PID, raising=False)
    dyn = SimpleNamespace(gateway_workers=3)
    assert gateway.gateway_worker_count(_server_args(8), dyn) == 3
    # The knob alone does not enable the mode: SGLang must run its router.
    assert gateway.gateway_worker_count(_server_args(1), dyn) == 1


def test_gateway_worker_count_only_on_leader_and_not_in_children(monkeypatch):
    dyn = SimpleNamespace(gateway_workers=None)
    monkeypatch.delenv(gateway.ENV_PARENT_PID, raising=False)
    assert gateway.gateway_worker_count(_server_args(8, node_rank=1), dyn) == 1
    monkeypatch.setenv(gateway.ENV_PARENT_PID, "123")
    assert gateway.is_gateway_child()
    assert gateway.gateway_worker_count(_server_args(8), dyn) == 1


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


def test_private_metrics_ipc_does_not_alias_parent(tmp_path):
    parent = SimpleNamespace(metrics_ipc_name="ipc:///parent", other="keep")
    child = gateway._private_metrics_ipc(parent)
    assert child is not parent
    assert child.other == "keep"
    assert child.metrics_ipc_name != parent.metrics_ipc_name
    assert parent.metrics_ipc_name == "ipc:///parent"


def test_serve_via_gateway_children_spawns_and_fails_on_dead_child(monkeypatch):
    spawned = []

    class FakeProc:
        def __init__(self, cmd, env):
            self.cmd, self.env, self.pid = cmd, env, 4000 + len(spawned)
            self.returncode = None
            self.terminated = False

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        def wait(self, timeout=None):
            return self.returncode

    def fake_popen(cmd, env):
        p = FakeProc(cmd, env)
        spawned.append(p)
        return p

    class FakeShm:
        unlinked = False

        def unlink(self):
            self.unlinked = True

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

    engine = SimpleNamespace(
        port_args=SimpleNamespace(),
        server_args=SimpleNamespace(),
        tokenizer_manager=SimpleNamespace(startup_time={"t": 0}),
        _scheduler_init_result=SimpleNamespace(
            scheduler_infos=[{"max_req_input_len": 8}]
        ),
    )

    with pytest.raises(RuntimeError, match="exited rc=1"):
        asyncio.run(gateway.serve_via_gateway_children(engine, 3, asyncio.Event()))

    assert len(spawned) == 3
    assert all(p.cmd[1:3] == ["-m", "dynamo.sglang"] for p in spawned)
    assert all(p.cmd[3:] == ["--model-path", "/m"] for p in spawned)
    assert all(gateway.ENV_PARENT_PID in p.env for p in spawned)
    assert all(p.terminated for p in spawned if p.pid != spawned[1].pid)
    assert shm.unlinked


def test_serve_via_gateway_children_reuses_engine_published_shm(monkeypatch):
    """An SGLang Engine that already published its args (tokenizer_router set) owns
    the shared memory; the gateway must not publish or unlink it."""
    calls = []
    monkeypatch.setattr(
        "sglang.srt.managers.multi_tokenizer_mixin.write_data_for_multi_tokenizer",
        lambda *a: calls.append(a),
    )
    monkeypatch.setattr(
        gateway.subprocess,
        "Popen",
        lambda cmd, env: SimpleNamespace(
            pid=1,
            poll=lambda: None,
            terminate=lambda: None,
            wait=lambda timeout=None: 0,
        ),
    )

    class OwnedShm:
        def unlink(self):
            raise AssertionError("gateway must not unlink the engine's shm")

    engine = SimpleNamespace(
        port_args=SimpleNamespace(),
        server_args=SimpleNamespace(),
        tokenizer_manager=SimpleNamespace(startup_time={}),
        _scheduler_init_result=SimpleNamespace(scheduler_infos=[{}]),
        _multi_tokenizer_shm=OwnedShm(),
    )
    stop = asyncio.Event()

    async def run():
        task = asyncio.create_task(gateway.serve_via_gateway_children(engine, 1, stop))
        await asyncio.sleep(0)
        stop.set()
        await task

    asyncio.run(run())
    assert calls == []
