# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

import asyncio
import sys
import types
from types import SimpleNamespace

import pytest


def _install_sglang_test_compat_modules() -> None:
    if "sglang.srt.observability.trace" not in sys.modules:
        observability_module = sys.modules.setdefault(
            "sglang.srt.observability",
            types.ModuleType("sglang.srt.observability"),
        )
        trace_module = types.ModuleType("sglang.srt.observability.trace")
        trace_module.set_global_trace_level = lambda *_args, **_kwargs: None
        setattr(observability_module, "trace", trace_module)
        sys.modules["sglang.srt.observability.trace"] = trace_module

    if "dynamo.sglang.register" not in sys.modules:
        register_module = types.ModuleType("dynamo.sglang.register")

        async def register_model_with_readiness_gate(*_args, **_kwargs):
            return True

        register_module.register_model_with_readiness_gate = (
            register_model_with_readiness_gate
        )
        sys.modules["dynamo.sglang.register"] = register_module


_install_sglang_test_compat_modules()
import dynamo.sglang.init_llm as init_llm
from dynamo.sglang.failover_watchdog import (
    _scheduler_dead,
    maybe_start_gms_failover_child_watchdog,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class _FakeEndpoint:
    def connection_id(self):
        return 1234


class _FakeRuntime:
    def endpoint(self, name):
        return _FakeEndpoint()


class _FakePublisher:
    def __init__(self):
        self.component_gauges = SimpleNamespace(set_model_load_time=lambda value: None)


@pytest.mark.asyncio
async def test_non_leader_decode_prepares_failover_before_publisher_loop(monkeypatch):
    events = []

    async def fake_prepare(engine, runtime, early_failover_activation):
        events.append("prepare")
        return object()

    async def fake_setup_sgl_metrics(engine, config, generate_endpoint):
        metrics_task = asyncio.create_task(asyncio.Event().wait())
        return _FakePublisher(), metrics_task, []

    async def fake_handle_non_leader_node(engine, publisher, metrics_task):
        events.append("handle_non_leader")
        metrics_task.cancel()

    monkeypatch.setattr(init_llm, "_prepare_non_leader_failover", fake_prepare)
    monkeypatch.setattr(init_llm, "setup_sgl_metrics", fake_setup_sgl_metrics)
    monkeypatch.setattr(init_llm, "handle_non_leader_node", fake_handle_non_leader_node)

    server_args = SimpleNamespace(
        node_rank=1,
        enable_forward_pass_metrics=False,
        enable_trace=False,
    )
    dynamo_args = SimpleNamespace(
        namespace="test-ns",
        component="backend",
        endpoint="generate",
        sglang_trace_level=0,
    )
    config = SimpleNamespace(server_args=server_args, dynamo_args=dynamo_args)
    engine = SimpleNamespace()

    await init_llm.init_decode(
        _FakeRuntime(),
        config,
        asyncio.Event(),
        [],
        snapshot_engine=engine,
    )

    assert events == ["prepare", "handle_non_leader"]


@pytest.mark.asyncio
async def test_non_leader_failover_controller_allows_missing_tokenizer_manager(
    monkeypatch,
):
    monkeypatch.delenv("DYN_GMS_FAILOVER_SHADOW_MODE", raising=False)
    controller = init_llm._NonLeaderFailoverController(SimpleNamespace())

    assert await controller.quiesce(["kv_cache"]) is True
    assert await controller.quiesce(["kv_cache"]) is False
    assert await controller.resume(["kv_cache"]) is True
    assert await controller.resume(["kv_cache"]) is False


@pytest.mark.asyncio
async def test_prepare_non_leader_failover_attaches_lock_owner(monkeypatch):
    calls = []

    class FakeActivation:
        enabled = True

        def attach_to(self, owner):
            owner.lock_attached = True

    async def fake_prepare(owner, runtime, **kwargs):
        calls.append((owner, runtime, kwargs))
        return FakeActivation()

    monkeypatch.setattr(init_llm, "prepare_gms_failover", fake_prepare)

    engine = SimpleNamespace(tokenizer_manager=SimpleNamespace())
    runtime = object()
    owner = await init_llm._prepare_non_leader_failover(engine, runtime, None)

    assert owner is calls[0][0]
    assert owner.lock_attached is True
    assert calls[0][1] is runtime
    assert calls[0][2] == {"backend_name": "sglang", "promotion_warmup": None}


def test_sglang_failover_watchdog_detects_dead_scheduler_process():
    class Proc:
        pid = 123
        exitcode = -9

        def is_alive(self):
            return False

    watchdog = SimpleNamespace(_processes=[Proc()], _names=["scheduler_0"])
    engine = SimpleNamespace(
        tokenizer_manager=SimpleNamespace(_subprocess_watchdog=watchdog)
    )

    assert _scheduler_dead(engine) is True


def test_sglang_failover_watchdog_detects_dead_detokenizer_process():
    class Proc:
        pid = 124
        exitcode = -9

        def is_alive(self):
            return False

    watchdog = SimpleNamespace(_processes=[Proc()], _names=["detokenizer"])
    engine = SimpleNamespace(
        tokenizer_manager=SimpleNamespace(_subprocess_watchdog=watchdog)
    )

    assert _scheduler_dead(engine) is True


@pytest.mark.asyncio
async def test_sglang_failover_watchdog_releases_lock_after_fence(monkeypatch):
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "1")
    monkeypatch.setenv("DYN_SGLANG_GMS_FAILOVER_CHILD_WATCHDOG_POLL_MS", "10")
    released = asyncio.Event()
    unregistered = asyncio.Event()

    class Lock:
        async def release(self):
            released.set()

    class Endpoint:
        async def unregister_endpoint_instance(self):
            unregistered.set()

    class Proc:
        pid = 456
        exitcode = -9

        def is_alive(self):
            return False

    target = SimpleNamespace(
        _gms_failover_lock=Lock(),
        generate_endpoint=Endpoint(),
        shutdown_event=asyncio.Event(),
    )
    engine = SimpleNamespace(
        get_all_child_pids=lambda: [],
        tokenizer_manager=SimpleNamespace(
            _subprocess_watchdog=SimpleNamespace(
                _processes=[Proc()], _names=["scheduler_0"]
            )
        ),
    )

    watchdog = maybe_start_gms_failover_child_watchdog(target, engine)
    try:
        await asyncio.wait_for(released.wait(), timeout=1.0)
        await asyncio.wait_for(unregistered.wait(), timeout=1.0)
        assert target._gms_failover_lock is None
        assert target.shutdown_event.is_set()
    finally:
        assert watchdog is not None
        watchdog.stop()


@pytest.mark.asyncio
async def test_sglang_failover_watchdog_hooks_subprocess_watchdog(monkeypatch):
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "1")
    released = asyncio.Event()

    class Lock:
        async def release(self):
            released.set()

    class Proc:
        pid = 789
        exitcode = -9

        def is_alive(self):
            return False

    class SGLangWatchdog:
        def __init__(self):
            self._processes = [Proc()]
            self._names = ["detokenizer"]
            self.original_called = False

        def _check_processes(self):
            self.original_called = True
            return True

    sglang_watchdog = SGLangWatchdog()
    target = SimpleNamespace(_gms_failover_lock=Lock(), shutdown_event=asyncio.Event())
    engine = SimpleNamespace(
        get_all_child_pids=lambda: [],
        tokenizer_manager=SimpleNamespace(_subprocess_watchdog=sglang_watchdog),
    )

    watchdog = maybe_start_gms_failover_child_watchdog(target, engine)
    try:
        assert getattr(sglang_watchdog, "_dynamo_gms_failover_hooked") is True
        assert sglang_watchdog._check_processes() is True
        await asyncio.wait_for(released.wait(), timeout=1.0)
        assert sglang_watchdog.original_called is True
        assert target._gms_failover_lock is None
        assert target.shutdown_event.is_set()
    finally:
        assert watchdog is not None
        watchdog.stop()
