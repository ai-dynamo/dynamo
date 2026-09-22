# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

failover_watchdog = None
init_llm = None
_scheduler_dead = None
maybe_start_gms_failover_child_watchdog = None

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.fixture(autouse=True)
def _load_sglang_failover_modules():
    """Defer optional SGLang imports so collection works without the pinned runtime."""
    global failover_watchdog
    global init_llm
    global _scheduler_dead
    global maybe_start_gms_failover_child_watchdog

    from dynamo.sglang import failover_watchdog as watchdog_module
    from dynamo.sglang import init_llm as init_llm_module
    from dynamo.sglang.failover_watchdog import _scheduler_dead as scheduler_dead
    from dynamo.sglang.failover_watchdog import (
        maybe_start_gms_failover_child_watchdog as start_child_watchdog,
    )

    failover_watchdog = watchdog_module
    init_llm = init_llm_module
    _scheduler_dead = scheduler_dead
    maybe_start_gms_failover_child_watchdog = start_child_watchdog


@pytest.mark.asyncio
async def test_snapshot_crash_failover_rejected_before_engine_creation(monkeypatch):
    from dynamo.sglang import snapshot

    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "1")
    monkeypatch.setattr(snapshot.SnapshotConfig, "from_env", lambda: object())

    def unexpected_engine(**kwargs):
        pytest.fail("unfenced snapshot engine must never be constructed")

    monkeypatch.setattr(snapshot.sgl, "Engine", unexpected_engine)
    with pytest.raises(RuntimeError, match="writer-cohort fence"):
        await snapshot.prepare_snapshot_engine(SimpleNamespace())


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
    config = SimpleNamespace(
        server_args=server_args,
        dynamo_args=dynamo_args,
        use_resolved_server_args=lambda resolved: resolved,
    )
    engine = SimpleNamespace(server_args=server_args)

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
    assert calls[0][2] == {
        "backend_name": "sglang",
        "tags": ["kv_cache"],
        "promotion_warmup": None,
    }


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


@pytest.mark.asyncio
async def test_sglang_failover_watchdog_releases_lock_after_fence(monkeypatch):
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "1")
    monkeypatch.setenv("DYN_SGLANG_GMS_FAILOVER_CHILD_WATCHDOG_POLL_MS", "10")
    shutdown_requested = asyncio.Event()
    monkeypatch.setattr(
        failover_watchdog, "_request_owner_shutdown", shutdown_requested.set
    )
    monkeypatch.setattr(failover_watchdog, "_fence_children", lambda _engine: True)
    released = asyncio.Event()
    unregistered = asyncio.Event()
    events = []

    class Lock:
        async def release(self):
            events.append("release")
            released.set()

    class Endpoint:
        async def unregister_endpoint_instance(self):
            events.append("unregister")
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
        assert events == ["unregister", "release"]
        assert target._gms_failover_lock is None
        assert target.shutdown_event.is_set()
        assert shutdown_requested.is_set()
    finally:
        assert watchdog is not None
        watchdog.stop()


@pytest.mark.asyncio
async def test_sglang_failover_watchdog_retains_lock_when_unregister_fails(
    monkeypatch,
):
    shutdown_requested = asyncio.Event()
    monkeypatch.setattr(
        failover_watchdog, "_request_owner_shutdown", shutdown_requested.set
    )
    lock = SimpleNamespace(release=AsyncMock())

    class Endpoint:
        async def unregister_endpoint_instance(self):
            raise RuntimeError("discovery unavailable")

    target = SimpleNamespace(
        _gms_failover_lock=lock,
        generate_endpoint=Endpoint(),
        shutdown_event=asyncio.Event(),
    )
    watchdog = failover_watchdog.SGLangGmsFailoverChildWatchdog(
        target, SimpleNamespace(), asyncio.get_running_loop()
    )

    await watchdog._release_after_fence()

    lock.release.assert_not_awaited()
    assert target._gms_failover_lock is lock
    assert target.shutdown_event.is_set()
    assert shutdown_requested.is_set()


@pytest.mark.asyncio
async def test_sglang_failover_watchdog_bounds_endpoint_unregister(monkeypatch):
    monkeypatch.setattr(failover_watchdog, "_UNREGISTER_TIMEOUT_S", 0.01)
    shutdown_requested = asyncio.Event()
    monkeypatch.setattr(
        failover_watchdog, "_request_owner_shutdown", shutdown_requested.set
    )
    unregister_cancelled = asyncio.Event()
    release = AsyncMock(return_value=True)
    monkeypatch.setattr(
        failover_watchdog, "release_attached_gms_failover_lock", release
    )

    class Endpoint:
        async def unregister_endpoint_instance(self):
            try:
                await asyncio.Event().wait()
            finally:
                unregister_cancelled.set()

    lock = object()
    target = SimpleNamespace(
        _gms_failover_lock=lock,
        generate_endpoint=Endpoint(),
        shutdown_event=asyncio.Event(),
    )
    watchdog = failover_watchdog.SGLangGmsFailoverChildWatchdog(
        target, SimpleNamespace(), asyncio.get_running_loop()
    )

    await asyncio.wait_for(watchdog._release_after_fence(), timeout=0.2)
    await asyncio.wait_for(unregister_cancelled.wait(), timeout=0.2)

    release.assert_not_awaited()
    assert target._gms_failover_lock is lock
    assert target.shutdown_event.is_set()
    assert shutdown_requested.is_set()


@pytest.mark.asyncio
async def test_sglang_failover_watchdog_shutdown_survives_release_error(monkeypatch):
    shutdown_requested = asyncio.Event()
    monkeypatch.setattr(
        failover_watchdog, "_request_owner_shutdown", shutdown_requested.set
    )

    async def fail_release(*_args, **_kwargs):
        raise RuntimeError("release failed")

    monkeypatch.setattr(
        failover_watchdog, "release_attached_gms_failover_lock", fail_release
    )
    target = SimpleNamespace(
        _gms_failover_lock=object(), shutdown_event=asyncio.Event()
    )
    watchdog = failover_watchdog.SGLangGmsFailoverChildWatchdog(
        target, SimpleNamespace(), asyncio.get_running_loop()
    )

    await watchdog._release_after_fence()

    assert target._gms_failover_lock is not None
    assert target.shutdown_event.is_set()
    assert shutdown_requested.is_set()


def test_sglang_failover_watchdog_shutdown_when_loop_is_closed(monkeypatch):
    shutdown = threading.Event()
    monkeypatch.setattr(failover_watchdog, "_request_owner_shutdown", shutdown.set)
    monkeypatch.setattr(failover_watchdog, "_fence_children", lambda _engine: True)

    def fail_schedule(*_args, **_kwargs):
        raise RuntimeError("closed")

    monkeypatch.setattr(
        failover_watchdog.asyncio, "run_coroutine_threadsafe", fail_schedule
    )
    target = SimpleNamespace(_gms_failover_lock=object())
    watchdog = failover_watchdog.SGLangGmsFailoverChildWatchdog(
        target, SimpleNamespace(), object()
    )

    watchdog._trigger_failure("test")
    assert watchdog._fence_thread is not None
    watchdog._fence_thread.join(timeout=1.0)
    assert not watchdog._fence_thread.is_alive()

    assert shutdown.is_set()


@pytest.mark.asyncio
async def test_sglang_failover_watchdog_hooks_subprocess_watchdog(monkeypatch):
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "1")
    monkeypatch.setattr(failover_watchdog, "_request_owner_shutdown", lambda: None)
    monkeypatch.setattr(failover_watchdog, "_fence_children", lambda _engine: True)
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
        assert sglang_watchdog.original_called is False
        assert target._gms_failover_lock is None
        assert target.shutdown_event.is_set()
    finally:
        assert watchdog is not None
        watchdog.stop()


def test_sglang_failover_watchdog_routes_sigquit_to_controlled_handoff(monkeypatch):
    class FakeLoop:
        def __init__(self):
            self.previous = lambda: None
            self._signal_handlers = {
                failover_watchdog.signal.SIGQUIT: SimpleNamespace(
                    _callback=self.previous, _args=()
                )
            }

        def add_signal_handler(self, signum, callback, *args):
            self._signal_handlers[signum] = SimpleNamespace(
                _callback=callback, _args=args
            )

        def remove_signal_handler(self, signum):
            self._signal_handlers.pop(signum, None)
            return True

    loop = FakeLoop()
    watchdog = failover_watchdog.SGLangGmsFailoverChildWatchdog(
        SimpleNamespace(), SimpleNamespace(), loop
    )
    reasons = []
    monkeypatch.setattr(watchdog, "_trigger_failure", reasons.append)

    watchdog._install_sigquit_hook()
    installed = loop._signal_handlers[failover_watchdog.signal.SIGQUIT]._callback
    installed()
    assert reasons == ["SGLang SIGQUIT reported child failure"]

    watchdog._restore_sigquit_hook()
    restored = loop._signal_handlers[failover_watchdog.signal.SIGQUIT]._callback
    assert restored is loop.previous


def test_gms_multinode_shadow_enables_nccl_prewarm(monkeypatch):
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", " true ")
    server_args = SimpleNamespace(nnodes=2, pre_warm_nccl=False)

    init_llm._enable_gms_nccl_prewarm(server_args)

    assert server_args.pre_warm_nccl is True


def test_nccl_prewarm_is_scoped_to_multinode_shadow(monkeypatch):
    monkeypatch.delenv("DYN_GMS_FAILOVER_SHADOW_MODE", raising=False)
    vanilla = SimpleNamespace(nnodes=2, pre_warm_nccl=False)
    init_llm._enable_gms_nccl_prewarm(vanilla)
    assert vanilla.pre_warm_nccl is False

    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "1")
    single_node = SimpleNamespace(nnodes=1, pre_warm_nccl=False)
    init_llm._enable_gms_nccl_prewarm(single_node)
    assert single_node.pre_warm_nccl is False


def test_sglang_failure_trigger_is_exactly_once(monkeypatch):
    class Result:
        def result(self, timeout=None):
            return None

    def schedule(coro, _loop):
        coro.close()
        return Result()

    fences = []

    def fence(_engine):
        fences.append(1)
        time.sleep(0.02)
        return True

    monkeypatch.setattr(failover_watchdog, "_fence_children", fence)
    monkeypatch.setattr(failover_watchdog.asyncio, "run_coroutine_threadsafe", schedule)

    target = SimpleNamespace(_gms_failover_lock=object())
    watchdog = failover_watchdog.SGLangGmsFailoverChildWatchdog(
        target, SimpleNamespace(), object()
    )
    threads = [
        threading.Thread(target=watchdog._trigger_failure, args=(f"source-{i}",))
        for i in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert watchdog._fence_thread is not None
    watchdog._fence_thread.join(timeout=1.0)

    assert fences == [1]


@pytest.mark.asyncio
async def test_sleeping_standby_quiesces_without_unmapping_kv(monkeypatch):
    monkeypatch.setenv("DYN_SGLANG_GMS_LOCK_BEFORE_INIT", "0")
    manager = SimpleNamespace(
        pause_generation=AsyncMock(),
        release_memory_occupation=AsyncMock(),
        resume_memory_occupation=AsyncMock(),
        continue_generation=AsyncMock(),
    )
    controller = init_llm._NonLeaderFailoverController(
        SimpleNamespace(tokenizer_manager=manager)
    )

    assert await controller.quiesce(["kv_cache"]) is True
    manager.pause_generation.assert_not_awaited()
    manager.release_memory_occupation.assert_not_awaited()

    assert await controller.resume(["kv_cache"]) is True
    manager.resume_memory_occupation.assert_not_awaited()
    manager.continue_generation.assert_not_awaited()


def test_lock_before_init_keeps_release_and_remap_path(monkeypatch):
    monkeypatch.setenv("DYN_SGLANG_GMS_LOCK_BEFORE_INIT", "1")
    assert init_llm._uses_mapped_sleeping_standby() is False

    monkeypatch.setenv("DYN_SGLANG_GMS_LOCK_BEFORE_INIT", "0")
    assert init_llm._uses_mapped_sleeping_standby() is True
    assert init_llm._can_prewarm_mapped_standby() is False

    monkeypatch.setenv("GMS_KV_LEASES", "1")
    assert init_llm._can_prewarm_mapped_standby() is True
