# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import threading
import time

import pytest

from dynamo.common.engine_monitor import (
    ENGINE_HEALTH_CHECK_INTERVAL,
    ENGINE_HEALTH_CHECK_INTERVAL_ENV,
    ENGINE_HEALTH_CHECK_TIMEOUT,
    ENGINE_HEALTH_CHECK_TIMEOUT_ENV,
    ENGINE_HEALTH_SHUTDOWN_TIMEOUT,
    ENGINE_HEALTH_SHUTDOWN_TIMEOUT_ENV,
)
from dynamo.trtllm import engine_monitor as engine_monitor_module
from dynamo.trtllm.engine_monitor import TrtllmEngineMonitor

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


class _FakeEngine:
    def __init__(
        self,
        health_results=None,
        *,
        supports_health_check=True,
        fatal_error=None,
        health_exception=None,
    ):
        self._health_results = list(health_results or [True])
        self._supports_health_check = supports_health_check
        self._fatal_error = fatal_error
        self._health_exception = health_exception
        self.check_count = 0
        self.shutdown_count = 0

    def supports_health_check(self):
        return self._supports_health_check

    def check_health(self):
        self.check_count += 1
        if self._health_exception is not None:
            raise self._health_exception
        if len(self._health_results) > 1:
            return self._health_results.pop(0)
        return self._health_results[0]

    def get_health_check_fatal_error(self):
        return self._fatal_error

    def shutdown(self):
        self.shutdown_count += 1


class _BlockingEngine(_FakeEngine):
    def __init__(self):
        super().__init__([True])
        self.release = threading.Event()

    def check_health(self):
        self.check_count += 1
        self.release.wait(timeout=1.0)
        return True


class _HangingShutdownEngine(_FakeEngine):
    def __init__(self, health_results=None, *, fatal_error=None):
        super().__init__(health_results, fatal_error=fatal_error)
        self.release = threading.Event()

    def shutdown(self):
        self.shutdown_count += 1
        self.release.wait()


class _FakeRuntime:
    def __init__(self, *, shutdown_exception=None):
        self.shutdown_count = 0
        self._shutdown_exception = shutdown_exception

    def shutdown(self):
        self.shutdown_count += 1
        if self._shutdown_exception is not None:
            raise self._shutdown_exception


@pytest.fixture(autouse=True)
def _fresh_fatal_exit_lock(monkeypatch):
    # os._exit is patched in these tests, so the process-wide run-once lock
    # would otherwise stay held by the first test that reaches it.
    monkeypatch.setattr(
        engine_monitor_module, "_fatal_exit_lock", threading.Lock(), raising=False
    )


def _record_exit(monkeypatch):
    exit_calls = []

    def exit_fn(code):
        exit_calls.append(code)

    monkeypatch.setattr(os, "_exit", exit_fn)
    return exit_calls


async def _wait_for_exit(exit_calls, timeout=1.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not exit_calls:
        if loop.time() >= deadline:
            raise AssertionError("timed out waiting for os._exit")
        await asyncio.sleep(0.01)


async def _wait_for_check_count(engine, expected=1, timeout=1.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while engine.check_count < expected:
        if loop.time() >= deadline:
            raise AssertionError(
                f"timed out waiting for check_count >= {expected}; "
                f"got {engine.check_count}"
            )
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_monitor_disables_without_health_api():
    engine = _FakeEngine(supports_health_check=False)
    runtime = _FakeRuntime()

    monitor = TrtllmEngineMonitor(
        engine,
        runtime=runtime,
        interval=0.01,
        shutdown_timeout=0.01,
    )

    assert monitor._monitor_task is None
    assert engine.shutdown_count == 0
    assert runtime.shutdown_count == 0
    await monitor.stop()


def test_monitor_env_non_finite_values_fall_back_to_defaults(monkeypatch):
    monkeypatch.setenv(ENGINE_HEALTH_CHECK_INTERVAL_ENV, "nan")
    monkeypatch.setenv(ENGINE_HEALTH_CHECK_TIMEOUT_ENV, "inf")
    monkeypatch.setenv(ENGINE_HEALTH_SHUTDOWN_TIMEOUT_ENV, "-inf")
    engine = _FakeEngine(supports_health_check=False)

    monitor = TrtllmEngineMonitor(engine)

    assert monitor.interval == ENGINE_HEALTH_CHECK_INTERVAL
    assert monitor.check_timeout == ENGINE_HEALTH_CHECK_TIMEOUT
    assert monitor.shutdown_timeout == ENGINE_HEALTH_SHUTDOWN_TIMEOUT


def test_monitor_reads_engine_health_env_overrides(monkeypatch):
    monkeypatch.setenv(ENGINE_HEALTH_CHECK_INTERVAL_ENV, "0.25")
    monkeypatch.setenv(ENGINE_HEALTH_CHECK_TIMEOUT_ENV, "7.5")
    monkeypatch.setenv(ENGINE_HEALTH_SHUTDOWN_TIMEOUT_ENV, "12.5")
    engine = _FakeEngine(supports_health_check=False)

    monitor = TrtllmEngineMonitor(engine)

    assert monitor.interval == 0.25
    assert monitor.check_timeout == 7.5
    assert monitor.shutdown_timeout == 12.5


@pytest.mark.asyncio
async def test_monitor_stops_cleanly_on_shutdown_event():
    shutdown_event = asyncio.Event()
    engine = _FakeEngine([True])
    runtime = _FakeRuntime()
    monitor = TrtllmEngineMonitor(
        engine,
        runtime=runtime,
        shutdown_event=shutdown_event,
        interval=0.01,
        shutdown_timeout=0.01,
    )

    await _wait_for_check_count(engine)
    shutdown_event.set()
    assert monitor._monitor_task is not None
    await asyncio.wait_for(monitor._monitor_task, timeout=1.0)

    assert engine.check_count >= 1
    assert engine.shutdown_count == 0
    assert runtime.shutdown_count == 0


@pytest.mark.asyncio
async def test_monitor_shuts_down_engine_runtime_and_exits_when_unhealthy(monkeypatch):
    exit_calls = _record_exit(monkeypatch)
    shutdown_event = asyncio.Event()
    engine = _FakeEngine([False])
    runtime = _FakeRuntime()

    monitor = TrtllmEngineMonitor(
        engine,
        runtime=runtime,
        shutdown_event=shutdown_event,
        interval=0.01,
        shutdown_timeout=0.01,
    )

    assert monitor._monitor_task is not None
    await asyncio.wait_for(monitor._monitor_task, timeout=1.0)

    assert engine.shutdown_count == 1
    assert runtime.shutdown_count == 1
    assert not shutdown_event.is_set()
    assert exit_calls == [1]


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_monitor_bounds_hanging_engine_shutdown(monkeypatch):
    """Stop admission before waiting for a teardown that cannot finish."""
    exit_calls = _record_exit(monkeypatch)
    engine = _HangingShutdownEngine([False])
    runtime = _FakeRuntime()
    runtime_calls_at_teardown = []
    engine_shutdown = engine.shutdown

    def shutdown_engine():
        runtime_calls_at_teardown.append(runtime.shutdown_count)
        engine_shutdown()

    monkeypatch.setattr(engine, "shutdown", shutdown_engine)

    try:
        monitor = TrtllmEngineMonitor(
            engine,
            runtime=runtime,
            interval=0.01,
            shutdown_timeout=0.2,
        )
        # The loop must keep serving while the teardown waits for its bound.
        ticks = 0
        while not exit_calls:
            await asyncio.sleep(0.01)
            ticks += 1
        await asyncio.wait_for(monitor._monitor_task, timeout=1.0)

        assert ticks > 5
        assert runtime_calls_at_teardown == [1]
        assert engine.shutdown_count == 1
        assert runtime.shutdown_count == 1
        assert exit_calls == [1]
    finally:
        engine.release.set()


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_monitor_stops_waiting_when_engine_dies_during_shutdown(monkeypatch):
    exit_calls = _record_exit(monkeypatch)
    engine = _HangingShutdownEngine([False])
    runtime = _FakeRuntime()

    try:
        monitor = TrtllmEngineMonitor(
            engine,
            runtime=runtime,
            interval=0.01,
            shutdown_timeout=60.0,
        )
        while engine.shutdown_count == 0:
            await asyncio.sleep(0.01)
        engine._fatal_error = RuntimeError("MPI worker rank 0 exited unexpectedly")
        await _wait_for_exit(exit_calls, timeout=3.0)
        await asyncio.wait_for(monitor._monitor_task, timeout=1.0)

        assert engine.shutdown_count == 1
        assert runtime.shutdown_count == 1
        assert exit_calls == [1]
    finally:
        engine.release.set()


@pytest.mark.parametrize(
    ("interval", "expected"),
    [(0.0, 0.05), (0.2, 0.2), (10.0, 0.5)],
)
def test_watchdog_poll_has_positive_floor_and_upper_bound(interval, expected):
    assert engine_monitor_module._watchdog_poll_seconds(interval) == expected


@pytest.mark.timeout(10)
@pytest.mark.parametrize("stop_method", [None, "shutdown_event", "monitor_stop"])
def test_watchdog_exits_when_event_loop_is_blocked(monkeypatch, stop_method):
    """A dead rank can block the loop thread in a TRT-LLM IPC send; the
    watchdog thread must still end the process."""
    exit_calls = _record_exit(monkeypatch)
    engine = _HangingShutdownEngine([True])
    runtime = _FakeRuntime()

    async def start_then_block_loop():
        shutdown_event = asyncio.Event()
        monitor = TrtllmEngineMonitor(
            engine,
            runtime=runtime,
            shutdown_event=shutdown_event,
            interval=0.01,
            shutdown_timeout=60.0,
        )
        engine._fatal_error = RuntimeError("MPI worker rank 0 exited unexpectedly")
        if stop_method == "shutdown_event":
            shutdown_event.set()
        elif stop_method == "monitor_stop":
            await monitor.stop()
        deadline = time.monotonic() + 5.0
        while not exit_calls and time.monotonic() < deadline:
            time.sleep(0.01)

    try:
        asyncio.run(start_then_block_loop())

        assert exit_calls == [1]
        assert engine.shutdown_count == 0
        assert runtime.shutdown_count == 1
    finally:
        engine.release.set()


def test_fatal_exit_runs_once(monkeypatch):
    exit_calls = _record_exit(monkeypatch)
    engine = _FakeEngine(supports_health_check=False)
    runtime = _FakeRuntime()

    engine_monitor_module.exit_worker_on_fatal_error(engine, runtime, 1.0)
    engine_monitor_module.exit_worker_on_fatal_error(engine, runtime, 1.0)

    assert engine.shutdown_count == 1
    assert runtime.shutdown_count == 1
    assert exit_calls == [1]


@pytest.mark.asyncio
async def test_monitor_exits_when_runtime_shutdown_raises(monkeypatch):
    exit_calls = _record_exit(monkeypatch)
    engine = _FakeEngine([False])
    runtime = _FakeRuntime(shutdown_exception=RuntimeError("runtime shutdown failed"))

    monitor = TrtllmEngineMonitor(
        engine,
        runtime=runtime,
        interval=0.01,
        shutdown_timeout=0.01,
    )

    assert monitor._monitor_task is not None
    await asyncio.wait_for(monitor._monitor_task, timeout=1.0)

    assert engine.shutdown_count == 1
    assert runtime.shutdown_count == 1
    assert exit_calls == [1]


@pytest.mark.asyncio
async def test_monitor_treats_health_exception_as_fatal(monkeypatch):
    exit_calls = _record_exit(monkeypatch)
    engine = _FakeEngine(health_exception=RuntimeError("health failed"))
    runtime = _FakeRuntime()

    monitor = TrtllmEngineMonitor(
        engine,
        runtime=runtime,
        interval=0.01,
        shutdown_timeout=0.01,
    )

    assert monitor._monitor_task is not None
    await asyncio.wait_for(monitor._monitor_task, timeout=1.0)

    assert engine.shutdown_count == 1
    assert runtime.shutdown_count == 1
    assert exit_calls == [1]


@pytest.mark.asyncio
async def test_monitor_treats_health_timeout_as_fatal(monkeypatch):
    exit_calls = _record_exit(monkeypatch)
    engine = _BlockingEngine()
    runtime = _FakeRuntime()

    try:
        monitor = TrtllmEngineMonitor(
            engine,
            runtime=runtime,
            interval=0.01,
            check_timeout=0.01,
            shutdown_timeout=0.01,
        )

        assert monitor._monitor_task is not None
        await asyncio.wait_for(monitor._monitor_task, timeout=1.0)

        assert engine.check_count == 1
        assert engine.shutdown_count == 1
        assert runtime.shutdown_count == 1
        assert exit_calls == [1]
    finally:
        engine.release.set()


@pytest.mark.asyncio
async def test_monitor_shutdown_works_without_runtime(monkeypatch):
    exit_calls = _record_exit(monkeypatch)
    engine = _FakeEngine([False])

    monitor = TrtllmEngineMonitor(
        engine,
        interval=0.01,
        shutdown_timeout=0.01,
    )

    assert monitor._monitor_task is not None
    await asyncio.wait_for(monitor._monitor_task, timeout=1.0)

    assert engine.shutdown_count == 1
    assert exit_calls == [1]


@pytest.mark.asyncio
async def test_monitor_stop_cancels_poll_task():
    engine = _FakeEngine([True])
    runtime = _FakeRuntime()
    monitor = TrtllmEngineMonitor(
        engine,
        runtime=runtime,
        interval=10.0,
        shutdown_timeout=0.01,
    )

    await _wait_for_check_count(engine)
    await monitor.stop()

    assert monitor._monitor_task is None
    assert engine.shutdown_count == 0
    assert runtime.shutdown_count == 0
