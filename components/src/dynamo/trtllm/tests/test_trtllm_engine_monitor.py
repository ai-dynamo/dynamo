# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import threading

import pytest

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


class _FakeRuntime:
    def __init__(self, *, shutdown_exception=None):
        self.shutdown_count = 0
        self._shutdown_exception = shutdown_exception

    def shutdown(self):
        self.shutdown_count += 1
        if self._shutdown_exception is not None:
            raise self._shutdown_exception


def _record_exit(monkeypatch):
    exit_calls = []

    def exit_fn(code):
        exit_calls.append(code)

    monkeypatch.setattr(os, "_exit", exit_fn)
    return exit_calls


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

    await asyncio.sleep(0.02)
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
    engine = _FakeEngine([False], fatal_error=RuntimeError("fatal"))
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

    await asyncio.sleep(0.02)
    await monitor.stop()

    assert monitor._monitor_task is None
    assert engine.shutdown_count == 0
    assert runtime.shutdown_count == 0
