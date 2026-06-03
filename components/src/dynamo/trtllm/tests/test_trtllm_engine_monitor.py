# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import signal

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


def _recorder():
    calls = []

    def kill_fn(pid, sig):
        calls.append(("kill", pid, sig))

    def exit_fn(code):
        calls.append(("exit", code))

    return calls, kill_fn, exit_fn


@pytest.mark.asyncio
async def test_monitor_disables_without_health_api():
    calls, kill_fn, exit_fn = _recorder()
    engine = _FakeEngine(supports_health_check=False)

    monitor = TrtllmEngineMonitor(
        engine,
        interval=0.01,
        shutdown_timeout=0.01,
        kill_fn=kill_fn,
        exit_fn=exit_fn,
        pid_fn=lambda: 123,
    )

    assert monitor._monitor_task is None
    assert calls == []
    await monitor.stop()


@pytest.mark.asyncio
async def test_monitor_stops_cleanly_on_shutdown_event():
    calls, kill_fn, exit_fn = _recorder()
    shutdown_event = asyncio.Event()
    engine = _FakeEngine([True])
    monitor = TrtllmEngineMonitor(
        engine,
        shutdown_event=shutdown_event,
        interval=0.01,
        shutdown_timeout=0.01,
        kill_fn=kill_fn,
        exit_fn=exit_fn,
        pid_fn=lambda: 123,
    )

    await asyncio.sleep(0.02)
    shutdown_event.set()
    assert monitor._monitor_task is not None
    await asyncio.wait_for(monitor._monitor_task, timeout=1.0)

    assert engine.check_count >= 1
    assert calls == []


@pytest.mark.asyncio
async def test_monitor_signals_and_exits_when_unhealthy():
    calls, kill_fn, exit_fn = _recorder()
    engine = _FakeEngine([False], fatal_error=RuntimeError("fatal"))

    monitor = TrtllmEngineMonitor(
        engine,
        interval=0.01,
        shutdown_timeout=0.01,
        kill_fn=kill_fn,
        exit_fn=exit_fn,
        pid_fn=lambda: 123,
    )

    assert monitor._monitor_task is not None
    await asyncio.wait_for(monitor._monitor_task, timeout=1.0)

    assert calls == [
        ("kill", 123, signal.SIGINT),
        ("exit", 1),
    ]


@pytest.mark.asyncio
async def test_monitor_treats_health_exception_as_fatal():
    calls, kill_fn, exit_fn = _recorder()
    engine = _FakeEngine(health_exception=RuntimeError("health failed"))

    monitor = TrtllmEngineMonitor(
        engine,
        interval=0.01,
        shutdown_timeout=0.01,
        kill_fn=kill_fn,
        exit_fn=exit_fn,
        pid_fn=lambda: 456,
    )

    assert monitor._monitor_task is not None
    await asyncio.wait_for(monitor._monitor_task, timeout=1.0)

    assert calls == [
        ("kill", 456, signal.SIGINT),
        ("exit", 1),
    ]


@pytest.mark.asyncio
async def test_monitor_stop_cancels_poll_task():
    calls, kill_fn, exit_fn = _recorder()
    engine = _FakeEngine([True])
    monitor = TrtllmEngineMonitor(
        engine,
        interval=10.0,
        shutdown_timeout=0.01,
        kill_fn=kill_fn,
        exit_fn=exit_fn,
        pid_fn=lambda: 123,
    )

    await asyncio.sleep(0.02)
    await monitor.stop()

    assert monitor._monitor_task is None
    assert calls == []
