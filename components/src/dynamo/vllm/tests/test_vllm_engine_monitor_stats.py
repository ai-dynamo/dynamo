# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for vLLM engine monitoring and explicit shutdown."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from vllm.v1.engine.exceptions import EngineDeadError

from dynamo.common.engine_monitor import EngineHealthMonitorConfig
from dynamo.vllm.engine_monitor import VllmEngineMonitor

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.fault_tolerance,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


@pytest.fixture
def mock_engine():
    """Create a mock engine client with log_stats enabled."""
    engine = AsyncMock()
    engine.log_stats = True
    engine.do_log_stats = AsyncMock()
    engine.check_health = AsyncMock()
    return engine


def _make_monitor(engine, shutdown_event=None):
    """Create a VllmEngineMonitor bypassing __init__ validation."""
    monitor = object.__new__(VllmEngineMonitor)
    monitor.runtime = MagicMock()
    monitor.engine_client = engine
    monitor.shutdown_event = shutdown_event
    monitor.health_config = EngineHealthMonitorConfig(
        interval=0.01,
        check_timeout=0.01,
        shutdown_timeout=0.01,
    )
    monitor._monitor_task = asyncio.get_event_loop().create_future()
    monitor._stats_task = asyncio.get_event_loop().create_future()
    return monitor


@pytest.mark.asyncio
async def test_periodic_log_stats_calls_do_log_stats(mock_engine):
    """Stats task calls do_log_stats after the interval."""
    shutdown_event = asyncio.Event()
    monitor = _make_monitor(mock_engine, shutdown_event)

    with patch.dict("os.environ", {"VLLM_LOG_STATS_INTERVAL": "0.05"}):
        task = asyncio.create_task(monitor._periodic_log_stats())
        await asyncio.sleep(0.15)
        shutdown_event.set()
        await task

    assert mock_engine.do_log_stats.call_count >= 1


@pytest.mark.asyncio
async def test_periodic_log_stats_skips_when_disabled(mock_engine):
    """Stats task exits immediately when log_stats is False."""
    mock_engine.log_stats = False
    monitor = _make_monitor(mock_engine)

    task = asyncio.create_task(monitor._periodic_log_stats())
    await task

    mock_engine.do_log_stats.assert_not_called()


@pytest.mark.asyncio
async def test_periodic_log_stats_skips_when_interval_zero(mock_engine):
    """Stats task exits immediately when interval is 0."""
    monitor = _make_monitor(mock_engine)

    with patch.dict("os.environ", {"VLLM_LOG_STATS_INTERVAL": "0"}):
        task = asyncio.create_task(monitor._periodic_log_stats())
        await task

    mock_engine.do_log_stats.assert_not_called()


@pytest.mark.asyncio
async def test_periodic_log_stats_skips_when_interval_negative(mock_engine):
    """Stats task exits immediately when interval is negative."""
    monitor = _make_monitor(mock_engine)

    with patch.dict("os.environ", {"VLLM_LOG_STATS_INTERVAL": "-1"}):
        task = asyncio.create_task(monitor._periodic_log_stats())
        await task

    mock_engine.do_log_stats.assert_not_called()


@pytest.mark.asyncio
async def test_periodic_log_stats_respects_shutdown_event(mock_engine):
    """Stats task stops when shutdown_event is set."""
    shutdown_event = asyncio.Event()
    monitor = _make_monitor(mock_engine, shutdown_event)

    with patch.dict("os.environ", {"VLLM_LOG_STATS_INTERVAL": "0.05"}):
        task = asyncio.create_task(monitor._periodic_log_stats())
        await asyncio.sleep(0.02)
        shutdown_event.set()
        await asyncio.wait_for(task, timeout=1.0)


@pytest.mark.asyncio
async def test_periodic_log_stats_handles_exception(mock_engine):
    """Stats task continues after do_log_stats raises."""
    shutdown_event = asyncio.Event()
    monitor = _make_monitor(mock_engine, shutdown_event)

    call_count = 0

    async def flaky_log_stats():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("transient error")

    mock_engine.do_log_stats = flaky_log_stats

    with patch.dict("os.environ", {"VLLM_LOG_STATS_INTERVAL": "0.05"}):
        task = asyncio.create_task(monitor._periodic_log_stats())
        await asyncio.sleep(0.2)
        shutdown_event.set()
        await task

    assert call_count >= 2


@pytest.mark.asyncio
async def test_periodic_log_stats_cancellation(mock_engine):
    """Stats task handles cancellation gracefully (exits without error)."""
    monitor = _make_monitor(mock_engine)

    with patch.dict("os.environ", {"VLLM_LOG_STATS_INTERVAL": "0.05"}):
        task = asyncio.create_task(monitor._periodic_log_stats())
        await asyncio.sleep(0.02)
        task.cancel()
        # The method catches CancelledError and breaks, so it completes normally
        await asyncio.wait_for(task, timeout=1.0)
        assert task.done()
        assert task.exception() is None


@pytest.mark.asyncio
async def test_periodic_log_stats_no_shutdown_event(mock_engine):
    """Stats task works without a shutdown_event (uses asyncio.sleep)."""
    monitor = _make_monitor(mock_engine, shutdown_event=None)

    with patch.dict("os.environ", {"VLLM_LOG_STATS_INTERVAL": "0.05"}):
        task = asyncio.create_task(monitor._periodic_log_stats())
        await asyncio.sleep(0.12)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert mock_engine.do_log_stats.call_count >= 1


@pytest.mark.asyncio
async def test_periodic_log_stats_malformed_interval(mock_engine):
    """Stats task falls back to default 10s when VLLM_LOG_STATS_INTERVAL is invalid."""
    shutdown_event = asyncio.Event()
    monitor = _make_monitor(mock_engine, shutdown_event)

    with patch.dict("os.environ", {"VLLM_LOG_STATS_INTERVAL": "not_a_number"}):
        # Should not crash — falls back to 10.0s default
        task = asyncio.create_task(monitor._periodic_log_stats())
        # Give it a moment to start (it will sleep 10s, so just cancel quickly)
        await asyncio.sleep(0.05)
        shutdown_event.set()
        await asyncio.wait_for(task, timeout=2.0)

    # Task ran without error (used 10s fallback, didn't crash)


@pytest.mark.asyncio
async def test_run_health_check_times_out(mock_engine):
    async def blocked_health_check():
        await asyncio.sleep(1.0)

    mock_engine.check_health = blocked_health_check
    monitor = _make_monitor(mock_engine)

    with pytest.raises(asyncio.TimeoutError):
        await monitor._run_health_check()


@pytest.mark.asyncio
@pytest.mark.parametrize("health_error", [EngineDeadError, asyncio.TimeoutError])
async def test_health_failure_during_worker_shutdown_stops_monitor(
    mock_engine, health_error
):
    shutdown_event = asyncio.Event()

    async def fail_health_check_during_shutdown():
        shutdown_event.set()
        raise health_error()

    mock_engine.check_health = fail_health_check_during_shutdown
    monitor = _make_monitor(mock_engine, shutdown_event)
    monitor._shutdown_engine = MagicMock()

    with patch("dynamo.vllm.engine_monitor.os._exit") as exit_process:
        await monitor._check_engine_health()

    monitor._shutdown_engine.assert_not_called()
    monitor.runtime.shutdown.assert_not_called()
    exit_process.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.timeout(5)
@pytest.mark.parametrize("with_shutdown_event", [False, True])
async def test_stop_finishes_background_tasks(
    mock_engine, monkeypatch, with_shutdown_event
):
    shutdown_event = asyncio.Event() if with_shutdown_event else None
    monitor = _make_monitor(mock_engine, shutdown_event)
    monitor.health_config = EngineHealthMonitorConfig(
        interval=60, check_timeout=0, shutdown_timeout=0
    )
    health_started = asyncio.Event()
    stats_started = asyncio.Event()
    blocked = asyncio.Event()

    async def check_health():
        health_started.set()
        await blocked.wait()

    async def log_stats():
        stats_started.set()
        await blocked.wait()

    mock_engine.check_health.side_effect = check_health
    mock_engine.do_log_stats.side_effect = log_stats
    monkeypatch.setenv("VLLM_LOG_STATS_INTERVAL", "0.001")
    monitor._monitor_task = asyncio.create_task(monitor._check_engine_health())
    monitor._stats_task = asyncio.create_task(monitor._periodic_log_stats())
    tasks = (monitor._monitor_task, monitor._stats_task)
    try:
        await asyncio.wait_for(health_started.wait(), timeout=1)
        await asyncio.wait_for(stats_started.wait(), timeout=1)
        monitor.stop()
        monitor.stop()
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=1)
        monitor.stop()

        assert all(task.done() for task in tasks)
        mock_engine.check_health.assert_awaited_once()
        mock_engine.do_log_stats.assert_awaited_once()
        mock_engine.shutdown.assert_not_called()
        monitor.runtime.shutdown.assert_not_called()
        if shutdown_event is not None:
            assert not shutdown_event.is_set()
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.timeout(5)
@pytest.mark.parametrize("stats_enabled", [False, True])
async def test_stop_before_tasks_start(mock_engine, stats_enabled):
    monitor = _make_monitor(mock_engine)
    monitor._monitor_task = asyncio.create_task(monitor._check_engine_health())
    monitor._stats_task = (
        asyncio.create_task(monitor._periodic_log_stats()) if stats_enabled else None
    )
    tasks = [
        task
        for task in (monitor._monitor_task, monitor._stats_task)
        if task is not None
    ]
    monitor.stop()
    monitor.stop()
    await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=1)

    assert all(task.cancelled() for task in tasks)
    mock_engine.check_health.assert_not_called()
    mock_engine.do_log_stats.assert_not_called()


@pytest.mark.asyncio
async def test_stop_after_stats_task_exits(mock_engine):
    mock_engine.log_stats = False
    monitor = _make_monitor(mock_engine)
    monitor._stats_task = asyncio.create_task(monitor._periodic_log_stats())
    await monitor._stats_task
    monitor.stop()
    monitor.stop()

    assert monitor._monitor_task.cancelled()
    assert monitor._stats_task.result() is None


def test_stop_after_failed_initialization():
    monitor = object.__new__(VllmEngineMonitor)
    with pytest.raises(ValueError, match="DistributedRuntime"):
        monitor.__init__(runtime=None, engine_client=None)
    monitor.stop()
    monitor.stop()
