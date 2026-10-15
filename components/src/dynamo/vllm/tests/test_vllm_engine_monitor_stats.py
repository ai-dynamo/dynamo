# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for vLLM engine monitoring and its scoped lifetime."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from vllm.v1.engine.exceptions import EngineDeadError

from dynamo.common.engine_monitor import EngineHealthMonitorConfig
from dynamo.vllm.engine_monitor import VllmEngineMonitor

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
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
async def test_monitor_context_stops_background_tasks(mock_engine):
    monitor = _make_monitor(mock_engine)

    async with monitor:
        assert not monitor._monitor_task.done()
        assert not monitor._stats_task.done()

    assert monitor._monitor_task.cancelled()
    assert monitor._stats_task.cancelled()


@pytest.mark.asyncio
async def test_monitor_context_preserves_wake_failure(mock_engine):
    monitor = _make_monitor(mock_engine)

    with pytest.raises(RuntimeError, match="wake failed"):
        async with monitor:
            raise RuntimeError("wake failed")

    assert monitor._monitor_task.cancelled()
    assert monitor._stats_task.cancelled()


@pytest.mark.asyncio
async def test_monitor_context_stops_tasks_on_cancellation(mock_engine):
    monitor = _make_monitor(mock_engine)
    entered = asyncio.Event()

    async def wait_in_standby():
        async with monitor:
            entered.set()
            await asyncio.Future()

    standby = asyncio.create_task(wait_in_standby())
    try:
        await asyncio.wait_for(entered.wait(), timeout=1.0)
        standby.cancel()
        with pytest.raises(asyncio.CancelledError):
            await standby
    finally:
        standby.cancel()
        await asyncio.gather(standby, return_exceptions=True)

    assert monitor._monitor_task.cancelled()
    assert monitor._stats_task.cancelled()


@pytest.mark.asyncio
async def test_monitor_stop_awaits_task_cleanup(mock_engine):
    monitor = _make_monitor(mock_engine)
    started = [asyncio.Event(), asyncio.Event()]
    stopped = [asyncio.Event(), asyncio.Event()]

    async def background_task(index):
        started[index].set()
        try:
            await asyncio.Future()
        finally:
            await asyncio.sleep(0)
            stopped[index].set()

    tasks = [asyncio.create_task(background_task(index)) for index in range(2)]
    monitor._monitor_task, monitor._stats_task = tasks
    try:
        async with monitor:
            await asyncio.wait_for(
                asyncio.gather(*(event.wait() for event in started)), timeout=1.0
            )

        assert all(event.is_set() for event in stopped)
        assert all(task.done() for task in tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_monitor_stop_surfaces_background_failure(mock_engine):
    monitor = _make_monitor(mock_engine)
    monitor._monitor_task.set_exception(RuntimeError("health task failed"))

    with pytest.raises(RuntimeError, match="health task failed"):
        async with monitor:
            pass

    assert monitor._stats_task.cancelled()


@pytest.mark.asyncio
async def test_dead_standby_uses_existing_fail_stop_shutdown(mock_engine):
    class WorkerExit(Exception):
        pass

    mock_engine.check_health.side_effect = EngineDeadError()
    monitor = _make_monitor(mock_engine)
    monitor._shutdown_engine = MagicMock()

    with patch(
        "dynamo.vllm.engine_monitor.os._exit", side_effect=WorkerExit
    ) as exit_process:
        with pytest.raises(WorkerExit):
            await monitor._check_engine_health()

    monitor._shutdown_engine.assert_called_once_with()
    monitor.runtime.shutdown.assert_called_once_with()
    exit_process.assert_called_once_with(1)
