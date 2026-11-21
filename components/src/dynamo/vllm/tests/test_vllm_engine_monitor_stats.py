# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for vLLM engine health monitoring, shutdown, and periodic stats."""

import asyncio
import ctypes
import multiprocessing
import signal
import time
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
    pytest.mark.fault_tolerance,
]


@pytest.fixture
def mock_engine():
    """Create a mock engine client with log_stats enabled."""
    engine = AsyncMock()
    engine.log_stats = True
    engine.do_log_stats = AsyncMock()
    engine.check_health = AsyncMock()
    engine.collective_rpc = AsyncMock()
    engine.shutdown = MagicMock()
    engine.errored = False
    return engine


def _make_monitor(engine, shutdown_event=None):
    """Create a VllmEngineMonitor bypassing __init__ validation."""
    monitor = object.__new__(VllmEngineMonitor)
    monitor.runtime = MagicMock()
    monitor.engine_client = engine
    monitor.shutdown_event = shutdown_event
    monitor.generate_endpoint = MagicMock()
    monitor.generate_endpoint.unregister_endpoint_instance = AsyncMock()
    monitor.is_busy = MagicMock(return_value=False)
    monitor.failed = False
    monitor._fatal_task = None
    monitor.processes = MagicMock()
    monitor.health_config = EngineHealthMonitorConfig(
        interval=0.01,
        check_timeout=0.01,
        shutdown_timeout=0.01,
    )
    monitor._monitor_task = asyncio.get_event_loop().create_future()
    monitor._stats_task = asyncio.get_event_loop().create_future()
    monitor._process_task = asyncio.get_event_loop().create_future()
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
async def test_run_health_check_does_not_queue_worker_rpc(mock_engine):
    monitor = _make_monitor(mock_engine)

    await monitor._run_health_check()

    mock_engine.check_health.assert_awaited_once_with()
    mock_engine.collective_rpc.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_health_check_local_failure_skips_worker_rpc(mock_engine):
    mock_engine.check_health.side_effect = EngineDeadError()
    monitor = _make_monitor(mock_engine)

    with pytest.raises(EngineDeadError):
        await monitor._run_health_check()

    mock_engine.collective_rpc.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["paused", "sleeping", "updating"])
async def test_busy_lifecycle_does_not_probe_execution_queue(mock_engine, state):
    monitor = _make_monitor(mock_engine)
    monitor.is_busy.return_value = True

    await monitor._run_health_check()

    mock_engine.check_health.assert_not_awaited()
    mock_engine.collective_rpc.assert_not_awaited()


@pytest.mark.asyncio
async def test_transition_to_busy_during_health_check_does_not_kill(mock_engine):
    async def begin_update():
        monitor.is_busy.return_value = True
        await asyncio.sleep(1)

    monitor = _make_monitor(mock_engine)
    mock_engine.check_health.side_effect = begin_update

    await monitor._run_health_check()

    assert not monitor.failed


@pytest.mark.asyncio
async def test_run_health_check_timeout_disabled(mock_engine):
    monitor = _make_monitor(mock_engine)
    monitor.health_config = EngineHealthMonitorConfig(0.01, 0, 0)

    await monitor._run_health_check()

    mock_engine.check_health.assert_awaited_once_with()
    mock_engine.collective_rpc.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "health_error", [EngineDeadError, asyncio.TimeoutError, RuntimeError]
)
async def test_worker_health_failure_terminates_parent(mock_engine, health_error):
    mock_engine.check_health.side_effect = health_error()
    monitor = _make_monitor(mock_engine)
    monitor.request_shutdown = MagicMock()

    await asyncio.wait_for(monitor._check_engine_health(), timeout=1.0)

    monitor.request_shutdown.assert_called_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize("busy", [True, False])
async def test_process_death_detection_is_independent_of_lifecycle(mock_engine, busy):
    monitor = _make_monitor(mock_engine)
    monitor.is_busy.return_value = busy
    monitor.processes.check.side_effect = RuntimeError("execution worker exited")
    monitor.request_shutdown = MagicMock()

    await monitor._check_engine_processes()

    monitor.request_shutdown.assert_called_once_with()
    mock_engine.check_health.assert_not_awaited()


@pytest.mark.asyncio
async def test_client_death_detected_during_sleep(mock_engine):
    monitor = _make_monitor(mock_engine)
    monitor.is_busy.return_value = True
    mock_engine.errored = True
    monitor.request_shutdown = MagicMock()

    await monitor._check_engine_processes()

    monitor.request_shutdown.assert_called_once_with()


@pytest.mark.asyncio
async def test_process_detection_during_blocked_health_check(mock_engine):
    monitor = _make_monitor(mock_engine)
    monitor.health_config = EngineHealthMonitorConfig(0.01, 0, 1)
    entered = asyncio.Event()

    async def block_health():
        entered.set()
        await asyncio.Event().wait()

    mock_engine.check_health.side_effect = block_health
    health_task = asyncio.create_task(monitor._run_health_check())
    await entered.wait()
    monitor.processes.check.side_effect = RuntimeError("execution worker exited")
    monitor.request_shutdown = MagicMock()
    try:
        await asyncio.wait_for(monitor._check_engine_processes(), timeout=1)
        monitor.request_shutdown.assert_called_once_with()
    finally:
        health_task.cancel()
        await asyncio.gather(health_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_fatal_shutdown_is_idempotent(mock_engine):
    monitor = _make_monitor(mock_engine)
    monitor._shutdown_worker = AsyncMock()

    first = monitor.request_shutdown()
    second = monitor.request_shutdown()
    await first

    assert first is second
    assert monitor.failed
    monitor._shutdown_worker.assert_awaited_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_cleanup", ["engine", "runtime", None])
async def test_fatal_shutdown_exits_even_if_cleanup_fails(mock_engine, failed_cleanup):
    monitor = _make_monitor(mock_engine)
    if failed_cleanup == "engine":
        mock_engine.shutdown.side_effect = RuntimeError("engine cleanup failed")
    elif failed_cleanup == "runtime":
        monitor.runtime.shutdown.side_effect = RuntimeError("runtime cleanup failed")

    with (
        patch("dynamo.vllm.engine_monitor.ShutdownWatchdog") as watchdog,
        patch(
            "dynamo.vllm.engine_monitor.os._exit", side_effect=SystemExit(1)
        ) as exit_process,
        pytest.raises(SystemExit),
    ):
        await monitor._shutdown_worker()

    mock_engine.shutdown.assert_called_once_with()
    monitor.runtime.shutdown.assert_called_once_with()
    watchdog.assert_called_once_with(0.01)
    watchdog.return_value.close.assert_called_once_with()
    exit_process.assert_called_once_with(1)


@pytest.mark.asyncio
async def test_fatal_shutdown_deadline_disabled(mock_engine):
    monitor = _make_monitor(mock_engine)
    monitor.health_config = EngineHealthMonitorConfig(0.01, 0.01, 0)

    with (
        patch("dynamo.vllm.engine_monitor.ShutdownWatchdog") as watchdog,
        patch(
            "dynamo.vllm.engine_monitor.os._exit", side_effect=SystemExit(1)
        ) as exit_process,
        pytest.raises(SystemExit),
    ):
        await monitor._shutdown_worker()

    watchdog.assert_not_called()
    exit_process.assert_called_once_with(1)


@pytest.mark.asyncio
@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(), reason="requires fork"
)
@pytest.mark.parametrize("blocked_cleanup", ["engine", "runtime", "gil"])
async def test_fatal_shutdown_deadline_exits_real_process(mock_engine, blocked_cleanup):
    monitor = _make_monitor(mock_engine)
    monitor.health_config = EngineHealthMonitorConfig(0.01, 0.01, 0.05)

    def block_cleanup():
        if blocked_cleanup == "gil":
            ctypes.PyDLL(None).sleep(10)
        else:
            time.sleep(10)

    if blocked_cleanup in ("engine", "gil"):
        mock_engine.shutdown.side_effect = block_cleanup
    else:
        monitor.runtime.shutdown.side_effect = block_cleanup

    def shutdown_child():
        asyncio.run(monitor._shutdown_worker())

    process = multiprocessing.get_context("fork").Process(target=shutdown_child)
    try:
        process.start()
        process.join(timeout=3.0)
        assert not process.is_alive(), "Fatal shutdown exceeded its deadline"
        assert process.exitcode == -signal.SIGKILL
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=3.0)
        process.close()


@pytest.mark.asyncio
async def test_watchdog_start_failure_still_exits(mock_engine):
    monitor = _make_monitor(mock_engine)
    with (
        patch("dynamo.vllm.engine_monitor.ShutdownWatchdog", side_effect=OSError),
        patch("dynamo.vllm.engine_monitor.os._exit", side_effect=SystemExit(1)),
        pytest.raises(SystemExit),
    ):
        await monitor._shutdown_worker()


@pytest.mark.asyncio
@pytest.mark.parametrize("withdrawal", ["ok", "error", "timeout"])
async def test_discovery_withdrawal_precedes_cleanup(mock_engine, withdrawal):
    monitor = _make_monitor(mock_engine)
    events = []

    async def unregister():
        events.append("unregister")
        if withdrawal == "error":
            raise RuntimeError("discovery unavailable")
        if withdrawal == "timeout":
            await asyncio.sleep(1)

    monitor.generate_endpoint.unregister_endpoint_instance.side_effect = unregister
    mock_engine.shutdown.side_effect = lambda: events.append("engine")
    monitor.runtime.shutdown.side_effect = lambda: events.append("runtime")
    with (
        patch("dynamo.vllm.engine_monitor.ShutdownWatchdog"),
        patch("dynamo.vllm.engine_monitor.os._exit", side_effect=SystemExit(1)),
        pytest.raises(SystemExit),
    ):
        await monitor._shutdown_worker()

    assert events == ["unregister", "engine", "runtime"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "health_error", [EngineDeadError, asyncio.TimeoutError, RuntimeError]
)
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
