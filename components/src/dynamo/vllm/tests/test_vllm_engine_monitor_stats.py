# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for VllmEngineMonitor._periodic_log_stats."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dynamo.vllm.engine_monitor import VllmEngineMonitor


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
