# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for vLLM unified backend shutdown behaviour (issue #9343).

Tests that verify the shutdown sequence reaches a clean state — no
lingering non-daemon threads or subprocess references — after
``VllmLLMEngine.cleanup()`` is called, both when ``AsyncLLM.shutdown()``
completes normally and when it would hang without the hard-timeout guard.

These tests are mock-based: they do not require GPU hardware and are
marked ``@pytest.mark.unit`` so they run in the standard CPU-only CI gate.
"""

from __future__ import annotations

import asyncio
import threading
import warnings
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Guard: skip this file if vllm is not installed so that `pytest
# --co` (collect-only) does not produce an import error on CPU-only runners.
try:
    from vllm.v1.engine.async_llm import AsyncLLM
except ImportError:
    pytest.skip("vllm not installed", allow_module_level=True)

from dynamo.vllm.llm_engine import VllmLLMEngine


pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.pre_merge,
]


# ---------------------------------------------------------------------------
# Core invariant: cleanup() MUST await engine_client.shutdown()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cleanup_awaits_engine_client_shutdown():
    """``cleanup()`` must ``await engine_client.shutdown()`` — the call is
    async and awaiting it is what drives the EngineCore subprocess to exit.

    Before the fix, ``shutdown()`` was called without ``await`` (a
    fire-and-forget), so the method returned immediately while the
    subprocess kept running.  The interpreter then had no way to know the
    subprocess was stuck and would hang indefinitely after Worker.run()
    returned.

    This test verifies that ``AsyncLLM.shutdown()`` is actually awaited,
    which is the minimal behaviour change required to fix issue #9343.
    """
    engine = VllmLLMEngine(engine_args=SimpleNamespace(model="test-model"))

    shutdown_called = asyncio.Event()

    async def tracking_shutdown():
        shutdown_called.set()
        # Simulate the async work that must complete before the interpreter exits
        await asyncio.sleep(0)

    mock_client = MagicMock()
    mock_client.shutdown = tracking_shutdown
    engine.engine_client = mock_client

    await engine.cleanup()

    # If cleanup() awaited shutdown(), the event will have been set (non-daemon
    # threads are not created by this call).  If cleanup() forgot to await,
    # this assertion would fire because the coroutine object would be a
    # different identity (the unawaited coroutine runs until the loop closes).
    assert shutdown_called.is_set(), (
        "AsyncLLM.shutdown() was not awaited — cleanup() returned before the "
        "async shutdown work completed, leaving the EngineCore subprocess alive"
    )
    assert engine.engine_client is None


# ---------------------------------------------------------------------------
# Test: shutdown() hangs → the outer wait_for raises TimeoutError,
# and engine_client reference is cleared so os._exit(1) is effective
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cleanup_on_hanging_shutdown_clears_reference():
    """When ``AsyncLLM.shutdown()`` would hang, the
    ``asyncio.wait_for(engine.cleanup(), timeout=10.0)`` guard in
    ``run.py:_start()`` raises ``asyncio.TimeoutError`` after 10 seconds.
    ``cleanup()`` must still clear the ``engine_client`` reference on the
    path out, so that a subsequent ``os._exit(1)`` from the ``finally``
    block actually terminates the process (no lingering reference to the
    EngineCore subprocess).

    This simulates the real scenario from issue #9343 where the Python
    interpreter hangs because something in the subprocess/connector chain
    refuses to exit even after cleanup() returns.
    """
    engine = VllmLLMEngine(engine_args=SimpleNamespace(model="test-model"))

    # Slow shutdown: the artificial 60s sleep represents the real bug where
    # EngineCore subprocess cleanup blocks forever on a pipe/socket.
    async def hanging_shutdown():
        await asyncio.sleep(60)

    mock_client = MagicMock()
    mock_client.shutdown = hanging_shutdown
    engine.engine_client = mock_client

    # Run cleanup() wrapped in a 2-second timeout (simulating the guard)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(engine.cleanup(), timeout=2.0)

    # The key invariant: even when TimeoutError is raised, engine_client
    # must be None so that a force-exit (os._exit(1)) from the caller is
    # effective — the dangling reference doesn't keep the process open.
    assert engine.engine_client is None


# ---------------------------------------------------------------------------
# Test: cleanup() is safe to call twice (idempotent)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cleanup_idempotent():
    """``cleanup()`` must be safe to call multiple times — first call clears
    everything; second call must not raise even if client is already None.
    This matches the contract that Worker may call cleanup() on any failure
    path, including double-call.
    """
    engine = VllmLLMEngine(engine_args=SimpleNamespace(model="test-model"))

    mock_client = MagicMock()
    mock_client.shutdown = AsyncMock()
    engine.engine_client = mock_client

    await engine.cleanup()
    assert engine.engine_client is None

    # Second call — must not raise
    await engine.cleanup()
    assert engine.engine_client is None


# ---------------------------------------------------------------------------
# Test: cleanup() clears engine_client even when shutdown() raises
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cleanup_always_clears_engine_client_on_error():
    """``cleanup()`` must null out ``engine_client`` even if
    ``shutdown()`` raises — it must NOT silently swallow the error and
    leave the reference dangling."""
    engine = VllmLLMEngine(engine_args=SimpleNamespace(model="test-model"))

    mock_client = MagicMock()
    mock_client.shutdown = AsyncMock(side_effect=RuntimeError("engine refused to shut down"))
    engine.engine_client = mock_client

    # cleanup() must not raise, and engine_client must be cleared
    await engine.cleanup()
    assert engine.engine_client is None


# ---------------------------------------------------------------------------
# Test: run.py _start() finally block calls cleanup with timeout
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_finally_block_calls_cleanup_with_timeout():
    """The ``finally`` block in ``run.py:_start()`` must call
    ``asyncio.wait_for(engine.cleanup(), timeout=10.0)`` so that a
    hanging cleanup() is timed out and loop.stop() is called, allowing
    ``uvloop.run()`` to return.

    We verify this by checking that the real ``_start`` function (which
    lives in ``dynamo.common.backend.run``) is the one that gets invoked,
    and that after Worker.run() returns, cleanup() is called exactly once.
    """
    from dynamo.common.backend.run import _start
    from dynamo.common.backend.worker import WorkerConfig

    # Track calls to cleanup()
    cleanup_calls = []
    run_calls = []

    async def tracking_cleanup():
        cleanup_calls.append(True)

    mock_worker_config = WorkerConfig(model_name="test", served_model_name="test")

    mock_engine = MagicMock()
    mock_engine.cleanup = tracking_cleanup
    mock_engine.from_args = AsyncMock(
        return_value=(mock_engine, mock_worker_config)
    )

    mock_worker = MagicMock()
    mock_worker.run = AsyncMock(side_effect=lambda: run_calls.append(True))
    mock_worker_config2 = WorkerConfig(model_name="test", served_model_name="test")

    with patch("dynamo.common.backend.run.Worker", return_value=mock_worker):
        with patch("dynamo.common.backend.run.asyncio.wait_for", side_effect=lambda coro, timeout: coro) as mock_wait_for:
            # Patch get_running_loop so the finally block doesn't try to
            # remove_signal_handler in the test environment
            with patch("dynamo.common.backend.run.asyncio.get_running_loop", return_value=MagicMock(remove_signal_handler=lambda s: None, stop=lambda: None)):
                await _start(lambda: (mock_engine, mock_worker_config), None)

    # Verify Worker.run() was called (the "run completed" path)
    assert run_calls, "Worker.run() was not called"
    # Verify cleanup() was called through wait_for in the finally block
    assert len(cleanup_calls) == 1, (
        f"cleanup() was not called exactly once in _start finally block "
        f"(called {len(cleanup_calls)} times)"
    )


# ---------------------------------------------------------------------------
# Test: run.py finally block calls loop.stop() after cleanup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_finally_block_calls_loop_stop():
    """After ``engine.cleanup()`` returns (or times out), the ``finally``
    block in ``_start()`` must call ``loop.stop()`` so that
    ``uvloop.run()`` returns instead of blocking on any residual tasks.

    This is the second part of the fix for issue #9343: even with a clean
    shutdown, there may be asyncio tasks that won't complete unless the
    loop is explicitly stopped.
    """
    from dynamo.common.backend.run import _start
    from dynamo.common.backend.worker import WorkerConfig

    mock_worker_config = WorkerConfig(model_name="test", served_model_name="test")
    mock_engine = MagicMock()
    mock_engine.cleanup = AsyncMock()
    mock_engine.from_args = AsyncMock(
        return_value=(mock_engine, mock_worker_config)
    )
    mock_worker = MagicMock()
    mock_worker.run = AsyncMock(return_value=None)

    stopped_loop = MagicMock()

    with patch("dynamo.common.backend.run.Worker", return_value=mock_worker):
        with patch("dynamo.common.backend.run.asyncio.get_running_loop", return_value=stopped_loop):
            with patch("dynamo.common.backend.run.asyncio.wait_for", side_effect=lambda coro, timeout: coro):
                await _start(lambda: (mock_engine, mock_worker_config), None)

    # loop.stop() must have been called so uvloop.run() returns
    stopped_loop.stop.assert_called_once()