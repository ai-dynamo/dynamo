# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for vLLM unified backend shutdown behaviour (issue #9343).

Tests that verify VllmLLMEngine.cleanup() correctly awaits
AsyncLLM.shutdown() so the EngineCore subprocess exits cleanly on SIGTERM.

These tests are mock-based: they do not require GPU hardware and are
marked ``@pytest.mark.unit`` so they run in the standard CPU-only CI gate.
"""

from __future__ import annotations

import asyncio
import warnings
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

# Guard: skip this file if dynamo.vllm or vllm is not installed/compatible so
# that `pytest --co` (collect-only) does not produce an import error on
# CPU-only runners.  Catches both ImportError and any other error raised
# during vllm initialisation (e.g. NotImplementedError from vllm._C).
try:
    from dynamo.vllm.llm_engine import VllmLLMEngine
except Exception:
    pytest.skip(
        "dynamo.vllm not importable (vllm absent or incompatible)",
        allow_module_level=True,
    )

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
    """When ``AsyncLLM.shutdown()`` would hang, a ``asyncio.wait_for``
    timeout raises ``asyncio.TimeoutError``.  ``cleanup()`` must still
    clear the ``engine_client`` reference on the path out, so that a
    subsequent ``os._exit(1)`` actually terminates the process (no
    lingering reference to the EngineCore subprocess).

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
    mock_client.shutdown = AsyncMock(
        side_effect=RuntimeError("engine refused to shut down")
    )
    engine.engine_client = mock_client

    # cleanup() must not raise, and engine_client must be cleared
    await engine.cleanup()
    assert engine.engine_client is None
