# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Worker's lifecycle state machine.

Pins down the invariants:
    * engine.cleanup() runs at most once, only after engine.start() succeeded.
    * engine.start() and engine.cleanup() never run concurrently.
    * cleanup() before start() (shutdown-before-start race) skips
      engine.cleanup() and makes a follow-up start() abort cleanly.
    * cleanup() arriving while start() is in flight waits for start to finish.
    * If engine.start() raises, engine.cleanup() is not called.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from dynamo.common.backend.worker import Worker, WorkerConfig
from dynamo.llm.exceptions import EngineShutdown

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def _make_worker() -> tuple:
    engine = AsyncMock()
    engine.start = AsyncMock(return_value=MagicMock())
    engine.cleanup = AsyncMock()
    config = WorkerConfig(namespace="test")
    return Worker(engine=engine, config=config), engine


def test_cleanup_after_start_calls_engine_cleanup_exactly_once():
    """The happy path: start -> cleanup -> repeated cleanup is a no-op."""
    worker, engine = _make_worker()

    async def _run():
        await worker._start_engine()
        await worker._cleanup_once()
        await worker._cleanup_once()
        await worker._cleanup_once()

    asyncio.run(_run())
    assert engine.start.await_count == 1
    assert engine.cleanup.await_count == 1


@pytest.mark.timeout(5)
def test_cleanup_once_concurrent_invocations_only_run_once():
    """Concurrent _cleanup_once invocations must coalesce AND serialize.

    Mirrors the real race: the signal-handler task awaits _cleanup_once while
    run()'s finally block awaits it. The second caller must wait for the
    first's engine.cleanup() to finish before returning — otherwise the
    signal handler can call runtime.shutdown() while the finally-block's
    cleanup is still mid-flight.
    """
    worker, engine = _make_worker()

    async def _run():
        await worker._start_engine()

        started = asyncio.Event()
        release = asyncio.Event()

        async def slow_cleanup():
            started.set()
            await release.wait()

        engine.cleanup.side_effect = slow_cleanup

        first = asyncio.create_task(worker._cleanup_once())
        await started.wait()
        # First caller is now suspended inside engine.cleanup(). A second
        # caller arriving here must NOT return until the first finishes.
        second = asyncio.create_task(worker._cleanup_once())
        # Yield enough times that a flag-only short-circuit would let `second`
        # complete while `first` is still inside engine.cleanup().
        for _ in range(10):
            await asyncio.sleep(0)
        assert not second.done(), (
            "second _cleanup_once returned while the first was still inside "
            "engine.cleanup() — late callers must wait for the in-flight cleanup"
        )

        release.set()
        await asyncio.gather(first, second)

    asyncio.run(_run())
    assert engine.cleanup.await_count == 1


def test_cleanup_after_failed_start_does_not_call_engine_cleanup():
    """If engine.start() raises, engine.cleanup() must not be called.

    The engine never reached a clean RUNNING state, so cleaning up is unsafe
    (handles may be half-built or never assigned).
    """
    worker, engine = _make_worker()
    engine.start.side_effect = RuntimeError("boom")

    async def _run():
        with pytest.raises(RuntimeError, match="boom"):
            await worker._start_engine()
        # Both shutdown paths are still safe to invoke; both must no-op.
        await worker._cleanup_once()
        await worker._cleanup_once()

    asyncio.run(_run())
    assert engine.start.await_count == 1
    assert engine.cleanup.await_count == 0


def test_cleanup_before_start_skips_engine_cleanup():
    """SIGTERM/SIGINT before _start_engine: engine.cleanup() must not run.

    The signal handler can fire any time after install_signal_handlers(),
    which Worker.run() calls before engine.start(). If the user kills the
    process during that window, engine.cleanup() would be called on an engine
    whose start() never ran.
    """
    worker, engine = _make_worker()

    async def _run():
        await worker._cleanup_once()

    asyncio.run(_run())
    assert engine.start.await_count == 0
    assert engine.cleanup.await_count == 0


def test_start_after_cleanup_signal_aborts_with_engine_shutdown():
    """If cleanup transitioned us to STOPPED before _start_engine ran,
    _start_engine must abort cleanly with EngineShutdown rather than starting
    an engine the worker is already shutting down.
    """
    worker, engine = _make_worker()

    async def _run():
        await worker._cleanup_once()
        with pytest.raises(EngineShutdown):
            await worker._start_engine()

    asyncio.run(_run())
    assert engine.start.await_count == 0
    assert engine.cleanup.await_count == 0


@pytest.mark.timeout(5)
def test_cleanup_during_start_waits_for_start_to_finish():
    """Cleanup arriving while engine.start() is in flight must serialize:
    engine.cleanup() must not run until engine.start() returns.
    """
    worker, engine = _make_worker()

    start_began = asyncio.Event()
    release_start = asyncio.Event()
    call_order: list[str] = []

    async def slow_start():
        call_order.append("start_begin")
        start_began.set()
        await release_start.wait()
        call_order.append("start_end")
        return MagicMock()

    async def record_cleanup():
        call_order.append("cleanup")

    engine.start.side_effect = slow_start
    engine.cleanup.side_effect = record_cleanup

    async def _run():
        start_task = asyncio.create_task(worker._start_engine())
        await start_began.wait()
        cleanup_task = asyncio.create_task(worker._cleanup_once())
        # Yield enough times to confirm cleanup is blocked on the lifecycle
        # lock — engine.cleanup() must not run while engine.start() is
        # mid-flight.
        for _ in range(10):
            await asyncio.sleep(0)
        assert not cleanup_task.done(), (
            "cleanup completed while start was still in flight — the "
            "lifecycle lock did not serialize start vs cleanup"
        )
        release_start.set()
        await asyncio.gather(start_task, cleanup_task)

    asyncio.run(_run())
    assert call_order == ["start_begin", "start_end", "cleanup"], call_order
    assert engine.start.await_count == 1
    assert engine.cleanup.await_count == 1


@pytest.mark.timeout(5)
def test_cleanup_during_failed_start_does_not_call_engine_cleanup():
    """Cleanup arriving while engine.start() is in flight, and start() then
    raises: cleanup must wait for start to finish, observe the failure, and
    skip engine.cleanup().
    """
    worker, engine = _make_worker()

    start_began = asyncio.Event()
    release_start = asyncio.Event()

    async def failing_start():
        start_began.set()
        await release_start.wait()
        raise RuntimeError("start blew up")

    engine.start.side_effect = failing_start

    async def _run():
        start_task = asyncio.create_task(worker._start_engine())
        await start_began.wait()
        cleanup_task = asyncio.create_task(worker._cleanup_once())
        for _ in range(10):
            await asyncio.sleep(0)
        assert not cleanup_task.done(), "cleanup ran before start finished"
        release_start.set()
        with pytest.raises(RuntimeError, match="start blew up"):
            await start_task
        await cleanup_task

    asyncio.run(_run())
    assert engine.cleanup.await_count == 0


def test_cleanup_propagates_exception_but_marks_done():
    """If engine.cleanup() raises, the worker is still marked stopped so a
    follow-up invocation from the other shutdown path is a no-op rather than
    a retry.
    """
    worker, engine = _make_worker()
    engine.cleanup.side_effect = RuntimeError("boom")

    async def _run():
        await worker._start_engine()
        with pytest.raises(RuntimeError, match="boom"):
            await worker._cleanup_once()
        # Second call must not re-invoke cleanup, even though the first raised.
        await worker._cleanup_once()

    asyncio.run(_run())
    assert engine.cleanup.await_count == 1
