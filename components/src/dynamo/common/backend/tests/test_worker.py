# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Worker._cleanup_once.

Locks down the invariant that engine.cleanup() runs exactly once even when
both shutdown paths fire (graceful-shutdown signal handler and run()'s
finally block).
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from dynamo.common.backend.worker import Worker, WorkerConfig

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def _make_worker() -> tuple:
    engine = AsyncMock()
    engine.cleanup = AsyncMock()
    config = WorkerConfig(namespace="test")
    return Worker(engine=engine, config=config), engine


def test_cleanup_once_calls_engine_cleanup_exactly_once():
    """Both shutdown paths racing must not double-cleanup the engine.

    The signal-handler path and run()'s finally block both call _cleanup_once.
    Engines like vLLM and TRT-LLM tear down NCCL process groups in cleanup();
    calling it twice can hang or raise.
    """
    worker, engine = _make_worker()

    async def _run():
        await worker._cleanup_once()
        await worker._cleanup_once()
        await worker._cleanup_once()

    asyncio.run(_run())
    assert engine.cleanup.await_count == 1


def test_cleanup_once_concurrent_invocations_only_run_once():
    """Concurrent _cleanup_once invocations must coalesce AND serialize.

    Mirrors the real race: the signal-handler task awaits _cleanup_once while
    run()'s finally block awaits it. The second caller must wait for the
    first's engine.cleanup() to finish before returning — otherwise the
    signal handler can call runtime.shutdown() while the finally-block's
    cleanup is still mid-flight, collapsing the loop under it.
    """
    worker, engine = _make_worker()

    async def _run():
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


def test_cleanup_once_propagates_exception_but_marks_done():
    """If engine.cleanup() raises, _cleanup_done is still set so a follow-up
    invocation from the other shutdown path is a no-op rather than a retry.
    """
    worker, engine = _make_worker()
    engine.cleanup.side_effect = RuntimeError("boom")

    async def _run():
        with pytest.raises(RuntimeError, match="boom"):
            await worker._cleanup_once()
        # Second call must not re-invoke cleanup, even though the first raised.
        await worker._cleanup_once()

    asyncio.run(_run())
    assert engine.cleanup.await_count == 1
