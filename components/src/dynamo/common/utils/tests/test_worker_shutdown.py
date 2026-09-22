# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shutdown contracts shared by the three Python engine entry points."""

import asyncio
import inspect
import subprocess
import sys
from unittest.mock import Mock

import pytest

from dynamo.common.utils import worker_shutdown as ws
from dynamo.sglang.shutdown import defer_engine_signals

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.timeout(15),
]


@pytest.fixture
def shutdown_env(monkeypatch):
    monkeypatch.setenv("DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS", "0")
    monkeypatch.setenv("DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT", "2")
    monkeypatch.setenv("DYN_WORKER_SHUTDOWN_KV_TRANSFER_FALLBACK", "skip")
    for name in (
        "DYN_WORKER_SHUTDOWN_INFLIGHT_TIMEOUT_SECS",
        "DYN_WORKER_SHUTDOWN_CLEANUP_TIMEOUT_SECS",
        "DYN_PREFILL_DRAIN_TIMEOUT_S",
    ):
        monkeypatch.delenv(name, raising=False)
    watchdog = Mock()
    monkeypatch.setattr(ws, "ShutdownWatchdog", Mock(return_value=watchdog))
    return watchdog


def test_shutdown_drains_admitted_stream_before_engine_and_runtime(shutdown_env):
    # Regression: setting the backend shutdown event too early aborts an admitted
    # request and frees engine resources before the response stream completes.
    async def run():
        events = []
        started = asyncio.Event()
        unregistered = asyncio.Event()
        release = asyncio.Event()
        stop = asyncio.Event()

        class Runtime:
            async def shutdown_and_wait(self):
                assert events[-1] == "engine"
                events.append("runtime")

        class Endpoint:
            async def unregister_endpoint_instance(self):
                events.append("unregister")
                unregistered.set()

        shutdown = ws.WorkerShutdown(Runtime(), [Endpoint()], stop)

        async def generate(request, context):
            events.append("request")
            yield 1
            await release.wait()
            assert not stop.is_set()
            yield 2
            events.append("response_done")

        async def worker():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                assert events[-1] == "response_done"
                assert stop.is_set()
                events.append("engine")

        owner = asyncio.create_task(shutdown.run(worker(), install_signals=False))
        await started.wait()
        stream = shutdown.wrap(generate)({}, None)
        assert await anext(stream) == 1
        shutdown.request_shutdown()
        await unregistered.wait()
        while shutdown.accepting:
            await asyncio.sleep(0)
        with pytest.raises(ws.WorkerDraining):
            await anext(shutdown.wrap(generate)({}, None))
        assert not stop.is_set()
        release.set()
        assert [item async for item in stream] == [2]
        await owner
        assert events == ["request", "unregister", "response_done", "engine", "runtime"]
        shutdown_env.finish.assert_called_once()

    asyncio.run(run())


def test_prefill_cleanup_floor(shutdown_env, monkeypatch):
    # Regression: the KV stage can spend the entire total, but cleanup must
    # still run.
    monkeypatch.setenv("DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT", "0")
    monkeypatch.setenv("DYN_WORKER_SHUTDOWN_KV_TRANSFER_FALLBACK", "wait")

    async def run():
        cleaned = asyncio.Event()
        started = asyncio.Event()

        class Runtime:
            async def shutdown_and_wait(self):
                assert cleaned.is_set()

        shutdown = ws.WorkerShutdown(Runtime(), [], asyncio.Event(), prefill=True)

        async def worker():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                cleaned.set()

        owner = asyncio.create_task(shutdown.run(worker(), install_signals=False))
        await started.wait()
        shutdown.request_shutdown()
        await owner
        assert cleaned.is_set()
        shutdown_env.finish.assert_called_once()

    asyncio.run(run())


def test_cleanup_timeout_is_failure_and_does_not_extend_stage(
    shutdown_env, monkeypatch
):
    # Regression: cancellation-resistant cleanup must not extend a configured
    # stage cap or allow a successful exit after abandoning engine cleanup.
    monkeypatch.setenv("DYN_WORKER_SHUTDOWN_CLEANUP_TIMEOUT_SECS", "0.01")
    monkeypatch.setenv("DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT", "10")

    async def run():
        started = asyncio.Event()
        finish = asyncio.Event()

        class Runtime:
            async def shutdown_and_wait(self):
                pass

        shutdown = ws.WorkerShutdown(Runtime(), [], asyncio.Event())

        async def worker():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                await finish.wait()

        owner = asyncio.create_task(shutdown.run(worker(), install_signals=False))
        await started.wait()
        shutdown.request_shutdown()
        try:
            # run() must observe a failed sequence even while the worker's
            # finally block is still pending.
            with pytest.raises(RuntimeError, match="shutdown did not complete"):
                await asyncio.wait_for(owner, timeout=1)
            shutdown_env.finish.assert_not_called()
        finally:
            finish.set()
            await asyncio.gather(shutdown._worker, return_exceptions=True)

    asyncio.run(run())


def test_push_handler_keeps_sender_and_releases_admission_on_close(shutdown_env):
    # Regression: hiding response_sender routes TRT-LLM through the wrong
    # transport; dropping a stream must also release its in-flight slot.
    async def run():
        shutdown = ws.WorkerShutdown(None, [], asyncio.Event())
        sender = object()
        closed = asyncio.Event()

        async def generate(request, context=None, response_sender=None):
            assert response_sender is sender
            try:
                yield 1
                yield 2
            finally:
                closed.set()

        wrapped = shutdown.wrap(generate)
        assert "response_sender" in inspect.signature(wrapped).parameters
        stream = wrapped({}, response_sender=sender)
        assert await anext(stream) == 1
        assert not shutdown.idle.is_set()
        await stream.aclose()
        assert closed.is_set()
        assert shutdown.idle.is_set()

    asyncio.run(run())


def test_native_watchdog_exits_even_when_python_holds_gil():
    # Regression: Python and asyncio timers cannot rescue a thread holding the
    # GIL inside engine cleanup. Run that failure in an isolated process.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from dynamo._core import ShutdownWatchdog; import ctypes; watchdog = ShutdownWatchdog(0.05); ctypes.PyDLL(None).sleep(10)",
        ],
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 70, result.stderr.decode()


def test_sglang_deferred_signals_run_after_runtime_teardown(shutdown_env):
    # Regression: an engine-installed signal handler can tear down schedulers
    # before the Dynamo request drain or exit before runtime teardown finishes.
    async def run():
        events = []
        started = asyncio.Event()

        class Runtime:
            async def shutdown_and_wait(self):
                events.append("runtime")

        shutdown = ws.WorkerShutdown(Runtime(), [], asyncio.Event())
        loop = asyncio.get_running_loop()
        original_add = loop.add_signal_handler

        async def worker():
            loop.add_signal_handler(
                ws.signal.SIGTERM, lambda: events.append("engine_signal")
            )
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                events.append("engine_cleanup")

        with defer_engine_signals(loop, shutdown) as deferred:
            shutdown.post_shutdown = deferred
            owner = asyncio.create_task(shutdown.run(worker(), install_signals=False))
            await started.wait()
            ws.signal.getsignal(ws.signal.SIGTERM)(ws.signal.SIGTERM, None)
            await owner
        assert events == ["engine_cleanup", "runtime", "engine_signal"]
        assert loop.add_signal_handler == original_add

    asyncio.run(run())
