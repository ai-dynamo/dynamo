# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import signal
from unittest.mock import Mock

import pytest

from dynamo.sglang import shutdown

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.timeout(10),
]


@pytest.mark.parametrize("back_to_back", [False, True])
def test_repeated_signals_join_original_teardown(monkeypatch, back_to_back):
    # Regression: replacing the first task lets deferred engine cleanup run
    # while runtime teardown is still pending.
    async def scenario():
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        release = asyncio.Event()
        handlers = {}
        engine_cleanup = Mock()

        async def teardown(*args, **kwargs):
            started.set()
            await release.wait()

        monkeypatch.setattr(shutdown, "graceful_shutdown_with_discovery", teardown)
        monkeypatch.setattr(
            signal, "signal", lambda sig, callback: handlers.update({sig: callback})
        )
        # Restore the instance attribute patched by the production helper.
        monkeypatch.setattr(loop, "add_signal_handler", loop.add_signal_handler)
        deferred = shutdown.install_graceful_shutdown(loop, Mock(), [], asyncio.Event())
        loop.add_signal_handler(signal.SIGTERM, engine_cleanup)
        handlers[signal.SIGTERM](signal.SIGTERM, None)
        if not back_to_back:
            await started.wait()
        handlers[signal.SIGINT](signal.SIGINT, None)
        await started.wait()
        await asyncio.sleep(0)
        joined = asyncio.create_task(deferred())
        try:
            # Let the queued signal task and joiner reach their awaits.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert not joined.done()
            engine_cleanup.assert_not_called()
        finally:
            release.set()
            await joined
        engine_cleanup.assert_called_once()

    asyncio.run(scenario())


@pytest.mark.parametrize("application_failed", [False, True])
def test_teardown_failure_survives_deferred_cleanup(monkeypatch, application_failed):
    # Regression: failed runtime teardown must not report success or replace an
    # existing application failure, and required engine cleanup must still run.
    async def scenario():
        loop = asyncio.get_running_loop()
        handlers = {}
        started = asyncio.Event()
        engine_cleanup = Mock()
        teardown_error = RuntimeError("runtime teardown incomplete")
        application_error = ValueError("application failed")

        async def teardown(*args, **kwargs):
            started.set()
            raise teardown_error

        monkeypatch.setattr(shutdown, "graceful_shutdown_with_discovery", teardown)
        monkeypatch.setattr(
            signal, "signal", lambda sig, callback: handlers.update({sig: callback})
        )
        monkeypatch.setattr(loop, "add_signal_handler", loop.add_signal_handler)
        deferred = shutdown.install_graceful_shutdown(loop, Mock(), [], asyncio.Event())
        loop.add_signal_handler(signal.SIGTERM, engine_cleanup)
        handlers[signal.SIGTERM](signal.SIGTERM, None)
        await started.wait()
        expected = application_error if application_failed else teardown_error
        with pytest.raises(type(expected)) as caught:
            try:
                if application_failed:
                    raise application_error
            finally:
                await deferred()
        assert caught.value is expected
        engine_cleanup.assert_called_once()

    asyncio.run(scenario())
