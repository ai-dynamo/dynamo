# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import signal
from unittest.mock import Mock

import pytest

from dynamo.sglang import shutdown

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


@pytest.mark.asyncio
async def test_sigterm_fences_resume_before_existing_shutdown_grace(monkeypatch):
    loop = asyncio.get_running_loop()
    handlers = {}
    entered, release, completed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    terminal_event, shutdown_event = asyncio.Event(), asyncio.Event()

    def install(signum, callback):
        handlers[signum] = callback

    async def sequence(runtime, endpoints, *, shutdown_event, grace_period_s):
        assert terminal_event.is_set()
        assert not shutdown_event.is_set()
        entered.set()
        await release.wait()
        shutdown_event.set()
        completed.set()

    monkeypatch.setattr(signal, "signal", install)
    monkeypatch.setattr(shutdown, "graceful_shutdown_with_discovery", sequence)
    # Restore the loop's method after the production installer wraps it.
    monkeypatch.setattr(loop, "add_signal_handler", loop.add_signal_handler)
    deferred = shutdown.install_graceful_shutdown(
        loop,
        Mock(),
        [],
        shutdown_event,
        signals=(signal.SIGTERM,),
        terminal_event=terminal_event,
    )
    handlers[signal.SIGTERM](signal.SIGTERM, None)
    await asyncio.wait_for(entered.wait(), 1)
    assert terminal_event.is_set()
    assert not shutdown_event.is_set()
    release.set()
    await asyncio.wait_for(completed.wait(), 1)
    await deferred()
