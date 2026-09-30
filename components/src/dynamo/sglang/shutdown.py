# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import inspect
import signal
from collections import defaultdict
from contextlib import contextmanager

from dynamo.common.utils.worker_shutdown import WorkerShutdown


@contextmanager
def defer_engine_signals(loop: asyncio.AbstractEventLoop, shutdown: WorkerShutdown):
    """Keep engine signal callbacks behind engine cleanup and runtime teardown."""
    signals = (signal.SIGTERM, signal.SIGINT)
    callbacks: dict[int, list] = defaultdict(list)
    previous = {}
    signum = None
    original_add = loop.add_signal_handler

    def on_signal(sig, _frame):
        nonlocal signum
        signum = sig
        shutdown.request_shutdown(sig)

    def capture(sig, callback, *args):
        if sig in signals:
            callbacks[sig].append((callback, args))
        else:
            original_add(sig, callback, *args)

    async def run_deferred():
        for callback, args in callbacks.get(signum, []):
            result = callback(*args)
            if inspect.isawaitable(result):
                await result

    for sig in signals:
        previous[sig] = signal.signal(sig, on_signal)
    loop.add_signal_handler = capture
    try:
        yield run_deferred
    finally:
        loop.add_signal_handler = original_add
        for sig, handler in previous.items():
            signal.signal(sig, handler)
