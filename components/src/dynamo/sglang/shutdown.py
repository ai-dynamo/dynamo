# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import inspect
import logging
import signal
from collections import defaultdict
from typing import Any, Awaitable, Callable, DefaultDict

from dynamo.common.utils.graceful_shutdown import graceful_shutdown_with_discovery

SignalCallback = Callable[..., Any]


def install_graceful_shutdown(
    loop: asyncio.AbstractEventLoop,
    runtime: Any,
    endpoints: list,
    shutdown_event: asyncio.Event,
    *,
    signals: tuple[int, ...] = (signal.SIGTERM, signal.SIGINT),
) -> Callable[[], Awaitable[None]]:
    """
    Set up graceful shutdown with discovery unregister and grace period.

    Owns OS-level SIGTERM/SIGINT via signal.signal() so SGLang's internal
    loop.add_signal_handler registrations cannot replace our handler.
    Monkey-patches loop.add_signal_handler to capture (defer) those
    registrations. Returns run_deferred_handlers to be invoked in init
    finally blocks (after the asyncio loop / serve_endpoint is done).
    """
    deferred_handlers: DefaultDict[
        int, list[tuple[SignalCallback, tuple[Any, ...]]]
    ] = defaultdict(list)  # type: ignore[assignment]

    shutdown_started = False
    shutdown_signum: int | None = None
    deferred_handlers_ran = False

    async def run_deferred_handlers() -> None:
        nonlocal deferred_handlers_ran
        if not shutdown_started or deferred_handlers_ran:
            return
        deferred_handlers_ran = True

        signums = (
            [shutdown_signum]
            if shutdown_signum is not None
            else list(deferred_handlers.keys())
        )
        for sig in signums:
            for cb, args in list(deferred_handlers.get(sig, [])):
                try:
                    res = cb(*args)
                    if inspect.isawaitable(res):
                        await res
                except Exception:
                    logging.exception("Deferred signal callback failed: %r", cb)

    async def _shutdown_sequence(signum: int, frame: Any | None) -> None:
        nonlocal shutdown_started, shutdown_signum
        if shutdown_started:
            return
        shutdown_signum = signum
        shutdown_started = True

        logging.info("Received signal %s, starting graceful shutdown", signum)
        await graceful_shutdown_with_discovery(
            runtime,
            endpoints,
            shutdown_event=shutdown_event,
            grace_period_s=None,
        )

    def _schedule_shutdown(signum: int, frame: Any | None) -> None:
        def _kick() -> None:
            asyncio.create_task(_shutdown_sequence(signum, frame))

        loop.call_soon_threadsafe(_kick)

    def _os_signal_handler(signum: int, frame: Any) -> None:
        _schedule_shutdown(signum, frame)

    for sig in signals:
        signal.signal(sig, _os_signal_handler)

    orig_add = loop.add_signal_handler

    def watching_add_signal_handler(sig: int, callback: SignalCallback, *args: Any):
        if sig in signals:
            logging.debug(
                "Captured underlying service trying to register for loop.add_signal_handler(%s, %r, ...).",
                sig,
                callback,
            )
            deferred_handlers[sig].append((callback, args))
            return None
        return orig_add(sig, callback, *args)

    loop.add_signal_handler = watching_add_signal_handler  # type: ignore[assignment]

    return run_deferred_handlers
