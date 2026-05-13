# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
import signal
from typing import Any, Callable, Coroutine, Iterable, Optional

from dynamo._core import DistributedRuntime

logger = logging.getLogger(__name__)

# TODO: make this using cli flag
_DEFAULT_GRACE_PERIOD_SECS = 5.0
_DEFAULT_DRAIN_TIMEOUT_SECS = 30.0
_GRACE_PERIOD_ENV = "DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS"
_shutdown_started = asyncio.Event()


def get_grace_period_seconds() -> float:
    value = os.getenv(_GRACE_PERIOD_ENV)
    if value is None or value == "":
        return _DEFAULT_GRACE_PERIOD_SECS
    try:
        parsed = float(value)
    except ValueError:
        logger.warning(
            "Invalid %s=%r; using default %s",
            _GRACE_PERIOD_ENV,
            value,
            _DEFAULT_GRACE_PERIOD_SECS,
        )
        return _DEFAULT_GRACE_PERIOD_SECS
    if parsed < 0:
        logger.warning(
            "Negative %s=%r; using 0",
            _GRACE_PERIOD_ENV,
            value,
        )
        return 0.0
    return parsed


async def _unregister_endpoints(endpoints: Iterable) -> None:
    seen = set()
    tasks = []
    for endpoint in endpoints:
        endpoint_id = id(endpoint)
        if endpoint_id in seen:
            continue
        seen.add(endpoint_id)
        tasks.append(endpoint.unregister_endpoint_instance())

    if not tasks:
        return

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results:
        if isinstance(result, Exception):
            logger.warning(
                "Failed to unregister endpoint instance from discovery: %s",
                result,
            )


async def graceful_shutdown_with_discovery(
    runtime: DistributedRuntime,
    endpoints: Iterable,
    shutdown_event: Optional[asyncio.Event] = None,
    grace_period_s: Optional[float] = None,
    drain_callback: Optional[Callable[[], Coroutine]] = None,
) -> None:
    """Perform graceful shutdown with endpoint unregistration and optional drain.

    Args:
        runtime: The distributed runtime to shut down.
        endpoints: Endpoints to unregister from discovery before shutdown.
        shutdown_event: Optional event to set before calling runtime.shutdown().
        grace_period_s: Seconds to wait after unregistering before drain/shutdown.
            Defaults to DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS env var or 5s.
        drain_callback: Optional async callable awaited after the grace period
            but *before* runtime.shutdown(). Use this on prefill workers to wait
            for in-flight NIXL KV transfers to complete, preventing decode workers
            from segfaulting due to use-after-free on freed GPU memory (#7319).
            Any exception raised by drain_callback is logged and swallowed so that
            shutdown still proceeds even if draining times out or fails.
    """
    if _shutdown_started.is_set():
        return
    _shutdown_started.set()

    if grace_period_s is None:
        grace_period_s = get_grace_period_seconds()

    logger.info("Received shutdown signal; unregistering endpoints from discovery")
    await _unregister_endpoints(list(endpoints))

    if grace_period_s > 0:
        logger.info("Grace period %.2fs before stopping endpoints", grace_period_s)
        await asyncio.sleep(grace_period_s)

    if drain_callback is not None:
        logger.info(
            "Draining in-flight transfers before shutdown (issue #7319 safeguard)"
        )
        try:
            await asyncio.wait_for(
                drain_callback(), timeout=_DEFAULT_DRAIN_TIMEOUT_SECS
            )
            logger.info("Drain complete")
        except asyncio.TimeoutError:
            logger.warning(
                "Drain callback timed out after %.0fs, proceeding with shutdown",
                _DEFAULT_DRAIN_TIMEOUT_SECS,
            )
        except Exception:
            logger.exception(
                "Drain callback raised an exception; proceeding with shutdown"
            )

    if shutdown_event is not None:
        shutdown_event.set()

    logger.info("Initiating runtime shutdown")
    runtime.shutdown()


def install_signal_handlers(
    loop: asyncio.AbstractEventLoop,
    runtime: Any,
    endpoints: Iterable,
    shutdown_event: Optional[asyncio.Event] = None,
    grace_period_s: Optional[float] = None,
    drain_callback: Optional[Callable[[], Coroutine]] = None,
) -> None:
    shutdown_task: Optional[asyncio.Task[None]] = None

    def _on_shutdown_done(task: asyncio.Task[None]) -> None:
        nonlocal shutdown_task
        try:
            task.result()
        except asyncio.CancelledError:
            logger.info("Graceful shutdown task cancelled")
        except Exception:
            logger.exception("Graceful shutdown task failed")
        finally:
            if shutdown_task is task:
                shutdown_task = None

    def signal_handler() -> None:
        nonlocal shutdown_task
        if shutdown_task is not None and not shutdown_task.done():
            logger.debug("Shutdown already in progress; ignoring duplicate signal")
            return

        shutdown_task = asyncio.create_task(
            graceful_shutdown_with_discovery(
                runtime,
                endpoints,
                shutdown_event=shutdown_event,
                grace_period_s=grace_period_s,
                drain_callback=drain_callback,
            )
        )
        shutdown_task.add_done_callback(_on_shutdown_done)

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    logger.info(
        "Signal handlers set up for graceful shutdown "
        "(discovery unregister + grace period)"
    )
