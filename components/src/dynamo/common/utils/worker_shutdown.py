# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ordered shutdown for Python workers that own their engine in the serve task."""

import asyncio
import logging
import math
import os
import signal
import time
from collections.abc import Awaitable, Callable, Coroutine
from contextlib import aclosing
from functools import wraps
from typing import Any

from dynamo._core import ShutdownWatchdog, WorkerDraining, worker_shutdown_timeout_secs
from dynamo.common.utils.graceful_shutdown import _unregister_endpoints

logger = logging.getLogger(__name__)
_CLEANUP_FLOOR = 5.0
_MAX_SECONDS = 315_360_000.0


def _seconds(name: str, default: float, *, positive: bool = False) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not math.isfinite(value) or value > _MAX_SECONDS or (positive and value <= 0):
        logger.warning("Invalid %s=%r; using %s", name, raw, default)
        return default
    return max(0.0, value)


class WorkerShutdown:
    """Own admission, the shutdown deadline, and the task that owns the engine.

    The serve task's finally blocks release engine resources. Cancelling that
    task is therefore the cleanup stage, after requests and KV transfers drain.
    """

    def __init__(
        self,
        runtime: Any,
        endpoints: list,
        shutdown_event: asyncio.Event,
        *,
        prefill: bool = False,
        pre_shutdown: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.runtime = runtime
        self.endpoints = endpoints
        self.shutdown_event = shutdown_event
        self.prefill = prefill
        self.pre_shutdown = pre_shutdown
        self.post_shutdown: Callable[[], Awaitable[None]] | None = None
        # Gateway parents delegate discovery and request draining to children.
        self.notify_children: Callable[[], None] | None = None
        self.wait_for_children: Callable[[], Awaitable[None]] | None = None
        self.accepting = True
        self.inflight = 0
        self.idle = asyncio.Event()
        self.idle.set()
        self._worker: asyncio.Task | None = None
        self._sequence: asyncio.Task | None = None
        self._watchdog: ShutdownWatchdog | None = None
        self._deadline = 0.0
        self._hard_deadline = 0.0
        self._started = False
        self._requested = asyncio.Event()
        self._caps: dict[str, float] = {}

    def wrap(self, handler: Callable) -> Callable:
        """Keep admission and tracking outside the backend generator's body."""

        # Rust inspects the original signature through __wrapped__, retaining
        # context injection and TRT-LLM's response_sender transport selection.
        @wraps(handler)
        async def tracked(*args, **kwargs):
            if not self.accepting:
                # Raised on the first stream poll, preserving the typed rejection
                # through both pull and push response transports.
                raise WorkerDraining("worker is not accepting new requests")
            self.inflight += 1
            self.idle.clear()
            try:
                async with aclosing(handler(*args, **kwargs)) as stream:
                    async for item in stream:
                        yield item
            finally:
                self.inflight -= 1
                if self.inflight == 0:
                    self.idle.set()

        return tracked

    def request_shutdown(self, signum: int = signal.SIGTERM) -> None:
        """Arm the watchdog in the signal handler, before scheduling any async work."""
        if self._started:
            os._exit(70)
        self._started = True
        origin = time.monotonic()
        grace = _seconds("DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS", 5.0)
        timeout = float(worker_shutdown_timeout_secs())
        total = min(timeout, _MAX_SECONDS) + grace
        self._caps = {
            "router_grace": grace,
            "inflight": _seconds("DYN_WORKER_SHUTDOWN_INFLIGHT_TIMEOUT_SECS", math.inf),
            "kv_transfer": _seconds("DYN_PREFILL_DRAIN_TIMEOUT_S", 30.0),
            "cleanup": _seconds(
                "DYN_WORKER_SHUTDOWN_CLEANUP_TIMEOUT_SECS", timeout, positive=True
            ),
        }
        self._deadline = origin + total
        self._hard_deadline = self._deadline + _CLEANUP_FLOOR
        self._watchdog = ShutdownWatchdog(
            max(0.0, self._hard_deadline - time.monotonic())
        )
        logger.info("Received signal %s; starting worker shutdown", signum)
        self._requested.set()
        # A synchronous signal handler can interrupt an idle selector, which
        # Python then retries. Write the loop's wakeup fd so shutdown runs now.
        asyncio.get_running_loop().call_soon_threadsafe(self._start_sequence)

    def _start_sequence(self) -> None:
        if self._sequence is None:
            self._sequence = asyncio.create_task(self._shutdown())

    def _remaining(self) -> float:
        return max(0.0, self._deadline - time.monotonic())

    async def _stage(self, name: str, awaitable: Awaitable, seconds: float) -> bool:
        start = time.monotonic()
        task = asyncio.ensure_future(awaitable)
        # Unlike wait_for, wait does not extend the deadline when cancellation
        # is ignored. The OS-thread watchdog bounds blocked Python cleanup too.
        done, _ = await asyncio.wait({task}, timeout=seconds)
        if not done:
            task.cancel()
            task.add_done_callback(self._consume_result)
            logger.warning(
                "shutdown stage=%s reason=timed_out elapsed_s=%.3f remaining_total_s=%.3f",
                name,
                time.monotonic() - start,
                self._remaining(),
            )
            return False
        try:
            task.result()
        except asyncio.CancelledError:
            raise
        except Exception:
            # Continue to later teardown stages; cleanup/runtime failures become
            # a terminal error in _shutdown rather than abandoning resources here.
            logger.exception("shutdown stage=%s failed", name)
            return False
        logger.info(
            "shutdown stage=%s reason=completed elapsed_s=%.3f remaining_total_s=%.3f",
            name,
            time.monotonic() - start,
            self._remaining(),
        )
        return True

    @staticmethod
    def _consume_result(task: asyncio.Future) -> None:
        if not task.cancelled():
            task.exception()

    async def _cleanup_worker(self) -> None:
        assert self._worker is not None
        if not self._worker.done():
            self._worker.cancel()
        try:
            await asyncio.shield(self._worker)
        except asyncio.CancelledError:
            if not self._worker.cancelled():
                raise

    async def _shutdown(self) -> None:
        if self.notify_children is not None:
            self.notify_children()
        await self._stage(
            "unregister", _unregister_endpoints(list(self.endpoints)), self._remaining()
        )
        grace = min(self._caps["router_grace"], self._remaining())
        if grace:
            await self._stage("router_grace", asyncio.sleep(grace), self._remaining())
        self.accepting = False
        logger.info("shutdown stage=stop_admission reason=completed")

        async def inflight():
            await self.idle.wait()
            if self.wait_for_children is not None:
                await self.wait_for_children()

        await self._stage(
            "inflight", inflight(), min(self._caps["inflight"], self._remaining())
        )

        # Request completion is not evidence of remote KV-read completion.
        # These engines declare the same conservative fallback as the sidecars.
        fallback = (
            os.environ.get("DYN_WORKER_SHUTDOWN_KV_TRANSFER_FALLBACK", "wait")
            .strip()
            .lower()
        )
        if fallback not in ("wait", "skip", ""):
            logger.warning("Invalid KV-transfer fallback %r; using wait", fallback)
            fallback = "wait"
        if self.prefill and fallback != "skip":
            allowance = min(self._caps["kv_transfer"], self._remaining())
            logger.info(
                "shutdown stage=kv_transfer reason=unsupported fallback=wait timeout_s=%.3f",
                allowance,
            )
            await asyncio.sleep(allowance)
        else:
            logger.info("shutdown stage=kv_transfer reason=skipped")

        if self.pre_shutdown is not None:
            await self._stage("withdraw", self.pre_shutdown(), self._remaining())
        self.shutdown_event.set()
        # Match cleanup_once: the remaining total, not a short explicit cap,
        # determines whether cleanup needs its reserve.
        remaining = self._remaining()
        cleanup_budget = (
            _CLEANUP_FLOOR
            if remaining < _CLEANUP_FLOOR
            else min(self._caps["cleanup"], remaining)
        )
        clean = await self._stage("cleanup", self._cleanup_worker(), cleanup_budget)
        runtime_clean = await self._stage(
            "runtime",
            self.runtime.shutdown_and_wait(),
            max(0.0, self._hard_deadline - time.monotonic()),
        )
        if clean and runtime_clean and self.post_shutdown is not None:
            runtime_clean = (
                await self._stage(
                    "deferred_signals",
                    self.post_shutdown(),
                    max(0.0, self._hard_deadline - time.monotonic()),
                )
                and runtime_clean
            )
        if clean and runtime_clean:
            assert self._watchdog is not None
            self._watchdog.finish()
        else:
            raise RuntimeError("worker shutdown did not complete")

    async def run(self, worker: Coroutine, *, install_signals: bool = True) -> None:
        """Run the engine owner and join the complete sequence before returning."""
        previous = {}
        if install_signals:
            for sig in (signal.SIGTERM, signal.SIGINT):
                previous[sig] = signal.signal(
                    sig, lambda signum, _frame: self.request_shutdown(signum)
                )
        self._worker = asyncio.create_task(worker)
        requested = asyncio.create_task(self._requested.wait())
        try:
            try:
                await asyncio.wait(
                    {self._worker, requested}, return_when=asyncio.FIRST_COMPLETED
                )
                if self._worker.done():
                    self._worker.result()
            except asyncio.CancelledError:
                if not self._started:
                    self.request_shutdown()
                    raise
            finally:
                if not self._started:
                    self.request_shutdown()
                self._start_sequence()
                assert self._sequence is not None
                await asyncio.shield(self._sequence)
        finally:
            requested.cancel()
            await asyncio.gather(requested, return_exceptions=True)
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def serve_endpoint(
    endpoint,
    handler,
    *,
    shutdown: WorkerShutdown | None,
    bidirectional: bool = False,
    **kwargs,
):
    """Register a handler with the worker's admission gate, retaining push egress."""
    if shutdown is not None:
        handler = shutdown.wrap(handler)
    if bidirectional:
        return endpoint.serve_bidirectional_endpoint(handler, **kwargs)
    return endpoint.serve_endpoint(handler, **kwargs)
