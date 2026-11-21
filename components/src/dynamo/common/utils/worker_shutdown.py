# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ordered shutdown for Python workers that own their engine in the serve task."""

import asyncio
import logging
import math
import os
import signal
import struct
import subprocess
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable, Coroutine
from contextlib import aclosing, contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps
from typing import Any

from dynamo._core import (
    ShutdownWatchdog,
    WorkerDraining,
    _ShutdownSignals,
    worker_shutdown_timeout_secs,
)
from dynamo.common.utils.graceful_shutdown import _unregister_endpoints

logger = logging.getLogger(__name__)
_DEFAULT_CLEANUP_SECONDS = 5.0
_MAX_SECONDS = 315_360_000.0
_CHILD_TOTAL_ENV = "_DYN_WORKER_SHUTDOWN_CHILD_TOTAL_SECS"
_child_deadline_fd: ContextVar[int | None] = ContextVar(
    "child_deadline_fd", default=None
)
_PARENT_DEADLINE_FD_ENV = "_DYN_WORKER_SHUTDOWN_PARENT_DEADLINE_FD"


def _seconds(
    name: str, default: float, *, positive: bool = False, alias: str | None = None
) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw and alias is not None:
        name = alias
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


def _total_timeout() -> float:
    total = min(float(worker_shutdown_timeout_secs()), _MAX_SECONDS)
    return min(total, _seconds(_CHILD_TOTAL_ENV, total))


def _parent_deadline_fd() -> int | None:
    value = os.environ.get(_PARENT_DEADLINE_FD_ENV)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        # Fail closed on receipt of a signal just as for a closed descriptor.
        return -1


def check_child_exit(child: subprocess.Popen) -> None:
    """Report a reaped child's failure without replacing a startup or serving error."""
    if child.returncode == 0:
        return
    message = f"worker child pid={child.pid} did not shut down cleanly (status={child.returncode})"
    error = sys.exception()
    if error is None or isinstance(error, asyncio.CancelledError):
        raise RuntimeError(message)
    error.add_note(message)


def child_shutdown_environment() -> dict[str, str]:
    """Leave the engine-owning parent its cleanup reserve after children exit."""
    total = _total_timeout()
    cleanup = _seconds(
        "DYN_WORKER_SHUTDOWN_CLEANUP_TIMEOUT_SECS",
        _DEFAULT_CLEANUP_SECONDS,
        positive=True,
    )
    env = {**os.environ, _CHILD_TOTAL_ENV: str(max(0.0, total - cleanup))}
    parent_fd = _child_deadline_fd.get()
    env.pop(_PARENT_DEADLINE_FD_ENV, None)
    if parent_fd is not None:
        env[_PARENT_DEADLINE_FD_ENV] = str(parent_fd)
    return env


def child_shutdown_fds() -> tuple[int, ...]:
    """Keep the parent's deadline descriptor open across child exec."""
    parent_fd = _child_deadline_fd.get()
    return (parent_fd,) if parent_fd is not None else ()


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
        self.finalize_worker: Callable[[], Awaitable[None]] | None = None
        self.post_shutdown: Callable[[], Awaitable[None]] | None = None
        self.wait_for_control_tasks: Callable[[], Awaitable[None]] | None = None
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
        self._signals: _ShutdownSignals | None = None
        self._deadline = 0.0
        self._hard_deadline = 0.0
        self._started = False
        self.requested_event = asyncio.Event()
        self._caps: dict[str, float] = {}
        self._metrics = None
        self._child_deadline_fd: int | None = None

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
        received = self._signals.received_at() if self._signals is not None else None
        origin = received if received is not None else time.monotonic()
        grace = _seconds(
            "DYN_WORKER_SHUTDOWN_ROUTER_GRACE_SECS",
            5.0,
            alias="DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS",
        )
        total = _total_timeout()
        parent_fd = _parent_deadline_fd()
        if parent_fd is not None:
            try:
                deadline = struct.unpack("d", os.pread(parent_fd, 8, 0))[0]
            except (OSError, ValueError, struct.error):
                logger.exception("Cannot read parent shutdown deadline")
                deadline = origin
            total = min(total, max(0.0, deadline - origin))
        self._caps = {
            "router_grace": grace,
            "inflight": _seconds("DYN_WORKER_SHUTDOWN_INFLIGHT_TIMEOUT_SECS", math.inf),
            "kv_transfer": _seconds(
                "DYN_WORKER_SHUTDOWN_KV_TRANSFER_TIMEOUT_SECS",
                30.0,
                alias="DYN_PREFILL_DRAIN_TIMEOUT_S",
            ),
            "cleanup": _seconds(
                "DYN_WORKER_SHUTDOWN_CLEANUP_TIMEOUT_SECS",
                _DEFAULT_CLEANUP_SECONDS,
                positive=True,
            ),
        }
        self._deadline = origin + total
        self._hard_deadline = self._deadline
        self._drain_deadline = self._deadline - min(self._caps["cleanup"], total)
        self._watchdog = ShutdownWatchdog(
            max(0.0, self._hard_deadline - time.monotonic())
        )
        if self._child_deadline_fd is not None:
            # Publish before any child is signalled. pread/pwrite do not share
            # file offsets, and the inherited anonymous file needs no pathname.
            os.pwrite(
                self._child_deadline_fd, struct.pack("d", self._drain_deadline), 0
            )
        logger.info("Received signal %s; starting worker shutdown", signum)
        self.requested_event.set()
        # A synchronous signal handler can interrupt an idle selector, which
        # Python then retries. Write the loop's wakeup fd so shutdown runs now.
        asyncio.get_running_loop().call_soon_threadsafe(self._start_sequence)

    def _start_sequence(self) -> None:
        if self._sequence is None:
            self._sequence = asyncio.create_task(self._shutdown())

    def _remaining(self) -> float:
        return max(0.0, self._deadline - time.monotonic())

    @property
    def started(self) -> bool:
        return self._started

    def cleanup_remaining(self) -> float:
        """Time still available for engine cleanup and runtime teardown."""
        return max(0.0, self._hard_deadline - time.monotonic())

    def _drain_remaining(self) -> float:
        return max(0.0, self._drain_deadline - time.monotonic())

    async def _stage(
        self,
        name: str,
        awaitable: Awaitable,
        seconds: float,
        *,
        errors: list[Exception] | None = None,
    ) -> bool:
        start = time.monotonic()
        self._observe_stage(name, "started", 0.0)
        logger.info(
            "shutdown stage=%s reason=started timeout_s=%.3f remaining_total_s=%.3f inflight=%d",
            name,
            seconds,
            self._remaining(),
            self.inflight,
        )
        task = asyncio.ensure_future(awaitable)
        # Unlike wait_for, wait does not extend the deadline when cancellation
        # is ignored. The OS-thread watchdog bounds blocked Python cleanup too.
        done, _ = await asyncio.wait({task}, timeout=seconds)
        if not done:
            task.cancel()
            task.add_done_callback(self._consume_result)
            self._observe_stage(name, "timed_out", time.monotonic() - start)
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
        except Exception as error:
            if errors is not None:
                errors.append(error)
            # Continue to later teardown stages; cleanup/runtime failures become
            # a terminal error in _shutdown rather than abandoning resources here.
            logger.exception("shutdown stage=%s failed", name)
            self._observe_stage(name, "failed", time.monotonic() - start)
            return False
        self._observe_stage(name, "completed", time.monotonic() - start)
        logger.info(
            "shutdown stage=%s reason=completed elapsed_s=%.3f remaining_total_s=%.3f",
            name,
            time.monotonic() - start,
            self._remaining(),
        )
        return True

    def _observe_stage(self, stage: str, reason: str, elapsed: float) -> None:
        if self._metrics is not None:
            self._metrics.record_stage(stage, reason, elapsed, self._remaining())
            self._metrics.record_state(self.inflight, None)

    @staticmethod
    def _consume_result(task: asyncio.Future) -> None:
        if not task.cancelled():
            task.exception()

    async def _cleanup_worker(self) -> None:
        assert self._worker is not None
        if self._worker.done():
            # run() propagates this task's result after runtime teardown.
            return
        self._worker.cancel()
        try:
            await asyncio.shield(self._worker)
        except asyncio.CancelledError:
            if not self._worker.cancelled():
                raise

    async def _shutdown(self) -> None:
        assert self._worker is not None

        controls_active = self.runtime.close_engine_routes()

        async def join_controls():
            if controls_active:
                await self.runtime.shutdown_engine_routes()
            if self.wait_for_control_tasks is not None:
                await self.wait_for_control_tasks()

        controls_done = asyncio.create_task(join_controls())
        if self.notify_children is not None:
            self.notify_children()

        # Native terminal unregister waits for earlier discovery writes.
        await self._stage(
            "unregister",
            _unregister_endpoints(list(self.endpoints), report_failure=True),
            self._drain_remaining(),
        )
        grace = min(self._caps["router_grace"], self._drain_remaining())
        if grace:
            await self._stage(
                "router_grace", asyncio.sleep(grace), self._drain_remaining()
            )
        self.accepting = False
        self._observe_stage("stop_admission", "completed", 0.0)
        logger.info("shutdown stage=stop_admission reason=completed")

        await self._stage(
            "inflight",
            self.idle.wait(),
            min(self._caps["inflight"], self._drain_remaining()),
        )
        if self.wait_for_children is not None:
            # Children and remote ranks can still own KV transfers after their
            # requests finish. The request-only cap must not free their engine.
            await self._stage(
                "children", self.wait_for_children(), self._drain_remaining()
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
            allowance = min(self._caps["kv_transfer"], self._drain_remaining())
            started = time.monotonic()
            self._observe_stage("kv_transfer", "started", 0.0)
            logger.info(
                "shutdown stage=kv_transfer reason=unsupported fallback=wait timeout_s=%.3f",
                allowance,
            )
            await asyncio.sleep(allowance)
            self._observe_stage(
                "kv_transfer", "unsupported", time.monotonic() - started
            )
            logger.info(
                "shutdown stage=kv_transfer reason=unsupported fallback=wait elapsed_s=%.3f remaining_total_s=%.3f",
                time.monotonic() - started,
                self._remaining(),
            )
        else:
            self._observe_stage("kv_transfer", "skipped", 0.0)
            logger.info("shutdown stage=kv_transfer reason=skipped")

        if not await self._stage(
            "engine_controls", controls_done, self._drain_remaining()
        ):
            # Unwinding asyncio.run would cancel the engine owner and run its
            # cleanup concurrently with the callback. Exit without unwinding.
            logger.error(
                "Engine controls still own the engine at the cleanup deadline; force-exiting"
            )
            os._exit(70)

        # Engine cleanup and transport teardown share one cleanup allowance.
        self._hard_deadline = min(
            self._deadline, time.monotonic() + self._caps["cleanup"]
        )
        # Synchronous engine cleanup can block the event loop and hold the GIL.
        # Bound this shared allowance natively too; leave the total watchdog
        # armed so entering cleanup can never extend the original deadline.
        cleanup_watchdog = ShutdownWatchdog(self.cleanup_remaining())
        if self.pre_shutdown is not None:
            await self._stage(
                "withdraw",
                self.pre_shutdown(),
                max(0.0, self._hard_deadline - time.monotonic()),
            )
        self.shutdown_event.set()
        cleanup_budget = max(0.0, self._hard_deadline - time.monotonic())
        errors: list[Exception] = []
        clean = await self._stage(
            "cleanup", self._cleanup_worker(), cleanup_budget, errors=errors
        )
        runtime_budget = max(0.0, self._hard_deadline - time.monotonic())
        runtime_clean = await self._stage(
            "runtime",
            self.runtime.shutdown_and_wait(timeout_secs=runtime_budget),
            runtime_budget,
            errors=errors,
        )
        if self._worker.done() and self.finalize_worker is not None:
            finalized = await self._stage(
                "worker_resources",
                self.finalize_worker(),
                self.cleanup_remaining(),
                errors=errors,
            )
            runtime_clean = runtime_clean and finalized
        if clean and runtime_clean and self.post_shutdown is not None:
            runtime_clean = (
                await self._stage(
                    "deferred_signals",
                    self.post_shutdown(),
                    max(0.0, self._hard_deadline - time.monotonic()),
                    errors=errors,
                )
                and runtime_clean
            )
        if self._worker.done() and runtime_clean:
            assert self._watchdog is not None
            cleanup_watchdog.finish()
            self._watchdog.finish()
            if not self._worker.cancelled():
                self._worker.result()
        if not (clean and runtime_clean):
            raise RuntimeError("worker shutdown did not complete") from (
                errors[0] if errors else None
            )

    async def run(self, worker: Coroutine, *, install_signals: bool = True) -> None:
        """Run the engine owner and join the complete sequence before returning."""
        with tempfile.TemporaryFile(buffering=0) as deadline_file:
            self._child_deadline_fd = deadline_file.fileno()
            os.pwrite(self._child_deadline_fd, struct.pack("d", math.inf), 0)
            # Child launch tasks inherit this context without mutating libc
            # environ while native runtime threads may be reading it.
            token = _child_deadline_fd.set(self._child_deadline_fd)
            try:
                with (
                    self.handle_signals(lambda sig, _: self.request_shutdown(sig))
                    if install_signals
                    else nullcontext()
                ):
                    await self._run(worker)
            finally:
                self._child_deadline_fd = None
                _child_deadline_fd.reset(token)

    @contextmanager
    def handle_signals(self, callback):
        """Run the OS watchdog from signal delivery, even if Python cannot dispatch."""
        previous = {}
        self._signals = _ShutdownSignals(_total_timeout(), _parent_deadline_fd())
        try:
            for sig in (signal.SIGTERM, signal.SIGINT):
                previous[sig] = signal.signal(sig, callback)
                self._signals.intercept(sig)
            yield
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            self._signals.close()
            self._signals = None

    async def _run(self, worker: Coroutine) -> None:
        self._worker = asyncio.create_task(worker)
        requested = asyncio.create_task(self.requested_event.wait())
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


async def serve_endpoint(
    endpoint,
    handler,
    *,
    shutdown: WorkerShutdown | None,
    bidirectional: bool = False,
    **kwargs,
):
    """Register a handler with the worker's admission gate, retaining push egress."""
    if shutdown is not None:
        if shutdown.started:
            await shutdown.shutdown_event.wait()
            return
        if endpoint not in shutdown.endpoints:
            shutdown.endpoints.append(endpoint)
        if shutdown._metrics is None:
            shutdown._metrics = endpoint.metrics.shutdown_metrics()
        handler = shutdown.wrap(handler)
    try:
        if bidirectional:
            await endpoint.serve_bidirectional_endpoint(handler, **kwargs)
        else:
            await endpoint.serve_endpoint(handler, **kwargs)
    except Exception:
        if shutdown is not None and shutdown.started:
            # Registration can lose the race with the runtime fence. Do not
            # unwind the engine owner while other admitted requests still drain.
            await shutdown.shutdown_event.wait()
        raise
