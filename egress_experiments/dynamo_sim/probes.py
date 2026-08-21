# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loop instrumentation, reusing the methodology of ``queue_probe.py``
(``endpoints-launch/NVIDIA/src/sflow/tools/nvtx_patch/queue_probe.py``).

That probe exists because nsys *cannot* measure the one gap that matters: how
long work sits in the loop's ready deque before it runs. NVTX carries no
request id, and ingress spans 8 tokio threads while the handler is on 1 loop
thread, so a time-ordered join across range names is invalid. It takes two
independent measurements instead, and this module takes the same two so the
simulation's output is directly comparable to a real capture:

1. **LAG** -- a task sleeps ``lag_ms`` and reports how much *later* than
   requested it was resumed. Implementation-agnostic, no patching. If the loop
   is blocked for 11 ms, a 5 ms sleep returns ~11 ms late, and that overshoot
   is the delay any newly-enqueued callback would suffer.

2. **CB** -- per-callback wait, by wrapping ``call_soon_threadsafe`` so each
   enqueue is timestamped and the delta is taken when it finally runs.

The one addition: callbacks are **labelled by which coroutine was scheduled**,
not by the closure's name. Everything reaching the loop from another thread
goes through ``asyncio.run_coroutine_threadsafe``, whose internal closure is
always called ``callback``; that would collapse request admission and response
notification into one bucket, and telling them apart is the entire point.
"""

from __future__ import annotations

import asyncio
import collections
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

_perf = time.perf_counter_ns

#: Friendlier names for the coroutines that cross into the loop.
_LABELS = {
    "SyncQueue._notify_many": "response-notify (per IPC batch)",
    "anext_call": "pull anext (per response + admission)",
    "push_pump": "push pump (per request)",
}


def _percentile(sorted_ns: List[int], q: float) -> float:
    if not sorted_ns:
        return 0.0
    idx = min(len(sorted_ns) - 1, int(q * len(sorted_ns)))
    return sorted_ns[idx] / 1e6  # ms


@dataclass
class Bucket:
    samples: List[int] = field(default_factory=list)

    def add(self, ns: int, cap: int) -> None:
        if len(self.samples) < cap:
            self.samples.append(ns)

    def summary(self) -> Dict[str, float]:
        s = sorted(self.samples)
        return {
            "n": len(s),
            "p50_ms": _percentile(s, 0.50),
            "p90_ms": _percentile(s, 0.90),
            "p99_ms": _percentile(s, 0.99),
            "max_ms": (s[-1] / 1e6) if s else 0.0,
        }


class LoopProbe:
    """Install on a running loop; call :meth:`report` when the run is over."""

    def __init__(self, lag_ms: float = 5.0, cap: int = 200_000) -> None:
        self.lag_ms = lag_ms
        self.cap = cap
        self.lag = Bucket()
        self.callbacks: Dict[str, Bucket] = collections.defaultdict(Bucket)
        #: Total ready-deque entries created from other threads. This is the
        #: number the pull/push comparison lives or dies on.
        self.enqueues = 0
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._lag_task: Optional[asyncio.Task] = None
        self._orig_rcts = None
        self._orig_cst = None
        self._tag = threading.local()
        self.armed_callback_probe = False

    # -- install / uninstall ----------------------------------------------

    def install(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._install_lag(loop)
        self._install_callback_probe(loop)

    def _install_lag(self, loop: asyncio.AbstractEventLoop) -> None:
        step = self.lag_ms / 1000.0
        target_ns = int(step * 1e9)

        async def lagger() -> None:
            while True:
                t0 = _perf()
                await asyncio.sleep(step)
                over = (_perf() - t0) - target_ns
                if over > 0:
                    with self._lock:
                        self.lag.add(over, self.cap)

        self._lag_task = loop.create_task(lagger())

    def _install_callback_probe(self, loop: asyncio.AbstractEventLoop) -> None:
        probe = self
        tag = self._tag

        # run_coroutine_threadsafe -> call_soon_threadsafe happens synchronously
        # on the calling thread, so a thread-local carries the label across.
        orig_rcts = asyncio.run_coroutine_threadsafe

        def run_coroutine_threadsafe(coro, target_loop):
            tag.value = getattr(coro, "__qualname__", None) or type(coro).__name__
            try:
                return orig_rcts(coro, target_loop)
            finally:
                tag.value = None

        orig_cst = loop.call_soon_threadsafe

        def call_soon_threadsafe(callback, *args, **kwargs):
            label = getattr(tag, "value", None) or getattr(
                callback, "__name__", type(callback).__name__
            )
            label = _LABELS.get(label, label)
            t0 = _perf()

            def wrapped(*wa, **wk):
                dt = _perf() - t0
                with probe._lock:
                    probe.callbacks[label].add(dt, probe.cap)
                return callback(*wa, **wk)

            with probe._lock:
                probe.enqueues += 1
            return orig_cst(wrapped, *args, **kwargs)

        try:
            # Instance attribute first: it is scoped to this loop, so a second
            # loop in the same process (the tokio-side loop) is untouched.
            loop.call_soon_threadsafe = call_soon_threadsafe  # type: ignore[method-assign]
            asyncio.run_coroutine_threadsafe = run_coroutine_threadsafe  # type: ignore[assignment]
            self._orig_cst = orig_cst
            self._orig_rcts = orig_rcts
            self.armed_callback_probe = True
        except (AttributeError, TypeError):
            # Extension types (uvloop.Loop) refuse instance assignment; the LAG
            # measurement still carries the result. queue_probe.py hits the
            # same fork and reports it rather than failing.
            self.armed_callback_probe = False

    def uninstall(self) -> None:
        if self._lag_task is not None:
            self._lag_task.cancel()
            self._lag_task = None
        if self._orig_rcts is not None:
            asyncio.run_coroutine_threadsafe = self._orig_rcts  # type: ignore[assignment]
            self._orig_rcts = None
        if self._orig_cst is not None and self._loop is not None:
            try:
                del self._loop.call_soon_threadsafe  # type: ignore[attr-defined]
            except AttributeError:
                pass
            self._orig_cst = None

    # -- reporting ---------------------------------------------------------

    def reset(self) -> None:
        """Start a fresh measurement window without uninstalling the probe."""
        with self._lock:
            self.lag = Bucket()
            self.callbacks = collections.defaultdict(Bucket)
            self.enqueues = 0

    def report(self) -> Dict[str, object]:
        with self._lock:
            return {
                "lag": self.lag.summary(),
                "enqueues": self.enqueues,
                "callbacks": {k: v.summary() for k, v in self.callbacks.items()},
                "armed_callback_probe": self.armed_callback_probe,
            }


@dataclass
class RequestRecord:
    """Per-request timeline, in perf_counter_ns."""

    request_id: str
    accepted_ns: int = 0  # Rust ingress took it off the wire
    admitted_ns: int = 0  # the loop actually started the handler
    first_response_ns: int = 0
    last_response_ns: int = 0
    responses: int = 0

    @property
    def queue_wait_ms(self) -> float:
        """Time in the ONE asyncio deque before the handler could run."""
        return (self.admitted_ns - self.accepted_ns) / 1e6

    @property
    def ttft_ms(self) -> float:
        return (self.first_response_ns - self.accepted_ns) / 1e6

    @property
    def tpot_ms(self) -> float:
        if self.responses < 2:
            return 0.0
        return (
            (self.last_response_ns - self.first_response_ns)
            / 1e6
            / (self.responses - 1)
        )


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"n": 0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0, "mean": 0.0}
    s = sorted(values)

    def pct(q: float) -> float:
        return s[min(len(s) - 1, int(q * len(s)))]

    return {
        "n": len(s),
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p99": pct(0.99),
        "max": s[-1],
        "mean": sum(s) / len(s),
    }
