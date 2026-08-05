# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""EXPERIMENT 2 -- kill the per-request ingress GIL acquisition.

The problem, in the shipped code
--------------------------------
Every request crosses into Python through ``engine::invoke_generator``
(``lib/bindings/python/rust/engine.rs:85-115``)::

    let stream = tokio::task::spawn_blocking(move || {
        let _nvtx = dynamo_nvtx_range!("pybridge.invoke_generator");
        Python::with_gil(|py| { ... generator.call(py, (python_input,), Some(&kwarg)) ... })
    }).await

and the push engine calls exactly that helper too
(``lib/bindings/python/rust/push_egress.rs:475``). So *per request*, on both
egress paths, the worker pays:

1. a **cross-thread GIL acquisition** on a `spawn_blocking` pool thread, to
   build the request object, the ``Context`` and the ``ResponseSender`` and to
   call ``generate``;
2. a **ready-deque entry** on the one asyncio loop, because
   ``demand_driven_python_stream`` (``engine.rs:122-151``) ends in
   ``into_future_with_locals``, i.e. a ``call_soon_threadsafe``.

(1) is the expensive one, and it is expensive for a reason that has nothing to
do with how much work it does. The capture holds the GIL 98.7 % of a 7.357 s
span, so a pool thread asking for it waits behind an eval loop that only drops
it every ``sys.getswitchinterval()``; the diagram measures that box at p50
1.05 ms. Worse, the *loop* pays for it too: a forced GIL hand-off is the loop
losing the interpreter mid-stage.

What this file builds
---------------------
``batched-admission`` -- **the loop drains admission itself.**

The tokio side stops poking the loop per request. It appends a plain ticket to
an MPSC ring (``collections.deque``; ``append``/``popleft`` are atomic under the
GIL, and in Rust this is a ``tokio::sync::mpsc`` / ``crossbeam::ArrayQueue``
holding *Rust* values -- no Python object, no GIL). One resident coroutine on
the loop owns admission: it drains the ring, and for each ticket builds the
context and calls ``generate`` under the GIL **it already holds**. Zero
``spawn_blocking``, zero cross-thread GIL acquisitions, and the pump only has to
be woken when it is actually parked -- a coalesced doorbell, armed only after a
couple of idle loop turns, so a busy loop polls instead of being poked.

The *batching* in the name turned out to be the part that does not pay: see
:data:`DEFAULT_MAX_BURST`, where draining 32 tickets per loop turn is 5 % SLOWER
than draining 1. What pays is having no crossing to batch.

``batched-admission-offloop`` -- **the ablation.** Same ring, but the batch is
handed to ONE ``spawn_blocking`` that builds *N* request objects under a single
``Python::with_gil``, and reaches the loop as ONE ``call_soon_threadsafe``. This
keeps ``pythonize(request)`` off the loop -- which matters on the real worker,
where it is real work the simulator charges nothing for
(``Costs.invoke_generator_us`` is 0 by construction) -- while still amortising
the GIL acquisition and the deque entry by the burst size. It isolates how much
of the win is "no thread hop" versus "fewer thread hops".

Both keep the response path byte-for-byte identical to ``baseline-push``: the
same real ``push_egress.py`` decorator, the same ``ResponseSender.send``, the
same one ``loop_meter.item()`` per response on the loop thread.

Work conservation
-----------------
Nothing in ``Costs`` is deleted. The four pre-submit stages (58.46 us) and
``trtllm:engine_submit`` (154.64 us) still run on the loop, once per request,
exactly as before; the per-response stages are untouched. What is removed is
*scheduling*, which ``Costs`` does not model at all: a ThreadPoolExecutor round
trip, a ``concurrent.futures`` completion chained back across threads, and one
of the two ready-deque entries. ``all us/item`` therefore stays flat.

What to expect from the benchmark
---------------------------------
``bench`` is a closed loop of ``batch`` requests with ``max_tokens = 1e6``: every
request is admitted once, in the first few milliseconds, and then streams
forever. Ingress is ~0.00002 % of that run, so **this architecture should score
the same as ``baseline-push`` on the bench, and it does.** That is the expected
result, not a disappointing one -- the bench deliberately isolates egress. The
effect lives in the request-heavy regime, which is why this module ships its
own harness::

    python3 -m egress_experiments.architectures.batched_admission

Honest caveats
--------------
* **The doorbell's coalescing is not what wins, and is barely exercised.** In
  every geometry measured here the ring is non-empty essentially always -- the
  loop is the bottleneck, so admission is always behind -- and the pump parks
  only a handful of times in 15,000 requests (``doorbells_per_request`` 0.000).
  The doorbell is what keeps an *idle* worker from spinning; it is not the
  source of the throughput number.
* **The simulator's tokio side is Python and holds the GIL where real tokio does
  not** (``rust_bridge.py``, "Known deviations" 1). Part of what this removes --
  the ``run_in_executor`` round trip and the ``concurrent.futures`` completion
  chained back across threads -- is Python-only plumbing that real Rust does
  with tokio primitives and no GIL. The part that is genuinely there in Rust,
  and genuinely removed, is ``spawn_blocking`` + ``Python::with_gil`` +
  ``generator.call`` per request: a real forced GIL hand-off off the loop.
  Treat the measured ratio as an upper bound on the real win for that reason,
  and as a lower bound for the opposite reason the README gives (this process
  has 3 GIL-capable threads against the capture's 50).
* **``pythonize(request)`` moves onto the loop** in ``batched-admission``.
  ``Costs.invoke_generator_us`` is 0 by construction, so the simulator charges
  nothing for it and this architecture is not billed for the move.
  ``batched-admission-offloop`` exists to bound that: it keeps the crossing on
  the pool and still lands most of the win.

Real-code change this stands for
--------------------------------
* ``engine.rs:75-120`` -- ``invoke_generator`` grows a batched sibling that takes
  a ``Vec`` of pending requests and does one ``Python::with_gil`` for all of
  them (``batched-admission-offloop``), or is bypassed entirely in favour of a
  Python-side admission pump fed by a Rust channel (``batched-admission``).
* ``push_egress.rs:459-500`` -- ``PythonPushEngine::generate`` stops awaiting
  ``invoke_generator`` inline and instead pushes ``(python_input, ctx, sender)``
  onto the admission channel, returning the ``ResponseStream`` immediately. It
  already builds the sender before the crossing, so the ordering works.
* ``push_egress.py`` -- gains a module-level ``admission_pump()`` coroutine that
  the worker starts once and that calls ``handler.generate(...)`` per drained
  ticket. ``push_egress_capable`` is unchanged.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import statistics
import sys
import time
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

from egress_experiments import architectures
from egress_experiments.architectures import Architecture
from egress_experiments.costs import Costs
from egress_experiments.dynamo_sim.probes import RequestRecord
from egress_experiments.dynamo_sim.rust_bridge import (
    Driver,
    FakeContext,
    PushDriver,
    ResponseSender,
    push_pump,
)

_perf = time.perf_counter_ns

#: Tickets admitted per burst before the pump hands the loop back.
#:
#: **Measured, and the answer is 1.** Admission is 213.1 us of modelled loop
#: work per request, so a burst of N blocks the response drain for N x 213 us,
#: and that costs more than the per-burst scheduling it saves. Interleaved,
#: 3 runs each, closed loop / concurrency 1000 / max_tokens 4::
#:
#:     max_burst=1    5,339 items/s   loop 72 % busy
#:     max_burst=4    5,218 items/s   loop 70 % busy
#:     max_burst=32   5,073 items/s   loop 68 % busy
#:
#: So the win is NOT amortising the crossing over a batch -- it is having no
#: cross-thread crossing at all. The knob is kept because it is the "admit only
#: at controlled points" control and because ``batched-admission-offloop``,
#: which really does need a batch to amortise its one ``spawn_blocking``, uses
#: the same constant.
DEFAULT_MAX_BURST = 1

#: Burst for the off-loop ablation, which has an actual crossing to amortise.
DEFAULT_OFFLOOP_BURST = 32

#: Loop turns the pump spins on an empty ring before arming the doorbell. This
#: is the "the loop polls instead of being poked" knob: a turn costs one
#: loop-LOCAL ``call_soon`` (no lock, no ``_write_to_self`` syscall), whereas a
#: doorbell costs a cross-thread ``call_soon_threadsafe``. Two turns is enough
#: to swallow the arrivals of a busy loop and still park promptly when idle.
DEFAULT_IDLE_SPINS = 2


# ---------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------


async def _drive_one(driver: Driver, anext: Any, sender: ResponseSender) -> None:
    """Advance one push-mode generator to exhaustion, on the loop.

    Identical in shape to ``rust_bridge.push_pump`` -- one ``__anext__`` runs the
    whole request and raises ``StopAsyncIteration`` -- with the error handling
    that ``PushDriver.run`` normally does on the tokio side folded in, because
    here there is no tokio-side future to carry it.
    """
    counter = [0]
    try:
        await push_pump(anext, counter)
    except asyncio.CancelledError:
        _safe_close(sender, None)
        raise
    except Exception as exc:  # pragma: no cover - defensive
        driver.errors.append(f"{type(exc).__name__}: {exc}")
        _safe_close(sender, f"{type(exc).__name__}: {exc}")
    finally:
        # MUST stay 0: a yield here is push_egress.rs's fallback arm, which puts
        # the per-response GIL acquisition straight back.
        driver.fallback_yields += counter[0]


def _safe_close(sender: ResponseSender, error: Optional[str]) -> None:
    """``close``/``close_with_error`` that survives a torn-down tokio loop."""
    try:
        if error is None:
            sender.close()
        else:
            sender.close_with_error(error)
    except RuntimeError:  # pragma: no cover - teardown race
        pass


# ---------------------------------------------------------------------------
# A: the loop drains admission itself
# ---------------------------------------------------------------------------


class _AdmissionRing:
    """MPSC ring the loop drains, plus a coalesced doorbell.

    Producer (tokio thread): ``append`` -- atomic under the GIL, no lock, no
    ``call_soon_threadsafe`` unless the consumer is demonstrably parked.

    Consumer (the ONE asyncio loop): a resident coroutine that pops in bursts,
    calls ``generate`` under the GIL it already holds, and spawns the per-request
    pump with a loop-LOCAL ``create_task``.

    The arm/disarm protocol is the Python spelling of ``AtomicBool::swap``: the
    consumer arms only after re-checking the ring, and the producer disarms
    before ringing. Both interleavings are safe -- a lost arm costs one extra
    poll, a duplicated ring costs one extra no-op wake -- and in Rust the swap
    makes it exactly-once.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        handler: Any,
        driver: Driver,
        max_burst: int = DEFAULT_MAX_BURST,
        idle_spins: int = DEFAULT_IDLE_SPINS,
    ) -> None:
        self._loop = loop
        self._handler = handler
        self._driver = driver
        self._max_burst = max(1, max_burst)
        self._idle_spins = max(0, idle_spins)

        self._ring: Deque[
            Tuple[dict, RequestRecord, ResponseSender]
        ] = collections.deque()
        self._armed = False
        self._waiter: Optional[asyncio.Future] = None
        self._task: Optional[asyncio.Task] = None
        self._stopped = False
        self._inflight: Set[asyncio.Task] = set()

        # -- counters, all reported --------------------------------------
        #: Cross-thread wakes actually issued. This is the number the whole
        #: architecture is about: baseline pays 1 per request.
        self.doorbells = 0
        #: Loop-local poll turns taken instead of a doorbell.
        self.idle_turns = 0
        self.admitted = 0
        self.bursts = 0
        self.burst_max = 0

    # -- consumer side (the loop) -----------------------------------------

    def start(self) -> None:
        """Called from ``on_started``, which runs ON the loop."""
        self._task = self._loop.create_task(self._pump(), name="admission-pump")

    def stop(self) -> None:
        """Called from ``on_finished``, also on the loop."""
        self._stopped = True
        self._wake()
        if self._task is not None:
            self._task.cancel()
            self._task = None
        for task in list(self._inflight):
            task.cancel()

    def _wake(self) -> None:
        waiter, self._waiter = self._waiter, None
        if waiter is not None and not waiter.done():
            waiter.set_result(None)

    async def _pump(self) -> None:
        ring = self._ring
        spins = 0
        while not self._stopped:
            n = 0
            while ring and n < self._max_burst:
                try:
                    ticket = ring.popleft()
                except IndexError:  # pragma: no cover - producer raced us
                    break
                self._admit(ticket)
                n += 1

            if n:
                self.bursts += 1
                self.admitted += n
                if n > self.burst_max:
                    self.burst_max = n
                spins = 0
                # Controlled point: everything already ready -- notably the
                # response notification for the last IPC batch -- runs before
                # the next burst does.
                await asyncio.sleep(0)
                continue

            if spins < self._idle_spins:
                # Poll rather than be poked. One loop-LOCAL call_soon, no lock
                # and no self-pipe write, versus a cross-thread wake.
                spins += 1
                self.idle_turns += 1
                await asyncio.sleep(0)
                continue

            # Genuinely idle: arm the doorbell and park. Costs nothing until a
            # producer rings it.
            waiter = self._loop.create_future()
            self._waiter = waiter
            self._armed = True
            if ring:  # a producer raced the arm
                self._armed = False
                self._waiter = None
                spins = 0
                continue
            try:
                await waiter
            except asyncio.CancelledError:
                return
            spins = 0

    def _admit(self, ticket: Tuple[dict, RequestRecord, ResponseSender]) -> None:
        """``invoke_generator``'s GIL section -- under the GIL the loop holds.

        On the real worker this is where ``pythonize(py, &request)``,
        ``Py::new(py, Context::new(..))`` and ``Py::new(py, sender)`` happen. All
        three need the GIL; none of them needs a *new* one.
        """
        request, _record, sender = ticket
        context = FakeContext(request["id"])
        context.response_sender = sender
        try:
            stream = self._handler.generate(request, context, response_sender=sender)
            anext = stream.__anext__
        except BaseException as exc:  # pragma: no cover - defensive
            self._driver.errors.append(f"{type(exc).__name__}: {exc}")
            _safe_close(sender, f"{type(exc).__name__}: {exc}")
            return
        task = self._loop.create_task(_drive_one(self._driver, anext, sender))
        self._inflight.add(task)
        task.add_done_callback(self._inflight.discard)

    # -- producer side (tokio) --------------------------------------------

    def offer(self, ticket: Tuple[dict, RequestRecord, ResponseSender]) -> None:
        if self._stopped:  # teardown: never leave a consumer waiting forever
            _safe_close(ticket[2], "admission closed")
            return
        self._ring.append(ticket)
        if self._armed:
            self._armed = False
            self.doorbells += 1
            self._loop.call_soon_threadsafe(self._wake)


class BatchedAdmissionDriver(PushDriver):
    """Push egress, but ingress never acquires the GIL cross-thread."""

    mode = "push"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        max_burst = kwargs.pop("max_burst", DEFAULT_MAX_BURST)
        idle_spins = kwargs.pop("idle_spins", DEFAULT_IDLE_SPINS)
        super().__init__(*args, **kwargs)
        self.ring = _AdmissionRing(
            self.py_loop,
            self.handler,
            self,
            max_burst=max_burst,
            idle_spins=idle_spins,
        )

    async def run(self, request: dict, record: RequestRecord) -> None:
        record.accepted_ns = _perf()

        # Rust-side objects: the channel halves exist before the crossing on the
        # shipped path too (push_egress.rs builds `response_channel` before
        # calling invoke_generator), so building them here is not a shortcut.
        sink: asyncio.Queue = asyncio.Queue()
        sender = ResponseSender(self.tokio.loop, sink, self.costs)
        self.senders.append(sender)

        consumer = asyncio.ensure_future(self._consume(sink, record))
        try:
            # The whole ingress hand-off: one atomic deque append. No thread
            # hop, no GIL acquisition, and a ready-deque entry only when the
            # pump is demonstrably parked.
            self.ring.offer((request, record, sender))
            # The request is done when the sender closes -- which is exactly
            # what `PythonPushEngine` waits for, since responses never come back
            # through the driver task.
            await consumer
        except BaseException:
            consumer.cancel()
            raise


class BatchedAdmission(Architecture):
    name = "batched-admission"
    description = "ingress via an MPSC ring the loop drains itself (no spawn_blocking)"
    egress = "push"

    max_burst = DEFAULT_MAX_BURST
    idle_spins = DEFAULT_IDLE_SPINS

    def __init__(self) -> None:
        self._driver: Optional[BatchedAdmissionDriver] = None

    def build_driver(self, handler, py_loop, tokio, costs) -> Driver:
        self._driver = BatchedAdmissionDriver(
            handler,
            py_loop,
            tokio,
            costs,
            max_burst=self.max_burst,
            idle_spins=self.idle_spins,
        )
        return self._driver

    def on_started(self, llm, driver) -> None:
        driver.ring.start()

    def on_finished(self, llm, driver) -> None:
        driver.ring.stop()

    def extra_report(self) -> Dict[str, Any]:
        driver = self._driver
        if driver is None:
            return {}
        ring = driver.ring
        admitted = max(1, ring.admitted)
        return {
            "admitted": ring.admitted,
            "bursts": ring.bursts,
            "mean_burst": round(ring.admitted / max(1, ring.bursts), 2),
            "max_burst": ring.burst_max,
            "doorbells": ring.doorbells,
            "doorbells_per_request": round(ring.doorbells / admitted, 3),
            "idle_poll_turns": ring.idle_turns,
        }


# ---------------------------------------------------------------------------
# B: one spawn_blocking per BURST (ablation)
# ---------------------------------------------------------------------------


def _invoke_many(handler: Any, batch: List[Tuple[dict, Any, ResponseSender]]) -> List:
    """``invoke_generator`` for N requests under ONE GIL acquisition.

    Runs on a ``spawn_blocking`` pool thread. This is what the batched sibling of
    ``engine.rs:85`` would look like: one ``Python::with_gil``, N
    ``generator.call``s inside it.
    """
    out: List[Tuple[Any, ResponseSender, Optional[BaseException]]] = []
    for request, context, sender in batch:
        try:
            stream = handler.generate(request, context, response_sender=sender)
            out.append((stream.__anext__, sender, None))
        except BaseException as exc:  # pragma: no cover - defensive
            out.append((None, sender, exc))
    return out


async def _pump_many(streams: List, driver: "BatchedOffLoopDriver") -> None:
    """Spawn every request's pump. ONE ready-deque entry got us all N here."""
    loop = asyncio.get_running_loop()
    for anext, sender, exc in streams:
        if driver._stopped:
            # Teardown got here first: the engine may already be gone, and
            # driving the generator now would try to submit to it.
            _safe_close(sender, "shutting down")
            continue
        if exc is not None:  # pragma: no cover - defensive
            driver.errors.append(f"{type(exc).__name__}: {exc}")
            _safe_close(sender, f"{type(exc).__name__}: {exc}")
            continue
        task = loop.create_task(_drive_one(driver, anext, sender))
        # Tracked so teardown can cancel a task that has been created but not
        # yet stepped: `loop.stop()` still drains the ready deque it is in, and
        # by then `llm.shutdown()` has run.
        driver.inflight.add(task)
        task.add_done_callback(driver.inflight.discard)


class BatchedOffLoopDriver(PushDriver):
    """Ingress batched, but still crossing on a ``spawn_blocking`` thread."""

    mode = "push"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._max_burst = max(1, kwargs.pop("max_burst", DEFAULT_MAX_BURST))
        super().__init__(*args, **kwargs)
        self._pending: Deque[Tuple[dict, Any, ResponseSender]] = collections.deque()
        self._nonempty: Optional[asyncio.Event] = None
        self._batcher: Optional[asyncio.Future] = None
        self._stopped = False
        self.inflight: Set[asyncio.Task] = set()
        self.admitted = 0
        self.bursts = 0
        self.burst_max = 0

    async def run(self, request: dict, record: RequestRecord) -> None:
        if self._batcher is None:
            # Lazily, because this is the first code of ours to run ON the
            # tokio loop; `on_started` runs on the Python loop instead.
            self._nonempty = asyncio.Event()
            self._batcher = asyncio.ensure_future(self._batch_loop())

        record.accepted_ns = _perf()
        sink: asyncio.Queue = asyncio.Queue()
        sender = ResponseSender(self.tokio.loop, sink, self.costs)
        self.senders.append(sender)
        context = FakeContext(request["id"])
        context.response_sender = sender

        consumer = asyncio.ensure_future(self._consume(sink, record))
        try:
            self._pending.append((request, context, sender))
            assert self._nonempty is not None
            if not self._nonempty.is_set():
                self._nonempty.set()
            await consumer
        except BaseException:
            consumer.cancel()
            raise

    async def _batch_loop(self) -> None:
        assert self._nonempty is not None
        while not self._stopped:
            if not self._pending:
                self._nonempty.clear()
                if not self._pending:
                    await self._nonempty.wait()
                continue

            batch: List[Tuple[dict, Any, ResponseSender]] = []
            while self._pending and len(batch) < self._max_burst:
                batch.append(self._pending.popleft())

            self.bursts += 1
            self.admitted += len(batch)
            if len(batch) > self.burst_max:
                self.burst_max = len(batch)

            # ONE cross-thread GIL acquisition for the whole burst. Under load
            # the burst grows on its own: requests accumulate while this await
            # is outstanding, which is the batching signal -- no timer, so a
            # lightly loaded worker still admits with burst size 1 and pays no
            # added latency.
            streams = await self.spawn_blocking(_invoke_many, self.handler, batch)
            if self._stopped:  # teardown raced the crossing
                for _anext, sender, _exc in streams:
                    _safe_close(sender, "shutting down")
                return

            # ONE ready-deque entry for the whole burst.
            self.loop_handoffs += 1
            future = asyncio.run_coroutine_threadsafe(
                _pump_many(streams, self), self.py_loop
            )
            future.add_done_callback(self._note_failure)

    def _note_failure(self, future) -> None:
        try:
            exc = future.exception()
        except Exception:  # pragma: no cover - cancelled at teardown
            return
        if exc is not None:  # pragma: no cover - defensive
            self.errors.append(f"{type(exc).__name__}: {exc}")


class BatchedAdmissionOffLoop(Architecture):
    name = "batched-admission-offloop"
    description = "ingress batched: ONE spawn_blocking + ONE deque entry per burst"
    egress = "push"

    max_burst = DEFAULT_OFFLOOP_BURST

    def __init__(self) -> None:
        self._driver: Optional[BatchedOffLoopDriver] = None

    def build_driver(self, handler, py_loop, tokio, costs) -> Driver:
        self._driver = BatchedOffLoopDriver(
            handler, py_loop, tokio, costs, max_burst=self.max_burst
        )
        return self._driver

    def on_finished(self, llm, driver) -> None:
        driver._stopped = True
        for task in list(driver.inflight):
            task.cancel()

    def extra_report(self) -> Dict[str, Any]:
        driver = self._driver
        if driver is None:
            return {}
        return {
            "admitted": driver.admitted,
            "bursts": driver.bursts,
            "mean_burst": round(driver.admitted / max(1, driver.bursts), 2),
            "max_burst": driver.burst_max,
        }


# ``python3 -m`` re-imports this module as ``__main__`` *after* the package's
# ``_discover()`` has already imported and registered it, so guard the names.
if BatchedAdmission.name not in architectures.names():
    architectures.register(BatchedAdmission)
if BatchedAdmissionOffLoop.name not in architectures.names():
    architectures.register(BatchedAdmissionOffLoop)


# ---------------------------------------------------------------------------
# Request-heavy harness
# ---------------------------------------------------------------------------
#
# `bench` is egress-only by construction (`batch` requests, `max_tokens` 1e6, so
# every request is admitted once in the first millisecond and then streams
# forever). It cannot see an ingress change. This is the same measurement --
# items/s off `loop_meter` -- with the geometry turned round: short requests, so
# the loop spends a real fraction of itself on admission.
#
# Loop work per request is  213.10 us of ingress
#   (normalize 1.16 + setup_disagg 37.95 + prepare_input 1.93 + sampling 17.42
#    + engine_submit 154.64)
# against  85.34 us x max_tokens  of egress, so `--max-tokens 2` puts ingress at
# 55 % of the loop's modelled work, `--max-tokens 4` at 38 %, `--max-tokens 16`
# at 13 %.
#
# The arrival must be OPEN. A closed loop cannot over-offer once `max_tokens` is
# finite: a request only releases its slot when the loop has delivered its LAST
# response, so admissions are paced by deliveries and the engine's offered rate
# collapses to exactly what the loop already achieved -- flat backlog, an idle
# fraction, and a measurement that is latency-limited rather than
# throughput-limited. Offering `--qps` well above the loop's request capacity
# instead keeps a queue in front of admission at all times, which is what makes
# items/s a throughput number. `loop occupancy` below is the evidence: modelled
# loop work per second, which has to sit near 100 % for the figure to mean
# anything.


def _heavy_config(architecture: str, args: argparse.Namespace, costs: Costs):
    from egress_experiments.fake_trtllm.engine import BatchConfig, ConstantIteration
    from egress_experiments.harness import SimConfig

    if args.arrival == "closed":
        # `requests` is a ceiling; the closed-loop orchestrator `gather`s one
        # coroutine per request up front, so it must stay finite and modest.
        requests = args.requests or args.concurrency * 60
    else:
        requests = args.requests or int(args.qps * args.duration_s * 1.2) + 100
    return SimConfig(
        architecture=architecture,
        arrival=args.arrival,
        qps=args.qps if args.arrival != "closed" else None,
        concurrency=args.concurrency,
        requests=requests,
        max_tokens=args.max_tokens,
        isl=8,
        batch=BatchConfig(total=args.batch),
        iteration=ConstantIteration(args.iteration_ms),
        stream_interval=1,
        max_backlog=args.max_backlog,
        duration_s=args.duration_s,
        costs=costs,
        lag_ms=5.0,
    )


def _run_heavy(architecture: str, args: argparse.Namespace) -> Dict[str, Any]:
    from egress_experiments.bench import _measure
    from egress_experiments.harness import run_simulation

    result = run_simulation(_heavy_config(architecture, args, Costs()))
    items_per_s, window_s, items = _measure(result, args.warmup_s)
    by_thread = dict(result.spin_us_by_thread)
    delivered = max(1, len(result.loop_item_times))
    loop_us_per_item = by_thread.get("MainThread", 0.0) / delivered
    return {
        "architecture": architecture,
        "items_per_s": items_per_s,
        "requests_per_s": items_per_s / max(1, args.max_tokens),
        "window_s": window_s,
        "items": items,
        # Modelled loop work per second of wall clock. Near 1.0 means the loop
        # had no idle fraction, which is what makes items/s a throughput number.
        "loop_occupancy": items_per_s * loop_us_per_item / 1e6,
        "backlog_growth_per_s": result.backlog_growth_per_s,
        "loop_us_per_item": loop_us_per_item,
        "all_us_per_item": sum(by_thread.values()) / delivered,
        "deque_entries_per_item": result.probe["enqueues"] / delivered,
        "blocking_gil_per_item": result.blocking_gil_acquisitions / delivered,
        "requests_completed": result.requests_completed,
        "meter_threads": result.loop_meter_threads,
        "errors": result.errors[:3],
        "arch": result.arch_report,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="batched_admission",
        description="Request-heavy loop throughput: items/s with real ingress load",
    )
    parser.add_argument(
        "--architecture",
        action="append",
        default=None,
        help="repeatable; default: baseline-push, batched-admission, "
        "batched-admission-offloop",
    )
    parser.add_argument("--baseline", default="baseline-push")
    parser.add_argument("--batch", type=int, default=240)
    parser.add_argument(
        "--arrival",
        choices=("closed", "constant", "poisson"),
        default="closed",
        help="closed holds --concurrency in flight (both paths then settle at "
        "the SAME responses-per-request, so loop us/item matches and the "
        "items/s ratio is a clean efficiency comparison); constant/poisson "
        "offer --qps regardless of completion",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1000,
        help="1000 is where every architecture settles at the SAME loop "
        "us/item, i.e. the same responses-per-request work mix, so the "
        "items/s ratio is efficiency and not a different job. Much higher "
        "and the up-front `gather` of one task per request starts to "
        "dominate the tokio loop; check `loop us/it` still matches",
    )
    parser.add_argument(
        "--qps",
        type=float,
        default=3000.0,
        help="offered request rate. Must exceed what the loop can admit "
        "(~1/(213.1 + 85.34 x max_tokens) per us) or the measurement is "
        "latency-limited rather than throughput-limited",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=0,
        help="0 = qps x duration x 1.2; a ceiling, not a target",
    )
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--iteration-ms", type=float, default=10.0)
    parser.add_argument("--duration-s", type=float, default=12.0)
    parser.add_argument("--max-backlog", type=int, default=100_000)
    parser.add_argument("--warmup-s", type=float, default=1.0)
    parser.add_argument("--repeat", type=int, default=3, help="median of N runs")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    wanted = args.architecture or [
        "baseline-push",
        "batched-admission",
        "batched-admission-offloop",
    ]

    runs: Dict[str, List[Dict[str, Any]]] = {name: [] for name in wanted}
    # Interleaved, not grouped: other agents benchmark on the same machine, so a
    # drift over the session must hit every architecture equally.
    for _ in range(args.repeat):
        for name in wanted:
            runs[name].append(_run_heavy(name, args))

    summary = []
    for name in wanted:
        rows = runs[name]
        summary.append(
            {
                "architecture": name,
                "items_per_s_median": statistics.median(r["items_per_s"] for r in rows),
                "items_per_s_runs": [round(r["items_per_s"], 1) for r in rows],
                "requests_per_s_median": statistics.median(
                    r["requests_per_s"] for r in rows
                ),
                "loop_us_per_item": statistics.median(
                    r["loop_us_per_item"] for r in rows
                ),
                "all_us_per_item": statistics.median(
                    r["all_us_per_item"] for r in rows
                ),
                "deque_entries_per_item": statistics.median(
                    r["deque_entries_per_item"] for r in rows
                ),
                "blocking_gil_per_item": statistics.median(
                    r["blocking_gil_per_item"] for r in rows
                ),
                "loop_occupancy": statistics.median(r["loop_occupancy"] for r in rows),
                "window_s": statistics.median(r["window_s"] for r in rows),
                "meter_threads": rows[-1]["meter_threads"],
                "errors": [e for r in rows for e in r["errors"]][:3],
                "arch": rows[-1]["arch"],
            }
        )

    if args.json:
        import json

        print(json.dumps(summary, indent=2, default=str))
        return 0

    base = next((s for s in summary if s["architecture"] == args.baseline), summary[0])
    width = 96
    print()
    print("=" * width)
    print("  REQUEST-HEAVY LOOP THROUGHPUT -- items/second through the asyncio loop")
    print("=" * width)
    offered = (
        f"closed loop, {args.concurrency} in flight"
        if args.arrival == "closed"
        else f"{args.arrival} @ {args.qps:g} qps"
    )
    print(
        f"  {offered} · batch {args.batch}"
        f" · si=1 · iteration {args.iteration_ms:g} ms"
        f" · max_tokens {args.max_tokens}"
    )
    print(
        f"  ingress is {100 * 213.10 / (213.10 + 85.34 * args.max_tokens):.0f} %"
        f" of the loop's modelled work per request"
        f" · median of {args.repeat} runs"
    )
    print("-" * width)
    print(
        f"{'architecture':<28}{'items/s':>11}{'vs base':>9}{'req/s':>9}"
        f"{'loop us/it':>12}{'all us/it':>11}{'deque/it':>10}{'sblk/it':>9}"
        f"{'busy':>7}"
    )
    print("-" * width)
    for s in summary:
        rel = (
            f"{s['items_per_s_median'] / base['items_per_s_median']:.3f}x"
            if base["items_per_s_median"]
            else "-"
        )
        print(
            f"{s['architecture']:<28}{s['items_per_s_median']:>11,.0f}{rel:>9}"
            f"{s['requests_per_s_median']:>9,.0f}"
            f"{s['loop_us_per_item']:>12.2f}{s['all_us_per_item']:>11.2f}"
            f"{s['deque_entries_per_item']:>10.4f}"
            f"{s['blocking_gil_per_item']:>9.4f}"
            f"{100 * s['loop_occupancy']:>6.0f}%"
        )
    print("-" * width)
    print()
    for s in summary:
        print(f"{s['architecture']}  runs: {s['items_per_s_runs']}")
        if s["arch"]:
            print(f"  arch: {s['arch']}")
        off_loop = {k: v for k, v in s["meter_threads"].items() if k != "MainThread"}
        if off_loop:
            print(f"  !! meter ticked OFF the loop: {off_loop}")
        if s["errors"]:
            print(f"  errors: {s['errors']}")
    print()
    print("deque/it = ready-deque entries per delivered item (ingress + response")
    print("notification); sblk/it = spawn_blocking GIL acquisitions per item;")
    print("busy = modelled loop work per second of wall clock -- the evidence")
    print("that items/s is a throughput number and not a latency artefact.")
    print()
    return 0


if __name__ == "__main__":
    # `python3 -m` has already imported this file once, as
    # `egress_experiments.architectures.batched_admission`, via the package's
    # `_discover()`. Delegate to THAT module object so the registry, the class
    # attributes and the code being run are one and the same. (runpy still
    # warns about the double import; it is benign for exactly this reason.)
    from egress_experiments.architectures import batched_admission as _module

    raise SystemExit(_module.main())
