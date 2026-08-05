# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""EXPERIMENT 3 -- two queues: one for requests, one for responses.

The claim under test
--------------------
``ASYNCIO_GIL_PATH.md``'s central structural statement is that
``call_soon_threadsafe`` enqueues a new request onto *the ONE asyncio deque the
response stream is draining through*, so egress cost converts directly into
admission latency.

asyncio gives a loop exactly one ready-deque, so the two cannot literally be
split while both live on one loop. What CAN be split is what lands in that
deque. Today the response path puts one entry there **per response**::

    proxy.dispatch_result_task              tensorrt_llm/executor/proxy.py:532
      queue.put_nowait(response)      x N   -- per-request AsyncQueue, free
      _SyncQueue.notify_many(loop, qs) x 1  -- llmapi/utils.py:475

``notify_many`` is one ``call_soon_threadsafe``, which is the number the
simulator's probe counts -- but it sets N ``asyncio.Event``s, and every one of
those resolves a future and does ``loop.call_soon(Task.__step)``. So the loop's
ready-deque really does receive **N task steps per IPC batch**, one per parked
``_generate_locally_impl``. A request admitted at that moment queues behind all
of them.

What this architecture builds
-----------------------------
Two explicit MPSC queues and ONE scheduler coroutine that decides, each pass,
how much of each to service:

* ``_Mux.responses`` -- a single ``collections.deque`` the proxy dispatch thread
  appends every response to (still a plain deque append off the loop, exactly as
  ``AsyncQueue.put_nowait`` is today). The waker is touched only on the
  empty->non-empty transition, i.e. only when the scheduler is parked -- the
  same discipline ``tokio::sync::mpsc`` uses and that ``push_egress.rs:29`` is
  describing when it says the tokio side "only ever does ``rx.recv().await``".
* ``_Mux.requests`` -- admissions handed over by the Rust ingress driver.
* ``TwoQueueScheduler`` -- one long-lived task on the loop. Per pass it drains
  up to ``response_quantum`` responses **inline**, then admits up to
  ``request_quantum`` requests, then yields once.

Consequence: N responses cost ONE ready-deque entry *and one task step*, not N.
The per-response coroutine machinery -- ``Event.wait`` future, task wakeup,
``GenerationResult.__anext__`` -> ``_aresult_step``, and the four nested
async-generator frames (``_generate_locally_impl`` -> ``generate_locally`` ->
``Handler.generate`` -> ``drive_push_egress``) -- collapses into a straight-line
call. No modelled work is deleted: ``_handle_response`` (23.97 us),
``build_response`` (50.65 us) and ``ResponseSender.send`` (10.72 us) all still
run, on the loop, in that order.

And admission is now bounded by the *quantum*, not by the backlog: a request
waits at most ``response_quantum`` responses, whatever the engine is offering.

Three registered variants
-------------------------
``two-queue-priority``    responses-first, R=64 / K=8. Throughput-leaning.
``two-queue-fair``        R=1 / K=1: strict alternation, the tightest possible
                          admission bound on one loop. The throughput it costs
                          is the fairness price, measured rather than argued.
``two-queue-split-loop``  admission moved to a SECOND event loop on its own
                          thread, so requests and responses share no ready-deque
                          at all. This is the variant the assignment expects to
                          be a net loss, and it is -- see the module's report.

What would change in the real tree
----------------------------------
* ``tensorrt_llm/executor/proxy.py:532`` (``dispatch_result_task``): append to
  one worker-level MPSC instead of N per-request ``AsyncQueue``s, and replace
  ``_SyncQueue.notify_many`` (``tensorrt_llm/llmapi/utils.py:475``) with a
  single waker touched only when the consumer is parked.
* ``tensorrt_llm/executor/result.py:1035`` (``_aresult_step``) stops being the
  per-response entry point; ``GenerationResultBase._handle_response``
  (``result.py:454``) is called directly by the pump.
* ``components/src/dynamo/trtllm/request_handlers/handler_base.py``
  (``_generate_locally_impl``): the ``async for res in generation_result`` loop
  becomes a per-request state object (cursor + sender) plus a ``deliver()``
  step; the pre-submit stages and ``llm.generate_async`` become the admission
  half.
* ``components/src/dynamo/trtllm/request_handlers/push_egress.py``:
  ``drive_push_egress`` (line 143) moves from per-request to worker-level. The
  ``ResponseSender`` contract (``send`` xN -> ``close()``) is unchanged, which
  is why ``lib/bindings/python/rust/push_egress.rs`` needs no change at all --
  ``ResponseSender::send`` (push_egress.rs:223) does not care which coroutine
  calls it, only that the caller already holds the GIL.
* ``lib/bindings/python/rust/engine.rs:85`` (``invoke_generator``) is untouched
  and still paid once per request; ``engine.rs:122``
  (``demand_driven_python_stream``) is still what advances the returned object
  once per request, exactly as ``push_egress.rs:475`` arranges today.

Honest caveat: this bypasses the shipped ``drive_push_egress_stream``, because
replacing that per-request driver with a worker-level pump IS the proposal.
``extra_report`` says so. The stage NVTX names are still emitted, so a capture
of this architecture reads back through ``capture_params`` unchanged.

What was measured
-----------------
1. ``bench.py`` (closed loop, 240 never-finishing requests) shows **nothing**,
   median of 3: 9,645 / 9,574 (0.99x) / 9,169 (0.95x) / 9,653 (1.00x) items/s
   for baseline / priority / fair / split-loop; a second session gave 9,576 /
   8,940 (0.93x) / 8,786 (0.92x) / 9,466 (0.99x). That geometry has one burst
   of 240 admissions and then pure response traffic, so the REQUEST queue is
   empty for the entire measurement window and there is no fairness question to
   answer. The loop's non-modelled overhead there (~15 us/item on top of
   85.34 us of modelled work) turns out to be mostly GIL wait while the tokio
   consumer burns its own ``rust_egress`` 11.56 us -- not the per-response
   coroutine machinery this design removes. Sweeping R on the bench:
   R=1 0.82x, R=16 0.99x, R=64 0.99x, R=256 1.03x, R=2048 1.02x.

   That spread hides a confound worth knowing about. bench.py's run LENGTH is
   set by ``MAX_BACKLOG`` on ``responses_dispatched - driver.delivered`` --
   the *tokio-side* consumer -- while its SCORE is measured on the loop, and
   the loop's instantaneous rate rises ~15 % over a run (measured per second:
   9,287 9,147 9,467 9,575 9,674 8,390 10,907 10,238 10,751 9,543 10,936).
   These architectures make the proxy dispatch thread cheaper (no
   ``frozenset(queues)`` over the batch, no ``run_coroutine_threadsafe`` per
   IPC message), so the engine is back-pressured less, the backlog fills
   sooner, and the run is cut short: full-run spans of 3.0-12.1 s against
   baseline's 11.2-16.3 s. Scoring a window every run can supply --
   ``[t0+1s, t0+3s]``, median of 5, interleaved -- removes it::

       baseline-push          9,095/s   1.000x
       two-queue-priority     9,472/s   1.041x
       two-queue-split-loop   9,125/s   1.003x
       two-queue-fair         8,552/s   0.940x

   So the honest reading of the bench is parity for priority and split-loop
   (+4 % / +0 % matched, -1 % to -7 % unmatched) and a real 6 % loss for strict
   alternation.

2. A mixed open-arrival load is where it shows. Both queues busy, batch 150,
   iteration 30 ms, 3 tok/req, ~1,650 qps, median of 3::

       baseline-push        4,475 items/s   admit p50 155.8 ms  p99 600.8  TTFT 525.9
       two-queue-priority   4,845  (1.08x)  admit p50  47.1 ms  p99 180.0  TTFT 172.6
       two-queue-fair       4,823  (1.08x)  admit p50  34.9 ms  p99 159.7  TTFT 218.0
       two-queue-split-loop 4,836  (1.08x)  admit p50  42.0 ms  p99 226.9  TTFT 178.5

   1.08x throughput AND 3.3x lower admission wait. Work is conserved to 0.3 %:
   163.45 vs 163.83 / 163.60 / 163.02 us per item across all threads.

3. The quantum IS the bound. At batch 400 / 55 ms the baseline's admission wait
   is set by the engine's batch (p50 70.6 ms); the scheduler holds it at
   ``R x 85.34 us`` = 5.5 ms and measures 5.44 ms. 13x, at 1.01x throughput.

4. Under the 45-thread contention regime (``--gil-noise 42``) the split loop is
   the biggest win, not the loss the design brief expected: 4,178 items/s
   against baseline's 2,849 (**1.47x**), because 71.0 of the 163.3 us of
   modelled per-item work leaves the response loop entirely. ``all us/item``
   drops 259 -> 203 there, and that is the *noise* threads, not the
   architecture: they hold the GIL on a wall-clock duty cycle, so their charge
   per item falls as items/s rises. The request+response work itself stays at
   162.4 vs 163.3.

5. The one place explicit priority BACKFIRES. At ``--gil-noise 42``,
   ``two-queue-priority`` (R=64, K=8) has admission p99 **1.12x** -- worse than
   baseline -- while its throughput is 1.20x. A pass that services 64 responses
   and 8 requests is a 1:8 admission ratio; the workload's ratio is 1:2.9, so
   once contention stretches a pass the fixed budget K becomes the bottleneck
   and the request queue grows without bound. ``two-queue-fair`` (1:1) does not
   have this failure mode and lands at 0.48x. **K must be sized against the
   offered qps, or the explicit budget is an explicit bottleneck.**
   ``DYN_SIM_TQ_K=32`` (1:2) fixes it in the same regime -- 3,668 items/s
   (1.12x) with admission p99 675 ms against baseline's 1,229 (0.55x). K=8 is
   left as the registered default because every table above was measured at it;
   a real deployment should size K from the offered qps, or use the split loop,
   which has no budget to get wrong.

6. Do not make R unbounded. ``DYN_SIM_TQ_R=1000000`` drains 1 M responses
   before yielding: the harness's own completion event cannot run, the run
   overshoots ``duration_s`` by 17x, and the egress consumer starves.
   "Drain everything, then admit" needs a cap.

Run the mixed measurement (the ``-m`` RuntimeWarning about a double import is
benign -- registration is idempotent)::

    python3 -m egress_experiments.architectures.two_queue \\
        --batch 150 --iteration-ms 30 --max-tokens 3 --requests 12000 --repeat 3
"""

from __future__ import annotations

import asyncio
import collections
import functools
import os
import threading
import time
from typing import Any, Deque, Dict, List, Optional, Tuple

from egress_experiments.architectures import Architecture, names, register
from egress_experiments.costs import Costs, pad_to, spin
from egress_experiments.dynamo_sim.probes import RequestRecord
from egress_experiments.dynamo_sim.rust_bridge import (
    Driver,
    FakeContext,
    PushDriver,
    ResponseSender,
    TokioRuntime,
)
from egress_experiments.dynamo_sim.worker import SamplingParams, TrtllmWorkerHandler
from egress_experiments.fake_trtllm.engine import EngineConfig
from egress_experiments.fake_trtllm.llm import FakeLLM
from egress_experiments.fake_trtllm.result import GenerationResult, Response
from egress_experiments.nvtx_shim import range_

_perf = time.perf_counter_ns


def _env_int(name: str, default: int) -> int:
    """Sweep knob. Reported by :meth:`extra_report`, so nothing is hidden."""
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


class _Yield:
    """``await`` this to give the loop exactly one pass.

    Cheaper than ``asyncio.sleep(0)``, which allocates a coroutine per call;
    this is a shared singleton whose ``__await__`` yields ``None`` once, which
    is precisely what ``Task.__step`` turns into ``call_soon(self.__step)``.
    """

    __slots__ = ()

    def __await__(self):
        yield


_YIELD = _Yield()


# ---------------------------------------------------------------------------
# Per-request state, and the two queues
# ---------------------------------------------------------------------------


class _Stream:
    """What ``_generate_locally_impl``'s frame held, as plain state.

    The per-choice cursor is the load-bearing part: TRT-LLM streams CUMULATIVE
    ``token_ids`` per output and dynamo must emit only the new slice.
    """

    __slots__ = ("sender", "record", "cursor", "num_input_tokens", "closed")

    def __init__(self, sender: ResponseSender, record: RequestRecord, isl: int) -> None:
        self.sender = sender
        self.record = record
        self.cursor: Dict[int, int] = {}
        self.num_input_tokens = isl
        self.closed = False


class _Job:
    """One admission waiting in the REQUEST queue."""

    __slots__ = ("request", "context", "sender", "record")

    def __init__(self, request: dict, context: Any, sender: Any, record: Any) -> None:
        self.request = request
        self.context = context
        self.sender = sender
        self.record = record


def _make_job(request: dict, context: Any, sender: Any, record: Any) -> _Job:
    """Built on a ``spawn_blocking`` thread, under the GIL -- engine.rs:85.

    Stands in for the Rust->Python crossing that builds the request object and
    calls ``generate``. Push pays this once per REQUEST on both the shipped path
    and this one; what changes is only what the resulting object is driven by.
    """
    return _Job(request, context, sender, record)


class _Mux:
    """The two queues plus the park/wake protocol between them.

    ``parked`` is the whole trick. Producers append and only touch the waker
    when the consumer is asleep, so under saturation the response side costs
    ZERO ready-deque entries -- against ``notify_many``'s one per IPC batch and
    its N internal task steps. The handshake is Dekker's: a producer does
    append-then-read-``parked``, the scheduler does set-``parked``-then-read-
    queue, and under the GIL that cannot lose a wakeup.
    """

    __slots__ = (
        "responses",
        "requests",
        "loop",
        "wake",
        "parked",
        "wakes",
        "errors",
    )

    def __init__(self) -> None:
        self.responses: Deque[Tuple[GenerationResult, Response]] = collections.deque()
        self.requests: Deque[_Job] = collections.deque()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.wake: Optional[asyncio.Event] = None
        self.parked = False
        self.wakes = 0
        self.errors: List[str] = []

    def bind(self, loop: asyncio.AbstractEventLoop, wake: asyncio.Event) -> None:
        self.wake = wake
        self.loop = loop

    def _kick(self) -> None:
        if not self.parked:
            return
        loop = self.loop
        wake = self.wake
        if loop is None or wake is None:
            return
        self.wakes += 1
        try:
            loop.call_soon_threadsafe(wake.set)
        except RuntimeError:
            pass  # loop already closed: teardown beat us to it

    def offer_responses(self) -> None:
        """Called once per IPC batch, after the appends."""
        self._kick()

    def offer_request(self, job: _Job) -> None:
        self.requests.append(job)
        self._kick()


# ---------------------------------------------------------------------------
# Proxy side -- one queue instead of N AsyncQueues
# ---------------------------------------------------------------------------


class TwoQueueLLM(FakeLLM):
    """``dispatch_result_task`` feeding one worker-level MPSC.

    Line-for-line the shipped ``proxy.py:532`` except for the last two steps:
    ``queue.put_nowait(response)`` into a per-request ``AsyncQueue`` becomes an
    append to the shared deque (same operation, same thread, same cost), and
    ``_SyncQueue.notify_many`` becomes a conditional waker.
    """

    def __init__(
        self,
        engine_config: Optional[EngineConfig] = None,
        costs: Optional[Costs] = None,
        mux: Optional[_Mux] = None,
    ) -> None:
        super().__init__(engine_config, costs=costs)
        self._mux = mux or _Mux()

    def dispatch_result_task(self) -> bool:
        engine = self._engine
        if engine is None:
            return False  # shutdown raced us
        res = engine.result_link.parent.get(timeout=0.25)
        if res is None:
            return not self._stop.is_set()

        iteration = range_("_handle_responses", color="green")
        iteration.__enter__()
        pending = self._mux.responses
        touched = False

        batch = res if isinstance(res, list) else [res]
        for item in batch:
            if item is None:
                iteration.__exit__()
                return False  # shutdown
            self.responses_dispatched += 1

            with self._results_lock:
                result = self._results.get(item.client_id)
            if result is None:
                continue  # late response for an already-finalised request
            pending.append((result, item))
            touched = True

            if item.has_error() or (item.result is not None and item.result.is_final):
                with self._results_lock:
                    self._results.pop(item.client_id, None)

        self.ipc_messages += 1
        self.ipc_times.append(_perf())
        self.ipc_batch_sizes.append(len(batch))

        if touched:
            before = self._mux.wakes
            self._mux.offer_responses()
            # Count REAL ready-deque entries, so responses_per_deque_entry stays
            # an honest number rather than inheriting notify_many's cadence.
            self.notify_many_calls += self._mux.wakes - before

        iteration.__exit__()
        return True


# ---------------------------------------------------------------------------
# Worker side -- admission and delivery, as two callables instead of a coroutine
# ---------------------------------------------------------------------------


class TwoQueueHandler(TrtllmWorkerHandler):
    """``handler_base`` split at the point where it parks.

    ``admit`` is everything up to and including ``llm.generate_async``;
    ``deliver`` is one turn of the ``async for res in generation_result`` loop.
    Between them there is no coroutine and no await, which is the entire point:
    the response half no longer owns a task.
    """

    def __init__(self, llm: FakeLLM, costs: Costs, records: Dict[str, RequestRecord]):
        super().__init__(llm, costs=costs, records=records)
        self.admissions = 0
        self.orphan_retries = 0

    # -- ingress half ------------------------------------------------------

    def admit(self, job: _Job) -> None:
        """Runs on whichever loop services the REQUEST queue."""
        costs = self.costs
        record = job.record
        if record is not None and not record.admitted_ns:
            # The scheduler has drained to this request. Everything between
            # accepted_ns and here is queueing -- the quantity the diagram
            # leaves blank.
            record.admitted_ns = _perf()

        for stage_name, stage_us in (
            ("trtllm:normalize_request", costs.normalize_request_us),
            ("trtllm:setup_disagg_params", costs.setup_disagg_params_us),
            ("trtllm:prepare_input", costs.prepare_input_us),
            ("trtllm:sampling_params", costs.sampling_params_us),
        ):
            with range_(stage_name, color="cyan"):
                spin(costs.scaled(stage_us))

        request = job.request
        sampling_params = SamplingParams(
            max_tokens=int(request.get("max_tokens", 64)),
            n=int(request.get("n", 1)),
        )
        token_ids = request.get("token_ids")

        result = self.llm.generate_async(
            inputs=token_ids,
            sampling_params=sampling_params,
            disaggregated_params=None,
            streaming=True,
            trace_headers=None,
            scheduling_params=None,
            priority=0.5,
            cache_salt=None,
        )
        # Set with no await in between, so the scheduler can never observe a
        # result without its stream on this loop. The split-loop variant sets it
        # from another thread, where the engine's whole iteration stands between
        # submit and the first response -- the orphan retry below covers it.
        result.tq_stream = _Stream(job.sender, record, len(token_ids or ()))
        self.admissions += 1

    # -- egress half -------------------------------------------------------

    def deliver(self, result: GenerationResult, response: Response) -> bool:
        """One response, ON the loop. Returns False if the stream isn't up yet."""
        stream: Optional[_Stream] = getattr(result, "tq_stream", None)
        if stream is None:
            self.orphan_retries += 1
            return False

        # result.py:454 via result.py:1035 -- the SAME call _aresult_step makes,
        # minus the coroutine that used to carry it. 23.97 us, on the loop.
        result._handle_response(response)

        costs = self.costs
        build_us = costs.scaled(costs.build_response_us)
        cursor = stream.cursor
        sender = stream.sender
        finished = result.finished
        num_input_tokens = stream.num_input_tokens

        for output in result.outputs:
            # trtllm:build_response -- inline, because npw=0. Verbatim from
            # handler_base._generate_locally_impl, cursor and all.
            with range_("trtllm:build_response", color="yellow"):
                start = _perf()

                output_idx = getattr(output, "index", 0) or 0
                tokens_so_far = cursor.get(output_idx, 0)
                next_total_toks = len(output.token_ids)

                out: Dict[str, Any] = {
                    "token_ids": output.token_ids[tokens_so_far:],
                    "index": output_idx,
                }
                if output.finish_reason:
                    out["finish_reason"] = output.finish_reason
                if output.stop_reason:
                    out["stop_reason"] = output.stop_reason

                if out.get("finish_reason") or finished:
                    if not out.get("finish_reason"):
                        out["finish_reason"] = "unknown"
                    total_completion_tokens = sum(
                        len(o.token_ids) for o in result.outputs
                    )
                    out["completion_usage"] = {
                        "prompt_tokens": int(num_input_tokens),
                        "completion_tokens": int(total_completion_tokens),
                        "total_tokens": int(num_input_tokens + total_completion_tokens),
                        "prompt_tokens_details": None,
                    }

                pad_to(start, build_us)

            self.responses_yielded += 1
            # push_egress.py:192 -- the same range the shipped driver emits, so
            # a capture of this run still reads back through capture_params.
            with range_("trtllm:push_send", color="cyan"):
                sender.send(out)
            cursor[output_idx] = next_total_toks

        if finished and not stream.closed:
            # push_egress.py:179 -- close() replaces the StopAsyncIteration the
            # pull path relied on. Idempotent on the Rust side either way.
            stream.closed = True
            sender.close()
        return True


# ---------------------------------------------------------------------------
# The scheduler
# ---------------------------------------------------------------------------


class TwoQueueScheduler:
    """One coroutine, two queues, an explicit service discipline."""

    def __init__(
        self,
        handler: TwoQueueHandler,
        mux: _Mux,
        response_quantum: int,
        request_quantum: int,
        admit_here: bool = True,
    ) -> None:
        self.handler = handler
        self.mux = mux
        self.response_quantum = response_quantum
        self.request_quantum = request_quantum
        #: False for the split-loop variant, where admission lives elsewhere.
        self.admit_here = admit_here

        self.task: Optional[asyncio.Task] = None
        self.stopping = False
        self.passes = 0
        self.responses_drained = 0
        self.requests_admitted = 0
        self.parks = 0
        self.max_pass = 0

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        wake = asyncio.Event()
        self.mux.bind(loop, wake)
        self.task = loop.create_task(self._run(wake))

    def stop(self) -> None:
        self.stopping = True
        task = self.task
        if task is not None and not task.done():
            task.cancel()
        self.task = None

    async def _run(self, wake: asyncio.Event) -> None:
        mux = self.mux
        responses = mux.responses
        requests = mux.requests
        deliver = self.handler.deliver
        admit = self.handler.admit
        resp_quantum = self.response_quantum
        req_quantum = self.request_quantum
        admit_here = self.admit_here
        orphans: List[Tuple[GenerationResult, Response]] = []

        try:
            while not self.stopping:
                did = 0

                # ---- RESPONSE queue: bulk drain, one task step for all of it
                n = 0
                while responses and n < resp_quantum:
                    result, response = responses.popleft()
                    try:
                        if not deliver(result, response):
                            orphans.append((result, response))
                    except Exception as exc:  # pragma: no cover - defensive
                        mux.errors.append(f"deliver: {type(exc).__name__}: {exc}")
                    n += 1
                if orphans:
                    # Stream not registered yet (split-loop only). Put them back
                    # at the FRONT so per-request ordering is preserved.
                    responses.extendleft(reversed(orphans))
                    del orphans[:]
                self.responses_drained += n
                did += n
                if n > self.max_pass:
                    self.max_pass = n

                # ---- REQUEST queue: explicit admission budget
                k = 0
                if admit_here:
                    while requests and k < req_quantum:
                        job = requests.popleft()
                        try:
                            admit(job)
                        except Exception as exc:  # pragma: no cover - defensive
                            mux.errors.append(f"admit: {type(exc).__name__}: {exc}")
                        k += 1
                    self.requests_admitted += k
                    did += k

                self.passes += 1

                if did:
                    # One pass done. Yield so timers, the harness's `done` event
                    # and the tokio-side call_soon_threadsafe callbacks get a
                    # turn -- a scheduler that never yields deadlocks the run.
                    await _YIELD
                    continue

                # Nothing to do: park, and let the loop go idle.
                mux.parked = True
                try:
                    if not responses and not (admit_here and requests):
                        self.parks += 1
                        await wake.wait()
                finally:
                    wake.clear()
                    mux.parked = False
        except asyncio.CancelledError:
            return


# ---------------------------------------------------------------------------
# Rust ingress side
# ---------------------------------------------------------------------------


class TwoQueueDriver(PushDriver):
    """``PythonPushEngine``, but the request lands in a queue, not the deque.

    ``push_egress.rs:475`` still calls ``engine::invoke_generator``
    (``engine.rs:85``) once per request, so the ``spawn_blocking`` GIL crossing
    is paid exactly as it is today. What changes is the next line: instead of
    ``run_coroutine_threadsafe`` dropping a task onto the shared ready-deque
    behind every pending response, the request goes onto its own queue and the
    scheduler decides when to service it.
    """

    mode = "push"

    def __init__(
        self,
        handler: Any,
        py_loop: asyncio.AbstractEventLoop,
        tokio: TokioRuntime,
        costs: Costs,
        mux: Optional[_Mux] = None,
        admit_loop: Optional["_AdmitLoop"] = None,
    ) -> None:
        super().__init__(handler, py_loop, tokio, costs)
        self.mux = mux or _Mux()
        self.admit_loop = admit_loop

    async def run(self, request: dict, record: RequestRecord) -> None:
        context = FakeContext(request["id"])
        record.accepted_ns = _perf()

        sink: asyncio.Queue = asyncio.Queue()
        sender = ResponseSender(self.tokio.loop, sink, self.costs)
        self.senders.append(sender)
        context.response_sender = sender

        # engine.rs:85-114 -- spawn_blocking + with_gil, once per request.
        job = await self.spawn_blocking(
            functools.partial(_make_job, request, context, sender, record)
        )

        consumer = asyncio.ensure_future(self._consume(sink, record))
        try:
            self.loop_handoffs += 1
            if self.admit_loop is not None:
                # Second loop, second ready-deque, second thread.
                self.admit_loop.post(self.handler.admit, job)
            else:
                self.mux.offer_request(job)
            await consumer
        except BaseException:
            try:
                consumer.cancel()
            except RuntimeError:
                pass
            raise


class _AdmitLoop:
    """A SECOND asyncio loop, on its own thread, that owns admission only."""

    def __init__(self, name: str = "admission-loop") -> None:
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._name = name
        self.posts = 0

    def start(self) -> None:
        def run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.loop = loop
            self._ready.set()
            loop.run_forever()

        self._thread = threading.Thread(target=run, name=self._name, daemon=True)
        self._thread.start()
        self._ready.wait()

    def post(self, fn, *args) -> None:
        loop = self.loop
        if loop is None:
            return
        self.posts += 1
        try:
            loop.call_soon_threadsafe(fn, *args)
        except RuntimeError:
            pass

    def stop(self) -> None:
        loop = self.loop
        if loop is not None:
            try:
                loop.call_soon_threadsafe(loop.stop)
            except RuntimeError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self.loop = None


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------


class TwoQueuePriority(Architecture):
    name = "two-queue-priority"
    description = "two MPSC queues + scheduler coroutine (R=64 responses, K=8 requests)"
    egress = "push"

    #: Responses drained per scheduler pass before requests get a turn.
    response_quantum = 64
    #: Requests admitted per pass.
    request_quantum = 8
    #: Admission on a second event loop instead of this one.
    split_loop = False

    def __init__(self) -> None:
        self.mux = _Mux()
        self.response_quantum = _env_int("DYN_SIM_TQ_R", type(self).response_quantum)
        self.request_quantum = _env_int("DYN_SIM_TQ_K", type(self).request_quantum)
        self._llm: Optional[TwoQueueLLM] = None
        self._handler: Optional[TwoQueueHandler] = None
        self._scheduler: Optional[TwoQueueScheduler] = None
        self._admit_loop: Optional[_AdmitLoop] = None

    # -- hooks -------------------------------------------------------------

    def build_llm(self, engine_config: EngineConfig, costs: Costs) -> FakeLLM:
        self._llm = TwoQueueLLM(engine_config, costs=costs, mux=self.mux)
        return self._llm

    def build_handler(self, llm, costs, records) -> Any:
        self._handler = TwoQueueHandler(llm, costs=costs, records=records)
        return self._handler

    def build_driver(self, handler, py_loop, tokio, costs) -> Driver:
        if self.split_loop:
            self._admit_loop = _AdmitLoop()
            self._admit_loop.start()
        self._scheduler = TwoQueueScheduler(
            handler,
            self.mux,
            response_quantum=self.response_quantum,
            request_quantum=self.request_quantum,
            admit_here=not self.split_loop,
        )
        return TwoQueueDriver(
            handler,
            py_loop,
            tokio,
            costs,
            mux=self.mux,
            admit_loop=self._admit_loop,
        )

    def on_started(self, llm: FakeLLM, driver: Driver) -> None:
        assert self._scheduler is not None
        self._scheduler.start(driver.py_loop)

    def on_finished(self, llm: FakeLLM, driver: Driver) -> None:
        if self._scheduler is not None:
            self._scheduler.stop()
        if self._admit_loop is not None:
            self._admit_loop.stop()

    def extra_report(self) -> Dict[str, Any]:
        sched = self._scheduler
        handler = self._handler
        if sched is None or handler is None:
            return {}
        passes = max(1, sched.passes)
        report: Dict[str, Any] = {
            "R_response_quantum": self.response_quantum,
            "K_request_quantum": self.request_quantum,
            "scheduler_passes": sched.passes,
            "responses_per_pass": round(sched.responses_drained / passes, 2),
            "max_responses_in_one_pass": sched.max_pass,
            "requests_admitted": handler.admissions,
            "waker_kicks": self.mux.wakes,
            "scheduler_parks": sched.parks,
            "orphan_retries": handler.orphan_retries,
            # This architecture replaces drive_push_egress_stream with a
            # worker-level pump, so the shipped per-request driver is not in
            # play. Said out loud rather than left to be discovered.
            "shipped_drive_push_egress": False,
        }
        if self._admit_loop is not None:
            report["admission_loop_posts"] = self._admit_loop.posts
        if self.mux.errors:
            report["errors"] = self.mux.errors[:3]
        return report


class TwoQueueFair(TwoQueuePriority):
    name = "two-queue-fair"
    description = "same two queues, strict alternation (R=1, K=1) -- fairness price"
    response_quantum = 1
    request_quantum = 1


class TwoQueueSplitLoop(TwoQueuePriority):
    name = "two-queue-split-loop"
    description = (
        "requests admitted on a SECOND event loop/thread; responses on the first"
    )
    split_loop = True


def _register_all() -> None:
    """Idempotent: this module is also runnable as ``__main__``."""
    existing = set(names())
    for cls in (TwoQueuePriority, TwoQueueFair, TwoQueueSplitLoop):
        if cls.name not in existing:
            register(cls)


_register_all()


# ---------------------------------------------------------------------------
# Mixed request+response measurement
# ---------------------------------------------------------------------------
#
# The bench's geometry is closed-loop with a fixed set of never-finishing
# requests, so after the first iteration it exercises the RESPONSE queue only.
# This exercises both: an open Poisson arrival process at the steady-state qps
# that keeps the batch full, with short requests so admission churn is
# continuous.


def _mixed_config(architecture: str, args) -> Any:
    from egress_experiments.dynamo_sim.gil_noise import GilNoiseConfig
    from egress_experiments.fake_trtllm.engine import BatchConfig, ConstantIteration
    from egress_experiments.harness import SimConfig

    return SimConfig(
        architecture=architecture,
        arrival=args.arrival,
        qps=args.qps,
        requests=args.requests,
        max_tokens=args.max_tokens,
        isl=args.isl,
        batch=BatchConfig(total=args.batch),
        iteration=ConstantIteration(args.iteration_ms),
        stream_interval=1,
        max_backlog=args.max_backlog,
        duration_s=args.duration_s,
        costs=Costs(),
        lag_ms=5.0,
        # The split-loop variant's whole question is what a second GIL-capable
        # thread costs, and at three threads that is nearly free. This is the
        # regime knob the README documents for exactly that reason.
        gil_noise=GilNoiseConfig(threads=args.gil_noise),
    )


def _main(argv: Optional[List[str]] = None) -> int:
    import argparse
    import statistics
    import sys

    from egress_experiments import bench
    from egress_experiments.harness import run_simulation

    p = argparse.ArgumentParser(
        prog="two_queue",
        description="mixed request+response measurement: both queues busy",
    )
    p.add_argument(
        "--architecture",
        action="append",
        default=None,
        help="repeatable; default is baseline-push + the three two-queue variants",
    )
    p.add_argument("--batch", type=int, default=80)
    p.add_argument("--iteration-ms", type=float, default=12.0)
    p.add_argument("--max-tokens", type=int, default=12)
    p.add_argument("--isl", type=int, default=8)
    p.add_argument("--requests", type=int, default=6000)
    p.add_argument("--qps", type=float, default=None, help="default: steady state")
    p.add_argument("--arrival", default="poisson", choices=("constant", "poisson"))
    p.add_argument("--max-backlog", type=int, default=200_000)
    p.add_argument("--duration-s", type=float, default=20.0)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument(
        "--gil-noise",
        type=int,
        default=0,
        metavar="N",
        help="extra GIL-capable threads; the decode worker had ~45",
    )
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    wanted = args.architecture or [
        "baseline-push",
        "two-queue-priority",
        "two-queue-fair",
        "two-queue-split-loop",
    ]

    rows: List[Dict[str, Any]] = []
    for name in wanted:
        runs = []
        for _ in range(args.repeat):
            result = run_simulation(_mixed_config(name, args))
            items_per_s, window_s, items = bench._measure(result, 1.0)
            by_thread = dict(result.spin_us_by_thread)
            delivered = max(1, len(result.loop_item_times))
            runs.append(
                {
                    "items_per_s": items_per_s,
                    "window_s": window_s,
                    "items": items,
                    "loop_us": by_thread.get("MainThread", 0.0) / delivered,
                    "all_us": sum(by_thread.values()) / delivered,
                    "admit_p50": result.queue_wait["p50"],
                    "admit_p90": result.queue_wait["p90"],
                    "admit_p99": result.queue_wait["p99"],
                    "admit_max": result.queue_wait["max"],
                    "ttft_p50": result.ttft["p50"],
                    "qps": result.achieved_qps,
                    "completed": result.requests_completed,
                    "by_thread": {
                        k: round(v / delivered, 2) for k, v in by_thread.items()
                    },
                    "meter": result.loop_meter_threads,
                    "arch": result.arch_report,
                    "errors": result.errors[:2],
                }
            )
        med = {
            k: statistics.median([r[k] for r in runs])
            for k in (
                "items_per_s",
                "loop_us",
                "all_us",
                "admit_p50",
                "admit_p90",
                "admit_p99",
                "admit_max",
                "ttft_p50",
                "qps",
            )
        }
        med["architecture"] = name
        med["detail"] = runs[-1]
        rows.append(med)

    width = 100
    print()
    print("=" * width)
    print("  MIXED request+response -- both queues busy")
    print("=" * width)
    print(
        f"  open {args.arrival} arrivals · batch {args.batch} · si=1"
        f" · iteration {args.iteration_ms:g} ms · {args.max_tokens} tok/req"
        f" · {args.requests} requests · gil-noise {args.gil_noise}"
        f" · median of {args.repeat}"
    )
    print("-" * width)
    print(
        f"{'architecture':<24}{'items/s':>10}{'loop us':>9}{'all us':>9}"
        f"{'qps':>8}{'admit p50':>11}{'admit p90':>11}{'admit p99':>11}{'TTFT p50':>10}"
    )
    print("-" * width)
    base = next((r for r in rows if r["architecture"] == "baseline-push"), None)
    for r in rows:
        print(
            f"{r['architecture']:<24}{r['items_per_s']:>10,.0f}{r['loop_us']:>9.2f}"
            f"{r['all_us']:>9.2f}{r['qps']:>8,.0f}{r['admit_p50']:>11.2f}"
            f"{r['admit_p90']:>11.2f}{r['admit_p99']:>11.2f}{r['ttft_p50']:>10.1f}"
        )
    print("-" * width)
    if base:
        print()
        for r in rows:
            if r["architecture"] == "baseline-push":
                continue
            print(
                f"  {r['architecture']:<24} items/s {r['items_per_s'] / base['items_per_s']:.2f}x"
                f" · admission p99 {r['admit_p99'] / max(1e-9, base['admit_p99']):.2f}x"
            )
    print()
    for r in rows:
        d = r["detail"]
        print(f"{r['architecture']}")
        print(f"  work us/item by thread: {d['by_thread']}")
        off = {k: v for k, v in d["meter"].items() if k != "MainThread"}
        if off:
            print(f"  !! meter ticked OFF the loop: {off}")
        if d["arch"]:
            print(f"  arch: {d['arch']}")
        if d["errors"]:
            print(f"  errors: {d['errors']}")
    print()
    print("admission p50/p90/p99 are milliseconds a request spent between the Rust")
    print("ingress accepting it and the loop starting its handler -- ASYNCIO_GIL_PATH")
    print("'admission wait'. items/s is measured exactly as bench.py measures it.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
