# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Do the per-response work on the thread that is already holding the response.

The question
------------
``proxy_dispatch_result_thread`` already touches every single response: it
reads the IPC message, unpickles it, looks the request up and ``put_nowait``s
into a per-request ``AsyncQueue`` (``tensorrt_llm/executor/proxy.py:532``,
ported at ``fake_trtllm/llm.py:169``). That is *all* it does. Everything
expensive then happens on the ONE event loop:

* ``handle_response``      23.97 us  -- ``result.py:454``, reached from
  ``_aresult_step`` (``result.py:1035``), i.e. inside the coroutine the
  dispatch thread woke, ON the loop,
* ``trtllm:build_response`` 50.65 us -- ``handler_base.py:1155-1241``, inline
  because ``num_postprocess_workers: 0``,
* ``trtllm:push_send``      10.72 us -- ``push_egress.rs``'s ``ResponseSender``.

85.34 us of loop time per response, against a dispatch thread that is
essentially idle. The obvious move is to push the first one -- or the first two
-- back onto the thread that already has the object in its hand.

The obvious counter-argument is the GIL. Both threads are in the same
interpreter, so moving pure-Python work does not create parallelism; it can only
add cross-thread hand-offs. This module exists to settle that empirically
rather than by assertion, so it builds every point on the curve:

``dispatch-handle``     handle_response on the dispatch thread; build+send stay
                        on the loop.
``dispatch-both``       handle_response AND build_response on the dispatch
                        thread; the loop only does push_send.
``dispatch-both-pN``    the same, but the reader fans responses out to a pool
                        of N worker threads, sharded by ``client_id``.
``dispatch-both-bpN``   the same N-thread pool, with a bounded credit window so
                        the producer can never be more than
                        :data:`CREDIT_WINDOW` responses ahead of the loop.
``loop-executor-N``     the same work, same GIL, different scheduling: the LOOP
                        offloads it with ``run_in_executor`` into an N-thread
                        pool and awaits the result.

Why the credit window is not optional
-------------------------------------
Moving the work to the producer also moves it from LAZY to EAGER. The baseline
does ``handle_response``/``build_response`` only when the loop pulls a response
out of the queue, so under overload the undelivered backlog costs nothing but
memory. A producer does the work the moment the response lands, for every
response the *engine* emitted -- and under this benchmark's deliberate 2.4x
oversubscription that is a lot of responses the loop will never reach. Measured
un-bounded at batch 240: 219,639 responses handled off-loop against 84,775 that
the loop got through, i.e. 2.6x the GIL time, all of it stolen from the loop.

That is a real property of eager production, not an artefact, so
``dispatch-both``/``-pN`` keep it and report it. But it is also fixable, and a
shippable version would have to fix it anyway (unbounded pre-built chunks is an
OOM). ``-bpN`` bounds it with a semaphore the loop releases as it consumes, so
the producer does work for at most :data:`CREDIT_WINDOW` responses the loop has
not reached. That restores the benchmark's conservation invariant and isolates
the question actually being asked: with the *same* total work, does splitting it
across threads in one interpreter help?

``loop-executor-N`` is the strict-lockstep end of the same axis -- the loop
awaits every offload, so the producer cannot run ahead at all.

**The eager variants are not benchmarkable and are kept only as the control.**
An unbounded producer keeps ``responses_dispatched`` climbing while the loop
falls behind, and the run stops taking ``bench.DURATION_S`` (12 s, documented
as "Hard bound on every run") seriously: measured 88 s for a single
``run_bench("dispatch-both", ladder=(240,))``, and one
``dispatch-both-p1`` rung was still running when a 1500 s external timeout
killed it. Their ``all us/item`` comes back at 210-2162 against the baseline's
89, i.e. they burn 2-24x the modelled work per delivered item. Read them as
"eager production is bad", not as throughput numbers.

The result
----------
Measured (5 sessions, medians, batch 240, baseline in the same session):

    baseline-push        9,330 items/s   loop 82.59 us/item   all 92.0
    dispatch-both-bp1    8,506  (0.91x)  loop 10.18 us/item   all 95.1
    dispatch-both-bp2    8,393  (0.90x)  loop  9.91 us/item   all 94.6
    dispatch-both-bp4    8,306  (0.89x)  loop  9.75 us/item   all 95.0
    dispatch-both-bp8    8,235  (0.88x)  loop  9.65 us/item   all 95.2
    loop-executor-1      7,230  (0.77x)  loop 10.61 us/item   all 95.8
    loop-executor-4      7,319  (0.78x)  loop 10.37 us/item   all 95.5

Taking 87 % of the loop's modelled work away -- 82.59 us/item down to 9.65 --
does not speed the loop up. It slows the system down by 9-12 %, and adding
threads makes it monotonically worse rather than better. There is no knee to
find. ``all us/item`` barely moves, which is the reason: the GIL serialises
the work wherever it runs, so total work per item is the invariant and the
only thing splitting it buys is hand-off overhead.

The arithmetic agrees. Total modelled work is 92 us/item and the baseline
achieves 9,330 items/s = 107 us/item of wall, so 86 % of the wall clock is
already modelled GIL-holding work. The ceiling is 1e6/92 = 10,870 items/s no
matter which thread runs it, and every architecture here is under it.

``sys.setswitchinterval`` does not change the answer, across a 100x range
(medians of 2, baseline re-measured under each setting):

    0.5 ms   baseline 10,382   bp1 0.84x   bp4 0.81x   executor-1 0.72x
    5 ms     baseline  9,615   bp1 0.86x   bp4 0.87x   executor-1 0.74x
    50 ms    baseline  9,671   bp1 0.85x   bp4 0.85x   executor-1 0.74x

So GIL hand-off granularity is not the mechanism; plain serialisation is.

Why ``run_in_executor`` is the worst of the three is visible in the benchmark's
own structural counter, and it is the diagram's point over again:

    deque entries/item   baseline 0.019   bp1 0.026   bp8 0.047   executor 1.043

Awaiting a pool future costs one ``call_soon_threadsafe`` per response to hand
the result back -- exactly the per-response ready-deque entry that
``push_egress.rs`` exists to remove (``SIMULATED_PATH.md``: "entries per
response: PULL 1.062, PUSH 0.062"). The dispatch-pool variants notify once per
shard per IPC message instead, which is why they lose 9 % rather than 23 %.

``DYN_SIM_SWITCH_INTERVAL=<seconds>`` calls :func:`sys.setswitchinterval` at
import, for the whole process, so the sensitivity of the answer to CPython's
GIL hand-off granularity can be measured against the *same* baseline in the
same session.

What would have to change in the real worker
--------------------------------------------
``dispatch-handle`` is a change to TRT-LLM only. ``proxy.py:539``'s
``process_res`` would call ``result._handle_response(res)`` before
``queue.put_nowait(res)``, and ``result.py:1035``'s ``_aresult_step`` would
drop its own call. That is a two-line move, but it needs the third piece
below.

``dispatch-both`` additionally needs dynamo's chunk construction
(``handler_base.py:1155-1241``) to be callable without the event loop. It
already is: everything it reads is on ``res.outputs`` / ``request``, and it
keeps its own per-choice cursor ``output_tokens_per_choice``
(``handler_base.py:1007``) rather than relying on loop state. Structurally this
is what TRT-LLM's own ``PostprocWorker`` path does -- ``result.py:_handle_response``
already has a branch where a *pre-built* ``PostprocWorker.Output`` replaces
``self._outputs[0]`` wholesale -- except that path uses 4 separate PROCESSES
(``trtllm-serve``'s column of the diagram) and this uses threads in the same
interpreter, which is exactly the difference under test.

The one genuine correctness issue, and how it is handled
--------------------------------------------------------
``CompletionOutput.token_ids`` is CUMULATIVE and dynamo emits only the new
slice using its own cursor. If ``handle_response`` runs ahead on another thread
while the loop is still building chunk *k*, the loop's
``output.token_ids[tokens_so_far:]`` picks up tokens from *k+1* as well. So the
producer records the cumulative length **as of that response** and the consumer
slices ``[tokens_so_far:total_len]``. Same work, no race. Responses for one
request are always handled by one thread (pool sharding is by ``client_id``),
so the per-request order TRT-LLM relies on is preserved.

Termination moves with the work. ``_handle_response`` sets ``self._done``,
and ``__anext__`` (``result.py:1104``) checks ``self._done`` *before* awaiting
-- so a producer that has already handled the final response would make the
consumer skip items still sitting in the queue. The producer therefore enqueues
an explicit end-of-stream marker after the last response's chunks and the
consumer drains until it sees it.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import queue as _queue
import sys
import threading
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from egress_experiments.architectures import Architecture, register
from egress_experiments.costs import Costs, pad_to, spin
from egress_experiments.dynamo_sim.probes import RequestRecord
from egress_experiments.dynamo_sim.rust_bridge import Driver, PushDriver, TokioRuntime
from egress_experiments.dynamo_sim.worker import SamplingParams, TrtllmWorkerHandler
from egress_experiments.fake_trtllm.aqueue import AsyncQueue, SyncQueue
from egress_experiments.fake_trtllm.engine import EngineConfig
from egress_experiments.fake_trtllm.llm import FakeLLM
from egress_experiments.fake_trtllm.result import GenerationResult, Response
from egress_experiments.nvtx_shim import range_

_perf = time.perf_counter_ns

# Process-wide, applied at import so the baseline measured in the same session
# sees it too. Without that the sensitivity sweep would compare a tuned
# experiment against an untuned baseline, which is not a comparison.
_SWITCH_INTERVAL = os.environ.get("DYN_SIM_SWITCH_INTERVAL", "").strip()
if _SWITCH_INTERVAL:
    sys.setswitchinterval(float(_SWITCH_INTERVAL))

#: How far ahead of the loop a backpressured producer may work, in responses.
#: ~0.05 s of loop time at the baseline's ~10,000 items/s: deep enough that
#: loop jitter never stalls the producer, shallow enough that the work done for
#: never-delivered responses is well under 1 % of a run.
CREDIT_WINDOW = int(os.environ.get("DYN_SIM_CREDIT_WINDOW", "512"))


class _Credit:
    """Bounds how far the producer may run ahead of the loop.

    Acquired once per queue entry by whoever does the work, released on the
    loop once the entry has been consumed. Blocking here drops the GIL, which
    is the point: a stalled producer must get out of the loop's way rather
    than spin.
    """

    __slots__ = ("_sem", "_stop")

    def __init__(self, limit: int, stop: threading.Event) -> None:
        self._sem = threading.Semaphore(limit)
        self._stop = stop

    def acquire(self) -> bool:
        """False means the run is shutting down; abandon the response."""
        while not self._stop.is_set():
            if self._sem.acquire(timeout=0.05):
                return True
        return False

    def release(self) -> None:
        self._sem.release()


class _EndOfStream:
    """Producer-side terminator; see the module docstring on ``self._done``."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<EOS>"


_EOS = _EndOfStream()


class _Handled:
    """One response after ``handle_response``, before ``build_response``.

    ``views`` is ``(index, total_len, finish_reason, stop_reason)`` per choice.
    ``total_len`` is the cumulative token count **as of this response**, which
    is what keeps the consumer's slice correct when the producer has already
    run ahead.
    """

    __slots__ = ("views", "finished")

    def __init__(
        self,
        views: List[Tuple[int, int, Optional[str], Optional[str]]],
        finished: bool,
    ) -> None:
        self.views = views
        self.finished = finished


class DispatchResult(GenerationResult):
    """A ``GenerationResult`` whose per-response work can run off the loop.

    Everything here is the same work as the baseline, in the same order, with
    the same NVTX names and the same padded costs -- only the calling thread
    changes.
    """

    def __init__(
        self,
        client_id: int,
        *,
        n: int = 1,
        streaming: bool = True,
        costs: Optional[Costs] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        num_input_tokens: int = 0,
        build_offloaded: bool = False,
    ) -> None:
        super().__init__(client_id, n=n, streaming=streaming, costs=costs, loop=loop)
        self.num_input_tokens = num_input_tokens
        self._build_offloaded = build_offloaded
        #: ``output_tokens_per_choice`` (handler_base.py:1007), moved onto the
        #: result because whoever builds the chunk needs it.
        self._cursor: Dict[int, int] = {}

    # -- producer side: whichever thread was handed the response ------------

    def offload_step(self, response: Response) -> List[Any]:
        """``handle_response`` (+ optionally ``build_response``) for ONE response.

        Returns what should be enqueued for the loop.
        """
        # result.py:454 -- unchanged, just not on the loop.
        self._handle_response(response)
        finished = self._done
        views = [
            (
                getattr(o, "index", 0) or 0,
                len(o.token_ids),
                o.finish_reason,
                o.stop_reason,
            )
            for o in self._outputs
        ]
        if not self._build_offloaded:
            return [_Handled(views, finished)]
        return [self.build_chunk(view, finished) for view in views]

    # -- trtllm:build_response, verbatim from handler_base.py:1155-1241 -----

    def build_chunk(
        self,
        view: Tuple[int, int, Optional[str], Optional[str]],
        finished: bool,
    ) -> Dict[str, Any]:
        output_idx, next_total_toks, finish_reason, stop_reason = view
        with range_("trtllm:build_response", color="yellow"):
            start = _perf()

            tokens_so_far = self._cursor.get(output_idx, 0)
            out: Dict[str, Any] = {
                # [tokens_so_far:next_total_toks], not [tokens_so_far:], so a
                # producer that has already accumulated the NEXT response into
                # the shared CompletionOutput cannot leak its tokens in here.
                "token_ids": self._outputs[output_idx].token_ids[
                    tokens_so_far:next_total_toks
                ],
                "index": output_idx,
            }
            if finish_reason:
                out["finish_reason"] = finish_reason
            if stop_reason:
                out["stop_reason"] = stop_reason

            if out.get("finish_reason") or finished:
                if not out.get("finish_reason"):
                    out["finish_reason"] = "unknown"
                total_completion_tokens = sum(len(o.token_ids) for o in self._outputs)
                out["completion_usage"] = {
                    "prompt_tokens": int(self.num_input_tokens),
                    "completion_tokens": int(total_completion_tokens),
                    "total_tokens": int(
                        self.num_input_tokens + total_completion_tokens
                    ),
                    "prompt_tokens_details": None,
                }

            pad_to(start, self._costs.scaled(self._costs.build_response_us))
        self._cursor[output_idx] = next_total_toks
        return out


class DispatchWorkLLM(FakeLLM):
    """``proxy.py``'s dispatch thread, given the per-response work to do.

    ``dispatch_mode``:

    ``inline``  the reader thread that owns the IPC socket does the work
                itself. Exactly one extra thread is involved, and there is no
                hand-off at all -- the truest minimal change to ``proxy.py``.
    ``pool``    the reader stays a reader and fans each IPC message out to
                ``workers`` threads, sharded by ``client_id`` so per-request
                ordering survives. Needed because the result socket is a single
                zmq PAIR and zmq sockets are not thread safe -- ``proxy.py:594``
                starts exactly ONE ``ManagedThread`` on it, and that is not
                negotiable.
    ``raw``     the dispatch thread behaves exactly like the baseline
                (``put_nowait`` only); the loop offloads instead. Used by the
                ``run_in_executor`` comparison.
    """

    def __init__(
        self,
        engine_config: EngineConfig,
        costs: Costs,
        *,
        dispatch_mode: str = "inline",
        build_offloaded: bool = True,
        workers: int = 0,
        credit_window: int = 0,
    ) -> None:
        super().__init__(engine_config, costs=costs)
        self.dispatch_mode = dispatch_mode
        self.build_offloaded = build_offloaded
        self.workers = workers if dispatch_mode == "pool" else 0
        #: Only meaningful in pool mode. Gating the INLINE reader would stop it
        #: draining the IPC lane, which freezes ``responses_dispatched`` and so
        #: hides the very backlog the benchmark uses to prove saturation.
        self.credit: Optional[_Credit] = (
            _Credit(credit_window, self._stop)
            if credit_window and dispatch_mode == "pool"
            else None
        )
        #: Responses abandoned because the run was shutting down while the
        #: producer was waiting for credit. Reported, so a stall cannot be
        #: mistaken for throughput.
        self._abandoned_cells: List[List[int]] = [[0] for _ in range(max(1, workers))]
        # notify_many is issued by several threads in pool mode, and `+= 1` on
        # a shared int is not atomic. One cell per producer, summed at
        # shutdown, which is before harness.py reads notify_many_calls.
        self._notify_cells: List[List[int]] = [[0] for _ in range(max(1, self.workers))]
        self._wqueues: List["_queue.SimpleQueue"] = []
        self._wthreads: List[threading.Thread] = []
        self.worker_errors: List[str] = []
        #: Responses whose work was done off the loop. Reported by the arch.
        self._offloaded_cells: List[List[int]] = [
            [0] for _ in range(max(1, self.workers))
        ]

    # -- lifecycle ---------------------------------------------------------

    def start(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        # Workers first: super().start() starts the reader, which can hand out
        # a shard on its very first message.
        for slot in range(self.workers):
            q: "_queue.SimpleQueue" = _queue.SimpleQueue()
            self._wqueues.append(q)
            thread = threading.Thread(
                target=self._worker_loop,
                args=(slot,),
                name=f"dispatch-worker-{slot}",
                daemon=True,
            )
            self._wthreads.append(thread)
            thread.start()
        super().start(loop)

    def shutdown(self) -> None:
        super().shutdown()
        for q in self._wqueues:
            q.put(None)
        for thread in self._wthreads:
            thread.join(timeout=5.0)
        self._wqueues = []
        self._wthreads = []
        self.notify_many_calls = sum(cell[0] for cell in self._notify_cells)

    @property
    def offloaded_responses(self) -> int:
        return sum(cell[0] for cell in self._offloaded_cells)

    # -- the boundary ------------------------------------------------------

    def generate_async(
        self,
        inputs: Any = None,
        sampling_params: Any = None,
        *,
        streaming: bool = True,
        **kwargs: Any,
    ) -> DispatchResult:
        """``FakeLLM.generate_async`` with a :class:`DispatchResult`.

        Identical otherwise, including the ``trtllm:engine_submit`` spin and
        the register-before-submit ordering ``proxy.py:550`` requires.

        ``num_input_tokens`` is captured here rather than assigned by the
        handler afterwards: the producer needs it for ``completion_usage`` and
        the first response can land before the handler's next bytecode.
        """
        if self._engine is None:
            raise RuntimeError("FakeLLM.start() must be called before generate_async()")

        client_id = next(self._client_ids)
        max_tokens = getattr(sampling_params, "max_tokens", None) or (
            self.engine_config.max_tokens
        )
        n = getattr(sampling_params, "n", None) or 1

        result = DispatchResult(
            client_id,
            n=n,
            streaming=streaming,
            costs=self.costs,
            loop=self._loop,
            num_input_tokens=len(inputs or []),
            build_offloaded=self.build_offloaded,
        )
        with self._results_lock:
            self._results[client_id] = result

        with range_("trtllm:engine_submit", color="red"):
            spin(self.costs.scaled(self.costs.engine_submit_us))
            self._engine.request_link.parent.put(
                {
                    "client_id": client_id,
                    "max_tokens": int(max_tokens),
                    "submitted_ns": _perf(),
                }
            )
        self.submitted += 1
        return result

    # -- proxy_dispatch_result_thread --------------------------------------

    def dispatch_result_task(self) -> bool:
        """``proxy.py:532``, with the per-response work put back on this thread."""
        engine = self._engine
        if engine is None:
            return False
        res = engine.result_link.parent.get(timeout=0.25)
        if res is None:
            return not self._stop.is_set()

        iteration = range_("_handle_responses", color="green")
        iteration.__enter__()

        batch = res if isinstance(res, list) else [res]
        n = self.workers
        shards: List[List[Tuple[DispatchResult, Response, bool]]] = [
            [] for _ in range(n)
        ]
        inline: List[Tuple[DispatchResult, Response, bool]] = []

        for item in batch:
            if item is None:
                iteration.__exit__()
                return False  # shutdown
            self.responses_dispatched += 1
            with self._results_lock:
                result = self._results.get(item.client_id)
            if result is None:
                continue  # late response, proxy.py:546
            final = item.has_error() or (
                item.result is not None and item.result.is_final
            )
            if final:
                with self._results_lock:
                    self._results.pop(item.client_id, None)
            job = (result, item, final)
            if n:
                # Shard by client_id: one request is always handled by one
                # thread, so the cumulative token accumulation TRT-LLM relies
                # on stays ordered.
                shards[item.client_id % n].append(job)
            else:
                inline.append(job)

        self.ipc_messages += 1
        self.ipc_times.append(_perf())
        self.ipc_batch_sizes.append(len(batch))

        alive = True
        if n:
            for slot, shard in enumerate(shards):
                if shard:
                    self._wqueues[slot].put(shard)
        elif inline:
            alive = self._run_jobs(inline, 0)
        iteration.__exit__()
        return alive

    # -- the work itself ---------------------------------------------------

    def _run_jobs(
        self, jobs: List[Tuple[DispatchResult, Response, bool]], slot: int
    ) -> bool:
        """Do the moved stages for a shard, then ONE notify for the shard.

        The notify comes after the whole shard, exactly as ``proxy.py``'s comes
        after the whole message: N responses still share one ready-deque entry.
        In pool mode there are up to ``workers`` of them per IPC message
        instead of one, which the benchmark's ``deque entries/item`` reports.
        """
        async_queues: List[SyncQueue] = []
        event_loop: Optional[asyncio.AbstractEventLoop] = None
        raw = self.dispatch_mode == "raw"
        credit = self.credit

        for result, response, final in jobs:
            queue = result.queue
            if raw:
                queue.put_nowait(response)
            else:
                # One credit per queue entry, taken BEFORE the work: the point
                # is to not do the work at all, not to do it and then wait.
                if credit is not None and not credit.acquire():
                    self._abandoned_cells[slot][0] += 1
                    break
                chunks = result.offload_step(response)
                if credit is not None:
                    for _ in range(len(chunks) - 1):
                        if not credit.acquire():
                            break
                for chunk in chunks:
                    queue.put_nowait(chunk)
                self._offloaded_cells[slot][0] += 1
            if final:
                # See the module docstring: _done has already been set by
                # _handle_response, so __anext__'s guard cannot be used.
                queue.put_nowait(_EOS)
            async_queues.append(queue)
            event_loop = event_loop or queue.loop

        if async_queues:
            try:
                SyncQueue.notify_many(event_loop, async_queues)
            except AsyncQueue.EventLoopShutdownError:
                return False
            self._notify_cells[slot][0] += 1
        return True

    def _worker_loop(self, slot: int) -> None:
        q = self._wqueues[slot]
        while True:
            jobs = q.get()
            if jobs is None:
                return
            try:
                if not self._run_jobs(jobs, slot):
                    return
            except Exception as exc:  # pragma: no cover - defensive
                self.worker_errors.append(f"{type(exc).__name__}: {exc}")
                return


class DispatchWorkHandler(TrtllmWorkerHandler):
    """The dynamo handler, consuming whatever the producer left it.

    ``consume``:

    ``prebuilt``  the chunk dicts are already built; the loop yields them
                  straight into ``ResponseSender.send``.
    ``build``     ``handle_response`` ran off-loop, ``build_response`` has not.
    ``executor``  nothing ran off-loop yet; the LOOP offloads both stages with
                  ``run_in_executor`` and awaits them.
    """

    def __init__(
        self,
        llm: FakeLLM,
        costs: Optional[Costs] = None,
        records: Optional[Dict[str, RequestRecord]] = None,
        *,
        consume: str = "prebuilt",
        executor: Optional[concurrent.futures.ThreadPoolExecutor] = None,
    ) -> None:
        super().__init__(llm, costs=costs, records=records)
        self.consume = consume
        self.executor = executor

    async def _generate_locally_impl(
        self, request: dict, context: Any
    ) -> AsyncGenerator[dict, None]:
        record = self.records.get(request["id"])
        if record is not None and not record.admitted_ns:
            record.admitted_ns = _perf()

        for stage_name, stage_us in (
            ("trtllm:normalize_request", self.costs.normalize_request_us),
            ("trtllm:setup_disagg_params", self.costs.setup_disagg_params_us),
            ("trtllm:prepare_input", self.costs.prepare_input_us),
            ("trtllm:sampling_params", self.costs.sampling_params_us),
        ):
            with range_(stage_name, color="cyan"):
                spin(self.costs.scaled(stage_us))

        sampling_params = SamplingParams(
            max_tokens=int(request.get("max_tokens", 64)),
            n=int(request.get("n", 1)),
        )
        generation_result = self.llm.generate_async(
            inputs=request.get("token_ids"),
            sampling_params=sampling_params,
            disaggregated_params=None,
            streaming=True,
            trace_headers=None,
            scheduling_params=None,
            priority=0.5,
            cache_salt=None,
        )

        # Drains the AsyncQueue directly instead of `async for res in
        # generation_result`. Same await, same single ready-deque entry per
        # notify_many -- but _handle_response is not called here, because the
        # producer already called it (or the executor is about to).
        aqueue = generation_result.aqueue
        mode = self.consume
        loop = asyncio.get_running_loop() if mode == "executor" else None
        credit = getattr(self.llm, "credit", None)

        while True:
            item = await aqueue.get()
            if item is _EOS:
                return
            if credit is not None:
                # Hand the producer back its room to work, on the loop, as soon
                # as the entry is off the queue.
                credit.release()
            if mode == "prebuilt":
                self.responses_yielded += 1
                yield item
            elif mode == "build":
                for view in item.views:
                    out = generation_result.build_chunk(view, item.finished)
                    self.responses_yielded += 1
                    yield out
            else:  # executor
                # Same GIL, different scheduling: the loop hands the work to a
                # pool thread and parks on the future instead of doing it.
                try:
                    chunks = await loop.run_in_executor(
                        self.executor, generation_result.offload_step, item
                    )
                except RuntimeError:
                    # Teardown: on_finished() has already shut the pool down
                    # while requests are still in flight (this benchmark's
                    # requests never complete). Do the work on the loop rather
                    # than dropping it -- conserving it matters more than which
                    # thread pays for it after the measurement window closed.
                    chunks = generation_result.offload_step(item)
                for out in chunks:
                    self.responses_yielded += 1
                    yield out


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------


class _DispatchWork(Architecture):
    """Base for every point on the curve."""

    egress = "push"
    dispatch_mode = "inline"
    build_offloaded = True
    workers = 0
    consume = "prebuilt"
    executor_threads = 0
    credit_window = 0

    def __init__(self) -> None:
        self._llm: Optional[DispatchWorkLLM] = None
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._handler: Optional[DispatchWorkHandler] = None

    def build_llm(self, engine_config: EngineConfig, costs: Costs) -> FakeLLM:
        self._llm = DispatchWorkLLM(
            engine_config,
            costs,
            dispatch_mode=self.dispatch_mode,
            build_offloaded=self.build_offloaded,
            workers=self.workers,
            credit_window=self.credit_window,
        )
        return self._llm

    def build_handler(
        self,
        llm: FakeLLM,
        costs: Costs,
        records: Dict[str, RequestRecord],
    ) -> Any:
        if self.executor_threads:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.executor_threads,
                thread_name_prefix="loop-offload",
            )
        self._handler = DispatchWorkHandler(
            llm,
            costs=costs,
            records=records,
            consume=self.consume,
            executor=self._executor,
        )
        return self._handler

    def build_driver(
        self, handler: Any, py_loop: Any, tokio: TokioRuntime, costs: Costs
    ) -> Driver:
        # Unchanged: this experiment is about WHERE the response work runs,
        # not about the Rust egress shape. push_send stays on the loop.
        return PushDriver(handler, py_loop, tokio, costs)

    def on_finished(self, llm: FakeLLM, driver: Driver) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def extra_report(self) -> Dict[str, Any]:
        llm = self._llm
        if llm is None:
            return {}
        executor_mode = self.consume == "executor"
        # In executor mode the offload runs from the handler, so count it
        # there; llm.offloaded_responses only sees the dispatch-side path.
        offloaded = (
            self._handler.responses_yielded
            if executor_mode and self._handler is not None
            else llm.offloaded_responses
        )
        report: Dict[str, Any] = {
            "producer": (
                "loop-executor"
                if executor_mode
                else ("dispatch-pool" if self.workers else "dispatch-reader")
            ),
            "producer_threads": self.executor_threads or self.workers or 1,
            #: Responses whose handle/build ran off the loop. Compare with the
            #: benchmark's item count: anything much larger is work spent on
            #: responses the loop never reached.
            "offloaded": offloaded,
            "credit_window": self.credit_window or "unbounded",
        }
        abandoned = sum(cell[0] for cell in llm._abandoned_cells)
        if abandoned:
            report["abandoned_at_shutdown"] = abandoned
        if llm.worker_errors:
            report["worker_errors"] = llm.worker_errors[:2]
        return report


class DispatchHandle(_DispatchWork):
    name = "dispatch-handle"
    description = "handle_response on proxy_dispatch_result_thread (1 thread)"
    dispatch_mode = "inline"
    build_offloaded = False
    consume = "build"


class DispatchBoth(_DispatchWork):
    name = "dispatch-both"
    description = "handle_response + build_response on the dispatch thread (1)"
    dispatch_mode = "inline"
    build_offloaded = True
    consume = "prebuilt"


class _DispatchPool(_DispatchWork):
    dispatch_mode = "pool"
    build_offloaded = True
    consume = "prebuilt"


class DispatchBothP1(_DispatchPool):
    name = "dispatch-both-p1"
    description = "handle+build on a 1-thread pool fed by the reader"
    workers = 1


class DispatchBothP2(_DispatchPool):
    name = "dispatch-both-p2"
    description = "handle+build on a 2-thread pool fed by the reader"
    workers = 2


class DispatchBothP4(_DispatchPool):
    name = "dispatch-both-p4"
    description = "handle+build on a 4-thread pool fed by the reader"
    workers = 4


class DispatchBothP8(_DispatchPool):
    name = "dispatch-both-p8"
    description = "handle+build on an 8-thread pool fed by the reader"
    workers = 8


class _DispatchPoolBP(_DispatchPool):
    """The fair form: same total work as the baseline, split across threads."""

    credit_window = CREDIT_WINDOW


class DispatchBothBP1(_DispatchPoolBP):
    name = "dispatch-both-bp1"
    description = "handle+build on 1 backpressured dispatch worker"
    workers = 1


class DispatchBothBP2(_DispatchPoolBP):
    name = "dispatch-both-bp2"
    description = "handle+build on 2 backpressured dispatch workers"
    workers = 2


class DispatchBothBP4(_DispatchPoolBP):
    name = "dispatch-both-bp4"
    description = "handle+build on 4 backpressured dispatch workers"
    workers = 4


class DispatchBothBP8(_DispatchPoolBP):
    name = "dispatch-both-bp8"
    description = "handle+build on 8 backpressured dispatch workers"
    workers = 8


class _LoopExecutor(_DispatchWork):
    dispatch_mode = "raw"
    build_offloaded = True
    consume = "executor"
    workers = 0


class LoopExecutor1(_LoopExecutor):
    name = "loop-executor-1"
    description = "loop run_in_executor(handle+build), 1 pool thread"
    executor_threads = 1


class LoopExecutor4(_LoopExecutor):
    name = "loop-executor-4"
    description = "loop run_in_executor(handle+build), 4 pool threads"
    executor_threads = 4


for _factory in (
    DispatchHandle,
    DispatchBoth,
    DispatchBothP1,
    DispatchBothP2,
    DispatchBothP4,
    DispatchBothP8,
    DispatchBothBP1,
    DispatchBothBP2,
    DispatchBothBP4,
    DispatchBothBP8,
    LoopExecutor1,
    LoopExecutor4,
):
    register(_factory)
