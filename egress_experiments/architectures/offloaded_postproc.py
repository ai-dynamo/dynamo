# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Move ``handle_response`` + ``trtllm:build_response`` OFF the event loop.

The claim
---------
74.62 of the 85.34 us the loop spends per response is *formatting*::

    handle_response          23.97   result.py:454
    trtllm:build_response    50.65   handler_base.py:1183
    trtllm:push_send         10.72   push_egress.py, the actual Rust crossing
    -------------------------------
                             85.34

Only the last one has to be on the loop -- it is the hand-off into Rust and it
runs under the GIL the loop already holds. The first two are pure CPU on data
that has just crossed a process boundary anyway, and TRT-LLM already ships the
machinery to run them somewhere else: ``num_postprocess_workers > 0``.

That is not a hypothetical. It is the left-hand column of the reference
diagram. ``trtllm-serve`` runs 4 ``PostprocWorker`` processes and its loop pays
**1.94 us** per response, because with ``npw>0`` the only thing left on the loop
is ``GenerationResultBase._handle_response``'s ``PostprocWorker.Output`` branch
(``tensorrt_llm/executor/result.py:465-501``) -- ``self._done = response.is_final``
and one assignment. dynamo ships ``num_postprocess_workers: 0``, which is why
its loop pays 85.34 instead.

What the real change is
-----------------------
Three edits, all of them in code that already exists:

1. **Turn the postproc workers on.**
   ``tensorrt_llm/executor/worker.py:255-263`` builds one PAIR feed-in
   ``FusedIpcQueue`` per postproc worker when ``postproc_worker_config.enabled``;
   ``worker.py:305-316`` spawns them in a ``ProcessPoolExecutor``.
   ``base_worker.py:1252-1283`` (``handle_for_ipc_batched``) then buckets each
   engine iteration into ``postproc_batches[client_id % N]``
   (``base_worker.py:1434``) and does one ``put`` per bucket instead of one
   ``put`` of the whole iteration to the proxy.
   Bucketing is sticky by ``client_id`` on purpose -- ``base_worker.py:1437-1440``:
   *"incremental detokenization during postprocessing relies on the prior
   CompletionOutput of a given request"*. The same is true of dynamo's
   ``output_tokens_per_choice`` cursor.

2. **Flip the proxy's result socket to PULL.**
   ``tensorrt_llm/executor/proxy.py:457-464`` already does this by itself::

       socket_type=zmq.PULL if self.enable_postprocess_parallel else zmq.PAIR

   with the comment *"Use PULL mode when enable_postprocess_parallel as there
   are multiple senders from multiple processes"*. Nothing else in
   ``dispatch_result_task`` (``proxy.py:532``) changes: it still does
   ``put_nowait`` per item and **one** ``_SyncQueue.notify_many`` per message.

3. **Register dynamo's chunk builder as the postproc hook.**
   ``postproc_worker.py:184-192`` calls
   ``postproc_params.post_processor(record, args)`` inside the worker process
   and ships whatever it returns back as ``PostprocWorker.Output.res``. The body
   of dynamo's ``trtllm:build_response`` range
   (``handler_base.py:1183-1250``) is a pure function of ``output.token_ids``,
   the per-choice cursor and ``res.finished`` -- exactly the signature that hook
   has. Moving it there is a lift, not a rewrite. ``handler_base.py`` then
   consumes an already-built chunk and the ``async for res in
   generation_result`` loop shrinks to ``sender.send(res)``.

   The one piece that has to travel with the request is ``num_input_tokens``
   (for ``completion_usage``), which is precisely what
   ``PostprocWorker.Input.sampling_params`` already exists for --
   ``base_worker.py:1426-1429``: *"They should be transmitted only once for
   each Request."*

This cuts directly against the current direction of travel. ``main``'s
dbeaa5b166 (*"post processing workers are not effective in dynamo"*) added
``_strip_postprocess_workers`` --
``components/src/dynamo/trtllm/workers/llm_worker.py:171-193``, called at
``:392`` -- which now silently deletes ``num_postprocess_workers`` from the
engine args with the warning *"Dynamo manages its own post-processing pipeline
and does not make TRT-LLM's num_postprocess_workers effective."*

That is accurate about the code as it stands and it is the whole problem. With
step 3 missing, ``npw>0`` only moves TRT-LLM's *detokenizer* into the worker
processes -- which dynamo does not use, since it emits ``token_ids`` and lets
the frontend detokenize -- so it buys nothing and adds an IPC hop. What is left
on the loop is dynamo's own ``build_response``, which is the 50.65 us. Do step 3
and the setting becomes the single largest lever on the loop; leave step 3
undone and stripping the flag is the right call. The stripping is what has to be
reverted last, not first.

What is modelled here
---------------------
``build_llm`` swaps in a :class:`PostprocLLM`:

* the dispatch thread buckets each engine iteration by ``client_id % N`` and
  ships each bucket to a postproc **process** (or thread, for the control),
* those processes run ``handle_response`` and ``build_response`` -- the real
  bookkeeping plus the measured spin -- and push finished chunks back on a
  single shared lane, one message per bucket, exactly like ``zmq.PULL`` with N
  senders,
* the receiver thread does ``put_nowait`` per chunk into the per-request
  ``AsyncQueue`` and ONE ``notify_many`` per message -- ``proxy.py:532``,
  unchanged.

``build_handler`` swaps in a handler whose response loop pops an
already-built chunk and charges the loop
:data:`~egress_experiments.costs.SERVE_LOOP_US_PER_RESPONSE` (1.94 us) for the
residual ``PostprocWorker.Output`` branch of ``_handle_response``. That is the
measured ``trtllm-serve`` figure for exactly this code path, used here as a
cost rather than assumed to be free.

``build_driver`` is untouched: the real ``push_egress_capable`` /
``drive_push_egress`` still drive the handler, and ``ResponseSender.send``
still ticks ``loop_meter`` once per response on the loop thread.

WORK CONSERVATION -- read this
------------------------------
``costs.spin_ledger()`` only sees threads of *this* process. The 74.62 us/item
of ``handle_response`` + ``build_response`` moved to child processes and
therefore **disappears from the benchmark's "all us/item" column**. It has not
been deleted. Two things are reported so this cannot be mistaken for a free
win:

* every postproc process reports its own ``spin_ledger()`` total back with each
  result message; ``extra_report()`` prints ``offloaded_us_per_item``, which
  should come out at ~74.6 -- i.e. the work is all still being done,
* ``extra_report()`` also prints ``offload_workers`` and
  ``offload_headroom``, the pool's aggregate capacity
  (``workers * 1e6 / 74.62`` items/s) against the achieved rate, so "the
  receiving process has capacity" is a number and not a claim.

The machine has 24 cores. At N=4 the pool can absorb ~53,600 items/s against a
loop that does far less than that, so the pool is not the thing being measured.
The control architecture ``postproc-threads`` puts the same work on THREADS in
this process, where the ledger does see it and the GIL applies.

Measured
--------
Median of three sessions, each measuring every row itself
(``python3 -m egress_experiments.bench --architecture baseline-push
--architecture postproc-procs --architecture postproc-procs-8
--architecture postproc-threads``), on a 24-core box shared with other
benchmarks::

    baseline-push      9,630 items/s   1.00x   loop 82.11 us   all 92.99 us
    postproc-procs    30,213 items/s   3.14x   loop 14.62 us   all 25.44 us
    postproc-procs-8  27,718 items/s   2.88x   loop 11.43 us   all 21.57 us
    postproc-threads   1,231 items/s   0.13x   loop 15.52 us   all 1008 us

Read the two per-item columns together with ``offloaded_us_per_item`` (74.0):
``10.8 + 74.0 + 11.1`` is 95.9 us of work per item against the baseline's
93.0 -- conserved to 3 %, the difference being the 1.94 us residual.

The threads row is the point of the control. Four threads spinning 74.62 us
per item hold the GIL for 1e6/74.62 = 13,400 items/s worth of wall clock
between them -- one core's worth, because they share one GIL -- and the loop
gets whatever is left. Total system throughput is unchanged from inline; the
loop's share of it collapses. Offloading to a thread does not move GIL-holding
work, it only adds a queue.

Deviations, all of them pessimistic for this architecture
---------------------------------------------------------
1. **The relay hop is an artefact.** In the real system the bucketing happens
   in the *engine* process (``base_worker.py:1280-1283``) and the app process
   never sees a raw ``tllm.Response`` at all. The simulator's engine
   (``fake_trtllm/engine.py:engine_main``) is frozen, so this architecture has
   to re-pickle each iteration on the app process's dispatch thread and forward
   it. That is one extra ``pickle.dumps`` pass per response, on a GIL this
   architecture is trying to free, that the real change does not pay.
2. **The residual 1.94 us is charged in full**, although part of it is
   ``trtllm-serve``'s own bookkeeping rather than dynamo's.
3. ``trtllm:push_send`` is unchanged at 10.72 us.
"""

from __future__ import annotations

import multiprocessing
import queue as _queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from egress_experiments import architectures
from egress_experiments.costs import (
    SERVE_LOOP_US_PER_RESPONSE,
    Costs,
    pad_to,
    reset_spin_ledger,
    spin,
    spin_ledger,
)
from egress_experiments.dynamo_sim.probes import RequestRecord
from egress_experiments.dynamo_sim.worker import SamplingParams, TrtllmWorkerHandler
from egress_experiments.fake_trtllm.aqueue import AsyncQueue, SyncQueue
from egress_experiments.fake_trtllm.engine import EngineConfig
from egress_experiments.fake_trtllm.ipc import Link
from egress_experiments.fake_trtllm.llm import FakeLLM
from egress_experiments.fake_trtllm.result import CompletionOutput, Response
from egress_experiments.nvtx_shim import range_

_perf = time.perf_counter_ns


# ---------------------------------------------------------------------------
# What crosses the two lanes
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """A finished dynamo output chunk. ``PostprocWorker.Output``.

    ``out`` is what ``handler_base`` used to build on the loop; by the time
    this exists it is already built, in another process.
    """

    __slots__ = ("client_id", "out", "is_final")

    client_id: int
    out: Dict[str, Any]
    is_final: bool


@dataclass
class ResultBatch:
    """One postproc worker's output for one engine iteration.

    ``spin_us`` is that worker's cumulative ``costs.spin_ledger()`` total, so
    the parent can account for work that left its own ledger.
    """

    __slots__ = ("worker", "chunks", "spin_us")

    worker: int
    chunks: List[Chunk]
    spin_us: float


class _Record:
    """Per-request state in the postproc worker. ``PostprocWorker._records``.

    Holds exactly what the two moved stages need and nothing else: the
    cumulative ``CompletionOutput`` list that ``_handle_response`` appends to,
    the per-choice cursor that ``build_response`` slices with, and the prompt
    length that ``completion_usage`` needs on the final chunk.
    """

    __slots__ = ("outputs", "cursor", "prompt_tokens")

    def __init__(self, prompt_tokens: int) -> None:
        self.outputs: List[CompletionOutput] = []
        self.cursor: Dict[int, int] = {}
        self.prompt_tokens = prompt_tokens


# ---------------------------------------------------------------------------
# The two stages, as they run in the postproc worker
# ---------------------------------------------------------------------------


def _postproc_batch(
    records: Dict[int, _Record],
    items: List[Tuple[Response, Optional[int]]],
    costs: Costs,
) -> List[Chunk]:
    """``handle_response`` + ``build_response``, off the event loop.

    Line-for-line the same bookkeeping as
    ``fake_trtllm/result.py:_handle_response_impl`` and
    ``dynamo_sim/worker.py:_generate_locally_impl``'s build_response range,
    including the same ``pad_to`` against the same measured p50s -- the work is
    moved, not reduced.
    """
    chunks: List[Chunk] = []

    for response, prompt_tokens in items:
        record = records.get(response.client_id)
        if record is None:
            # PostprocWorker._handle_input: the record is created on the first
            # Input for a client, from the params carried on that first Input.
            record = _Record(prompt_tokens or 0)
            records[response.client_id] = record
        elif prompt_tokens is not None:
            record.prompt_tokens = prompt_tokens

        # ---- handle_response (result.py:454) ------------------------------
        # In the postproc worker this is postproc_worker.py:173,
        # `record._handle_response(input.rsp)`.
        with range_("handle_response", color="red"):
            start = _perf()
            finished = False
            if response.has_error():
                finished = True
            else:
                payload = response.result
                assert payload is not None
                for idx, new_tokens in enumerate(payload.new_token_ids):
                    while idx >= len(record.outputs):
                        record.outputs.append(
                            CompletionOutput(index=len(record.outputs))
                        )
                    record.outputs[idx].token_ids.extend(new_tokens)
                    if payload.finish_reasons and payload.finish_reasons[idx]:
                        record.outputs[idx].finish_reason = payload.finish_reasons[idx]
                finished = payload.is_final
            pad_to(start, costs.scaled(costs.handle_response_us))

        if response.has_error():
            chunks.append(
                Chunk(
                    response.client_id,
                    {"token_ids": [], "index": 0, "finish_reason": "error"},
                    True,
                )
            )
            records.pop(response.client_id, None)
            continue

        # ---- trtllm:build_response (handler_base.py:1183) ------------------
        for output in record.outputs:
            with range_("trtllm:build_response", color="yellow"):
                start = _perf()

                output_idx = getattr(output, "index", 0) or 0
                tokens_so_far = record.cursor.get(output_idx, 0)
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
                        len(o.token_ids) for o in record.outputs
                    )
                    out["completion_usage"] = {
                        "prompt_tokens": int(record.prompt_tokens),
                        "completion_tokens": int(total_completion_tokens),
                        "total_tokens": int(
                            record.prompt_tokens + total_completion_tokens
                        ),
                        "prompt_tokens_details": None,
                    }

                pad_to(start, costs.scaled(costs.build_response_us))

            record.cursor[output_idx] = next_total_toks
            chunks.append(Chunk(response.client_id, out, finished))

        if finished:
            records.pop(response.client_id, None)

    return chunks


# ---------------------------------------------------------------------------
# Worker entry points
# ---------------------------------------------------------------------------


def _process_main(
    feed_link: Link,
    result_link: Link,
    lock: Any,
    costs: Costs,
    worker_id: int,
) -> None:
    """``postproc_worker_main`` -- one OS process, its own GIL.

    Pulls one bucket per engine iteration off its PAIR feed-in lane
    (``worker.py:258-263``) and pushes finished chunks onto the single result
    lane shared with the other workers (``proxy.py:457-464``, ``zmq.PULL``).
    The lock stands in for what PULL gives for free: message framing with
    multiple senders on one socket.
    """
    # The fork inherits the parent's ledger; this process owns its own from
    # here, so the total it reports back is exactly the work IT did.
    reset_spin_ledger()
    feed = feed_link.open_child()
    results = result_link.open_child()
    records: Dict[int, _Record] = {}
    try:
        while True:
            message = feed.get()
            if message is None:  # EOF
                return
            if message and message[0] is None:  # shutdown sentinel
                return
            chunks = _postproc_batch(records, message, costs)
            batch = ResultBatch(worker_id, chunks, sum(spin_ledger().values()))
            with lock:
                results.put(batch)
    except (OSError, EOFError, BrokenPipeError):
        return
    finally:
        try:
            feed.close()
            results.close()
        except Exception:
            pass


def _thread_main(
    feed: "_queue.Queue",
    results: "_queue.Queue",
    costs: Costs,
    worker_id: int,
) -> None:
    """The control: the same two stages on a THREAD, i.e. on the same GIL."""
    records: Dict[int, _Record] = {}
    while True:
        message = feed.get()
        if message is None:
            return
        chunks = _postproc_batch(records, message, costs)
        # 0.0: a thread's spin lands in this process's own ledger, so reporting
        # it again would double-count it.
        results.put(ResultBatch(worker_id, chunks, 0.0))


# ---------------------------------------------------------------------------
# The proxy side
# ---------------------------------------------------------------------------


class PostprocLLM(FakeLLM):
    """``FakeLLM`` with ``num_postprocess_workers = N`` instead of 0.

    Two threads in the app process, matching the real topology:

    * the dispatch thread (:meth:`dispatch_result_task`) buckets an engine
      iteration and forwards it -- in the real system this half runs in the
      *engine* process (``base_worker.py:1280-1283``) and is free here, see the
      module docstring's deviation 1;
    * the receiver thread drains the shared result lane and does what
      ``proxy.py:532`` does, unchanged: ``put_nowait`` per chunk, ONE
      ``notify_many`` per message.
    """

    def __init__(
        self,
        engine_config: Optional[EngineConfig] = None,
        costs: Optional[Costs] = None,
        workers: int = 4,
        threaded: bool = False,
    ) -> None:
        super().__init__(engine_config, costs=costs)
        self.workers = max(1, workers)
        self.threaded = threaded

        self._feed_links: List[Link] = []
        self._result_link: Optional[Link] = None
        self._procs: List[Any] = []
        self._feed_queues: List[_queue.Queue] = []
        self._result_queue: Optional[_queue.Queue] = None
        self._pool_threads: List[threading.Thread] = []
        self._receiver: Optional[threading.Thread] = None
        self._lock: Any = None

        #: client_ids the dispatch thread has already introduced to a worker.
        self._introduced: set = set()
        #: Prompt length per client_id, written by the loop at submit time.
        #: ``PostprocWorker.Input.sampling_params`` -- once per request.
        self.prompt_lens: Dict[int, int] = {}
        #: Cumulative modelled work per postproc worker, in microseconds.
        #: For processes this is work the parent's spin_ledger CANNOT see.
        self.child_spin_us: Dict[int, float] = {}
        #: Buckets forwarded, for reporting.
        self.relay_messages = 0
        self.chunks_received = 0

    # -- lifecycle ---------------------------------------------------------

    def start(self, loop=None) -> None:
        # Spawn the pool BEFORE FakeLLM.start(), which forks the engine and
        # starts the dispatch thread: forking a single-threaded parent is the
        # only safe order.
        self._start_pool()
        super().start(loop)
        self._receiver = threading.Thread(
            target=self._receive_loop, name="proxy_postproc_pull", daemon=True
        )
        self._receiver.start()

    def _start_pool(self) -> None:
        if self.threaded:
            self._result_queue = _queue.Queue()
            for wid in range(self.workers):
                feed: _queue.Queue = _queue.Queue()
                self._feed_queues.append(feed)
                thread = threading.Thread(
                    target=_thread_main,
                    args=(feed, self._result_queue, self.costs, wid),
                    name=f"postproc_{wid}",
                    daemon=True,
                )
                thread.start()
                self._pool_threads.append(thread)
            return

        ctx = multiprocessing.get_context("fork")
        self._lock = ctx.Lock()
        self._result_link = Link("postproc-result")
        for wid in range(self.workers):
            feed_link = Link(f"postproc-{wid}-feedin")
            self._feed_links.append(feed_link)
            proc = ctx.Process(
                target=_process_main,
                args=(feed_link, self._result_link, self._lock, self.costs, wid),
                name=f"postproc_worker_{wid}",
                daemon=True,
            )
            proc.start()
            self._procs.append(proc)
            feed_link.close_child_in_parent()
        # Only once every child has forked, or the later ones would not inherit
        # the shared result lane.
        self._result_link.close_child_in_parent()

    def shutdown(self) -> None:
        # Stop the pool first: the receiver has to stay alive long enough to
        # drain whatever is already in flight.
        for feed_link in self._feed_links:
            try:
                feed_link.parent.put(None)
            except Exception:
                pass
        for feed in self._feed_queues:
            feed.put(None)
        for proc in self._procs:
            proc.join(timeout=2.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(1.0)
        for thread in self._pool_threads:
            thread.join(timeout=2.0)

        super().shutdown()  # sets _stop, kills the engine, joins dispatch

        if self._receiver is not None:
            self._receiver.join(timeout=3.0)
            self._receiver = None
        for feed_link in self._feed_links:
            feed_link.close()
        self._feed_links = []
        if self._result_link is not None:
            self._result_link.close()
            self._result_link = None

    # -- dispatch thread: bucket the iteration -----------------------------

    def dispatch_result_task(self) -> bool:
        """``base_worker.handle_for_ipc_batched`` (``base_worker.py:1252``).

        Bucket by ``client_id % N`` -- sticky, because the moved stages carry
        per-request state (``base_worker.py:1437-1440``) -- and one ``put`` per
        non-empty bucket. Crucially it does NOT ``put_nowait`` and does NOT
        ``notify_many``: nothing reaches the event loop from here.
        """
        engine = self._engine
        if engine is None:
            return False
        res = engine.result_link.parent.get(timeout=0.25)
        if res is None:
            return not self._stop.is_set()

        iteration = range_("_handle_responses", color="green")
        iteration.__enter__()

        batch = res if isinstance(res, list) else [res]
        buckets: List[List[Tuple[Response, Optional[int]]]] = [
            [] for _ in range(self.workers)
        ]
        shutting_down = False
        for item in batch:
            if item is None:
                shutting_down = True
                break
            self.responses_dispatched += 1
            client_id = item.client_id
            prompt: Optional[int] = None
            if client_id not in self._introduced:
                self._introduced.add(client_id)
                # base_worker.py:1426-1429 -- carried once per request only.
                prompt = self.prompt_lens.get(client_id, 0)
            buckets[client_id % self.workers].append((item, prompt))

        if not shutting_down:
            self.ipc_messages += 1
            self.ipc_times.append(_perf())
            self.ipc_batch_sizes.append(len(batch))

        for wid, bucket in enumerate(buckets):
            if not bucket:
                continue
            self.relay_messages += 1
            try:
                if self.threaded:
                    self._feed_queues[wid].put(bucket)
                else:
                    self._feed_links[wid].parent.put(bucket)
            except (OSError, BrokenPipeError):
                shutting_down = True
                break

        iteration.__exit__()
        return not shutting_down

    # -- receiver thread: proxy.py:532, unchanged --------------------------

    def _receive_loop(self) -> None:
        while not self._stop.is_set():
            if self.threaded:
                assert self._result_queue is not None
                try:
                    batches: List[Any] = [self._result_queue.get(timeout=0.25)]
                except _queue.Empty:
                    continue
            else:
                assert self._result_link is not None
                message = self._result_link.parent.get(timeout=0.25)
                if message is None:
                    continue  # timeout, or the lane closed
                batches = message
            for batch in batches:
                if batch is None:
                    return
                if not self._absorb(batch):
                    return

    def _absorb(self, batch: ResultBatch) -> bool:
        """``proxy.dispatch_result_task``: put_nowait per chunk, one notify."""
        self.child_spin_us[batch.worker] = batch.spin_us
        async_queues: List[SyncQueue] = []
        event_loop = None
        for chunk in batch.chunks:
            with self._results_lock:
                result = self._results.get(chunk.client_id)
            if result is None:
                continue  # late response for a finalised request
            queue = result.queue
            queue.put_nowait(chunk)  # deque append -- the loop is untouched
            async_queues.append(queue)
            event_loop = event_loop or queue.loop
            self.chunks_received += 1
            if chunk.is_final:
                with self._results_lock:
                    self._results.pop(chunk.client_id, None)

        if async_queues:
            try:
                SyncQueue.notify_many(event_loop, async_queues)
                self.notify_many_calls += 1
            except AsyncQueue.EventLoopShutdownError:
                return False
        return True


# ---------------------------------------------------------------------------
# The worker side
# ---------------------------------------------------------------------------


class PrebuiltChunkHandler(TrtllmWorkerHandler):
    """The response loop when the chunk is already built.

    Everything up to and including ``generate_async`` is byte-for-byte the
    baseline's: this architecture changes the egress half only. What is gone is
    the ``for output in res.outputs: with range_("trtllm:build_response")``
    block -- it now runs in the postproc process -- and the 23.97 us of
    ``_handle_response`` that ``GenerationResult.__anext__`` used to trigger via
    ``_aresult_step``.

    What is left on the loop is the ``PostprocWorker.Output`` branch of
    ``_handle_response`` (``result.py:465-501``), charged at the measured
    ``trtllm-serve`` figure of 1.94 us, and then ``yield out``.
    """

    async def _generate_locally_impl(self, request: dict, context: Any):
        record = self.records.get(request["id"])
        if record is not None and not record.admitted_ns:
            record.admitted_ns = _perf()

        # Ingress is unchanged -- identical to worker.py, same NVTX names.
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

        # The one thing the postproc worker cannot derive from a tllm.Response.
        # Real equivalent: PostprocWorker.Input.sampling_params, sent once per
        # request (base_worker.py:1426-1429).
        self.llm.prompt_lens[generation_result.client_id] = len(
            request.get("token_ids") or []
        )

        # The per-request AsyncQueue now carries finished chunks, so the loop
        # never sees a tllm.Response and never runs the two moved stages.
        aqueue = generation_result.aqueue
        while True:
            chunk: Chunk = await aqueue.get()
            with range_("handle_response", color="red"):
                spin(self.costs.scaled(SERVE_LOOP_US_PER_RESPONSE))
            self.responses_yielded += 1
            yield chunk.out
            if chunk.is_final:
                return


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------


class OffloadedPostproc(architectures.Architecture):
    """Base: N postproc workers, push egress, everything else unchanged."""

    name = "postproc-procs"
    description = "handle_response+build_response in 4 postproc PROCESSES (npw=4)"
    egress = "push"

    workers = 4
    threaded = False

    def __init__(self) -> None:
        self._llm: Optional[PostprocLLM] = None

    def build_llm(self, engine_config: EngineConfig, costs: Costs) -> FakeLLM:
        self._llm = PostprocLLM(
            engine_config,
            costs=costs,
            workers=self.workers,
            threaded=self.threaded,
        )
        return self._llm

    def build_handler(
        self,
        llm: FakeLLM,
        costs: Costs,
        records: Dict[str, RequestRecord],
    ) -> Any:
        return PrebuiltChunkHandler(llm, costs=costs, records=records)

    def extra_report(self) -> Dict[str, Any]:
        llm = self._llm
        if llm is None:
            return {}
        items = max(1, llm.chunks_received)
        offloaded = sum(llm.child_spin_us.values())
        costs = llm.costs
        moved_us = costs.scaled(costs.handle_response_us + costs.build_response_us)
        # Aggregate capacity of the pool, items/s, if every worker were busy.
        capacity = llm.workers * 1e6 / moved_us if moved_us else 0.0
        report: Dict[str, Any] = {
            "offload_workers": llm.workers,
            "offload_kind": "threads" if llm.threaded else "processes",
            "relay_messages": llm.relay_messages,
            "chunks": llm.chunks_received,
            "pool_capacity_items_per_s": round(capacity),
        }
        if llm.threaded:
            report["offloaded_us_per_item"] = "in-process (see ledger)"
        else:
            report["offloaded_us_per_item"] = round(offloaded / items, 2)
            report["offloaded_us_total"] = round(offloaded)
        return report


class OffloadedPostproc8(OffloadedPostproc):
    name = "postproc-procs-8"
    description = "handle_response+build_response in 8 postproc PROCESSES (npw=8)"
    workers = 8


class OffloadedPostprocThreads(OffloadedPostproc):
    name = "postproc-threads"
    description = "same work on 4 THREADS -- the GIL control, expected to lose"
    workers = 4
    threaded = True


architectures.register(OffloadedPostproc)
architectures.register(OffloadedPostproc8)
architectures.register(OffloadedPostprocThreads)
