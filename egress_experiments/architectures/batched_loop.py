# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""**batched-loop** -- handle a whole IPC batch in ONE pass on the asyncio loop.

The engine already ships a whole iteration as one IPC message
(``handle_for_ipc_batched``, ``executor/base_worker.py:1252``). What the shipped
path then does with it (``executor/proxy.py:532`` ``dispatch_result_task``) is
unpack it, ``put_nowait`` each response into its **own** per-request
``AsyncQueue``, and issue one ``_SyncQueue.notify_many`` for the batch. One
ready-deque entry -- and then ``len(batch)`` separate ``asyncio.Event.set()``
calls, which wake ``len(batch)`` separate per-request coroutines, each of which
independently walks:

    Event.set -> Task.__step
      -> drive_push_egress's  `async for response in stream`      (asend frame)
        -> Handler.generate.__anext__                             (asend frame)
          -> generate_locally.__anext__                           (asend frame)
            -> _generate_locally_impl.__anext__                   (asend frame)
              -> GenerationResult.__anext__ -> _aresult_step      (coroutine)
                -> AsyncQueue.get()  (deque popleft + Event.clear)
                  -> _handle_response()        23.97 us   <- REAL per-response work
              -> build_response                50.65 us   <- REAL per-response work
        -> response_sender.send(out)           10.72 us   <- one Rust crossing
      -> park on Event.wait() again

Everything in that picture that is *not* one of the three cost stages is paid
``len(batch)`` times for a message the engine deliberately batched. This
architecture pays it **once**.

What it builds
--------------
``_BatchedLLM.dispatch_result_task`` keeps the proxy's structure -- one IPC
message in, one ``call_soon_threadsafe`` out, so the ready-deque cost per batch
is *identical* to ``notify_many``'s -- but hands the loop the **whole list**
instead of waking N coroutines. On the loop, ``_on_batch`` walks that list once:
per response it runs the real ``GenerationResult._handle_response`` and the real
per-response response-dict build, appends the chunk to one list, and then makes
**one** ``send_batch`` call carrying every chunk in the batch.

The per-request handler coroutine still exists and still runs the full ingress
prologue on the loop (normalize / setup_disagg / prepare_input / sampling_params
/ ``engine_submit``); after ``generate_async`` it registers itself and parks on a
single future for the whole request instead of being woken per response. It is
still wrapped in the **real** ``push_egress_capable`` and still driven by the
real ``drive_push_egress_stream``, advanced exactly once per request, so the
shipped push contract (``send`` x N -> ``close()``) is unchanged.

FIXED-per-call vs PER-ITEM -- the honesty question
--------------------------------------------------
``Costs`` models real work. Batching may only amortise what a *call* costs, not
what an *item* costs. The split used here, stage by stage:

``handle_response`` 23.97 us -- **100 % per item.** ``result.py``'s
``_handle_response`` extends *that response's* cumulative ``token_ids`` into
*that request's* ``CompletionOutput``, propagates its finish reasons and sets
``_done``. There is no per-batch component: a batch of N responses belongs to N
different requests, so the work is N independent accumulations. The
``_aresult_step`` coroutine, the ``aqueue.get()`` and the ``Event`` set/clear
around it are per-call, but they sit *outside* the measured range and are not
part of the 23.97 us -- so this architecture removes them for free, in real
Python, and the benchmark's wall-clock measurement is what shows it.

``trtllm:build_response`` 50.65 us -- **100 % per item.** ``handler_base.py``
lines 1183-1266: ``output.token_ids[tokens_so_far:]`` is a list slice whose size
is that response's token delta; the dict, the ``finish_reason`` /
``stop_reason`` propagation, ``_extract_logprobs`` (aligned to the same
per-choice cursor) and the final chunk's
``sum(len(o.token_ids) for o in res.outputs)`` are all that response's own work.
A batched builder can hoist a handful of per-request invariants
(``self.disaggregation_mode``, ``self.kv_block_size``, and
``num_input_tokens = len(request.get("token_ids", []))``, which ``handler_base``
recomputes at line 1234 on every finishing chunk) -- this module does hoist
them, because that is what the real change would do, but it charges the **full**
50.65 us per response anyway. Claiming otherwise would be deleting work.

``trtllm:push_send`` 10.72 us -- **2.00 us fixed per call, 8.72 us per item.**
This is the one stage with a defensible fixed component, and it is defensible
because ``send_batch`` is a concrete, small change to
``lib/bindings/python/rust/push_egress.rs``. Paid once per ``send()`` call
regardless of payload:

* the ``#[pymethods] ResponseSender::send`` pyo3 dispatch (push_egress.rs:351)
  and the ``Arc<dyn ResponseSink>`` virtual call at :352,
* ``dynamo_nvtx_range!("pybridge.push_send")`` (push_egress.rs:209),
* ``decode_response``'s envelope sniff -- ``downcast::<PyDict>`` +
  ``get_item(intern!("_dynamo_annotated"))`` + ``is_truthy``
  (push_egress.rs:288-298) -- three C-API calls independent of token count,
* ``TypedSink::sender()`` (push_egress.rs:151-156): a ``Mutex::lock`` plus an
  ``Option<mpsc::Sender>::clone``,
* and, on the Python side and *inside* the measured range, the
  ``_nvtx.start_range`` / ``end_range`` pair around each ``send``
  (push_egress.py:192-196).

Paid per item and NOT amortised here: ``depythonize::<Resp>(obj)``
(push_egress.rs:301-303), which walks the dict and allocates a
``serde_json::Value`` per token, and ``tx.try_send(frame)`` (push_egress.rs:221).

2.00 us of 10.72 (18.7 %) is deliberately a **lower bound** on that fixed part;
a pyo3 method call, two nvtx ranges, a mutex and three C-API probes plausibly
cost more. If the split is wrong in the other direction, ``batched-loop-strict``
bounds the damage: it is the identical architecture with the fixed part set to
**zero**, so every response pays the full 85.34 us and the entire measured win
is scheduling overhead that this module really does delete, in real Python, on
the real thread. Run both.

What would change in the real tree
----------------------------------
1. ``lib/bindings/python/rust/push_egress.rs`` -- add
   ``ResponseSender::send_batch(&self, py, objs: &Bound<PyList>)`` next to
   ``send`` (:351): one nvtx range, one ``self.sender()``, then
   ``decode_response`` + ``try_send`` per element. Backpressure keeps the
   ``py.allow_threads(|| tx.blocking_send(..))`` rule of :249 per element.
2. Cross-request batching needs one sink for the batch, not one per request.
   ``response_channel`` (push_egress.rs:385) builds a private
   ``mpsc::channel::<Annotated<Resp>>`` per request, so a batch spanning N
   requests is N channels and ``send_batch`` would degenerate to N calls. The
   change is a worker-level multiplexed sink -- one
   ``mpsc::Sender<(RequestId, Annotated<Resp>)>`` shared by every live request,
   with a single tokio demux task fanning into the per-request streams that
   ``lib/runtime/src/pipeline/network/ingress/push_handler.rs`` already
   consumes. The demux is a hash lookup per item on a **tokio** thread that
   never takes the GIL. That is what ``_BatchSink`` models.
3. ``tensorrt_llm/executor/proxy.py:532`` ``dispatch_result_task`` -- instead of
   ``put_nowait`` per response + ``_SyncQueue.notify_many``, hand the list to a
   single loop-side callback. The per-request ``AsyncQueue`` and its
   ``asyncio.Event`` disappear from the response path entirely.
4. ``components/src/dynamo/trtllm/request_handlers/handler_base.py`` -- the
   ``async for res in generation_result`` loop at :1158 becomes a registration
   plus one ``await`` on a per-request completion future; the body of the loop
   (:1179-1278) moves into the batch callback, verbatim, as a function of
   ``(per-request state, response)``.

Not modelled, and it counts against this design: a single batch callback holds
the loop for ``len(batch) x 85.34 us`` uninterrupted, where the shipped path
interleaves at every response. At the benchmark's geometry that is ~20 ms of
head-of-line blocking for ingress -- and it also lengthens the post-abort
drain, because a ready deque full of whole-batch callbacks takes far longer to
empty than the same responses spread over N tasks.

Result -- a NEGATIVE one at the shipped stage costs
---------------------------------------------------
``python3 -m egress_experiments.bench``, median of 4 sessions, always with
``baseline-push`` in the same session (machine shared with five other
experiments, so absolutes are provisional; ratios are not)::

    baseline-push          9,538 items/s   loop 81.9-82.2 us   all 88.0-93.4 us
    batched-loop           9,169 items/s   loop 82.3-82.6 us   all 93.7-93.9 us  0.961x
    batched-loop-strict    8,949 items/s   loop 84.2-84.4 us   all 95.5-95.7 us  0.938x

Three controls say what that 4 % is, and none of them is "batching does not
work":

1. **The tokio stand-in.** ``rust_bridge.py`` deviation #1 -- the simulator's
   tokio side is Python and HOLDS the GIL where real tokio does not. Under
   saturation the shipped path's 240 per-request ``_consume`` tasks fall behind
   (``drain_ratio`` ~0.84-0.94); one shared consumer draining lists does not
   (~0.98-1.00). Batching therefore causes MORE of that GIL-holding stand-in
   work to actually execute, and every microsecond of it is a microsecond the
   loop waits. Re-run with ``Costs(rust_egress_us=0.0)``, which deletes exactly
   that artefact and touches nothing that runs on the loop, and the gap
   disappears -- median of 3::

       baseline-push         10,307 items/s   loop 81.72 us
       batched-loop          10,383 items/s   loop 82.24 us   1.007x
       batched-loop-strict   10,214 items/s   loop 84.23 us   0.991x

   i.e. all of the deficit came from the Python tokio stand-in, not the loop.
   ``batched-loop-strict`` is instructive here: it carries 2.5 us/item MORE
   modelled loop work than baseline (it amortises nothing) and still lands
   within 0.9 %, so ~1.7 % of scheduling overhead really was removed.
2. **The measurement window.** ``--max-backlog`` cancels *ingress*; it does not
   stop the loop, which keeps draining its ready deque -- and ticking the meter
   -- until the ``done.set`` callback is finally serviced. A deque full of
   whole-batch callbacks takes much longer to empty than the same responses
   spread over N tasks, so batched-loop's window comes out ~17.5 s against
   baseline-push's ~11.8 s, and most of a batched-loop window is the post-abort
   drain (no ingress, tokio catching up). Scored over the SAME 1-6 s slice of
   each run, median of 3::

       baseline-push          9,348 items/s  (full window  9,551)
       batched-loop           9,462 items/s  1.012x  (full window  9,095, 0.952x)
       batched-loop-strict    9,267 items/s  0.991x  (full window  8,916, 0.934x)

3. **The per-item work dominates.** At ``--cost-scale 0.25``, where the same
   absolute scheduling saving is a much larger share of the total, the ordering
   flips hard: **batched-loop 33,176 vs baseline-push 30,549 -- 1.09x**, with
   non-modelled loop overhead falling from 14.6 to 10.2 us/item. The scheduling
   overhead this removes is real and worth ~4 us/item; at 85.34 us/response of
   GIL-holding modelled work it is simply 4 % of the problem.

So: the batch pass does delete N-1 coroutine wakeups, N-1 ``Event`` set/clear
pairs, N-1 four-deep ``__anext__`` round trips and N-1 cross-thread
``call_soon_threadsafe`` calls per IPC message. That is worth about 4 us per
response. It is not worth 85.
"""

from __future__ import annotations

import asyncio
import functools
import threading
import time
from typing import Any, Dict, List, Optional

from egress_experiments import loop_meter
from egress_experiments.architectures import Architecture, register
from egress_experiments.costs import Costs, pad_to, spin
from egress_experiments.dynamo_sim.probes import RequestRecord
from egress_experiments.dynamo_sim.rust_bridge import (
    Driver,
    FakeContext,
    TokioRuntime,
    push_pump,
)
from egress_experiments.dynamo_sim.worker import SamplingParams, push_egress_capable
from egress_experiments.fake_trtllm.engine import EngineConfig
from egress_experiments.fake_trtllm.llm import FakeLLM
from egress_experiments.fake_trtllm.result import GenerationResult, Response
from egress_experiments.nvtx_shim import range_

_perf = time.perf_counter_ns

#: Fixed, payload-independent cost of ONE ``ResponseSender.send`` call, in
#: microseconds, out of the measured 10.72 us. See the module docstring for the
#: line-by-line justification. Deliberately a lower bound.
PUSH_SEND_FIXED_US = 2.00

#: Frame tags on the shared (multiplexed) sink.
_DATA = "data"
_CLOSE = "close"


class _RequestState:
    """Everything the batch pass needs to build one request's chunks.

    Replaces the per-request coroutine that the shipped path parks on an
    ``asyncio.Event``. The per-choice cursor and ``num_input_tokens`` live here
    because they are per-REQUEST invariants that ``handler_base`` currently
    keeps as locals of ``_generate_locally_impl`` (:1079, :1234).
    """

    __slots__ = (
        "client_id",
        "request_id",
        "result",
        "record",
        "sender",
        "num_input_tokens",
        "cursor",
        "done",
    )

    def __init__(
        self,
        client_id: int,
        request_id: str,
        result: GenerationResult,
        record: Optional[RequestRecord],
        sender: "_BatchSender",
        num_input_tokens: int,
        done: "asyncio.Future",
    ) -> None:
        self.client_id = client_id
        self.request_id = request_id
        self.result = result
        self.record = record
        self.sender = sender
        self.num_input_tokens = num_input_tokens
        self.cursor: Dict[int, int] = {}
        self.done = done


class _BatchSink:
    """Worker-level multiplexed sink -- the ``send_batch`` half of the change.

    Stands in for one ``mpsc::Sender<(RequestId, Annotated<Resp>)>`` shared by
    every live request, with a tokio-side demux. Called from the loop, under the
    GIL the batch pass already holds, exactly like ``ResponseSender::send``.
    """

    def __init__(
        self,
        tokio_loop: asyncio.AbstractEventLoop,
        queue: "asyncio.Queue",
        costs: Costs,
        fixed_us: float,
    ) -> None:
        self._tokio_loop = tokio_loop
        self._queue = queue
        self._costs = costs
        #: Amortised over the batch: pyo3 dispatch, nvtx range, envelope sniff,
        #: mutex + Sender clone.
        self._fixed_us = costs.scaled(fixed_us)
        #: Charged per response, always: depythonize + try_send.
        self._per_item_us = costs.scaled(max(0.0, costs.push_send_us - fixed_us))
        self.batches = 0
        self.items = 0
        self.send_threads: Dict[str, int] = {}

    @property
    def per_item_us(self) -> float:
        return self._per_item_us

    def send_batch(self, outs: List[dict], records: List[Any]) -> None:
        """One Python->Rust crossing carrying ``len(outs)`` responses.

        The per-item half (``depythonize`` + ``try_send``) has already been
        charged by the batch pass, response by response, as it built each
        chunk -- see ``_BatchedLLM._on_batch``. What is charged here is the
        part a ``send_batch`` really does amortise.
        """
        if not outs:
            return
        name = threading.current_thread().name
        self.send_threads[name] = self.send_threads.get(name, 0) + len(outs)
        self.batches += 1
        self.items += len(outs)
        with range_("trtllm:push_send", color="cyan"):
            # ONE crossing: nvtx range, pyo3 dispatch, Mutex::lock +
            # Sender::clone, envelope sniff. Charged once for the batch.
            spin(self._fixed_us)
            self._tokio_loop.call_soon_threadsafe(
                self._queue.put_nowait, (_DATA, outs, records)
            )

    def send_one(self, out: dict, record: Any) -> None:
        """Compatibility path for a lone ``ResponseSender.send``."""
        start = _perf()
        loop_meter.item()
        pad_to(start, self._per_item_us)
        self.send_batch([out], [record])

    def close(self, error: Optional[str] = None) -> None:
        self._tokio_loop.call_soon_threadsafe(self._queue.put_nowait, (_CLOSE, error))


class _BatchSender:
    """Per-request handle with the shipped ``ResponseSender`` surface.

    ``push_egress.py``'s ``drive_push_egress`` calls exactly ``send`` /
    ``close`` / ``close_with_error`` on this, so the shipped driver runs
    unmodified. In this architecture the handler yields nothing, so ``send`` is
    only a compatibility path (a one-element batch); the real traffic goes
    through :meth:`_BatchSink.send_batch` from the batch pass.
    """

    __slots__ = ("_sink", "state", "_closed", "sends", "close_calls", "error_calls")

    def __init__(self, sink: _BatchSink) -> None:
        self._sink = sink
        self.state: Optional[_RequestState] = None
        self._closed = False
        self.sends = 0
        self.close_calls = 0
        self.error_calls = 0

    def send(self, obj: Any) -> None:
        self.sends += 1
        if self.state is not None:
            self._sink.send_one(obj, self.state.record)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.close_calls += 1
        self._sink.close()

    def close_with_error(self, message: str) -> None:
        if self._closed:
            return
        self._closed = True
        self.error_calls += 1
        self._sink.close(message)


class _BatchedLLM(FakeLLM):
    """``dispatch_result_task`` that hands the loop a LIST, not N wakeups."""

    def __init__(self, engine_config: EngineConfig, costs: Costs) -> None:
        super().__init__(engine_config, costs=costs)
        self.states: Dict[int, _RequestState] = {}
        self.sink: Optional[_BatchSink] = None
        #: One per IPC message: how many responses the loop absorbed per entry.
        self.loop_batches = 0
        self.loop_batch_items = 0
        self.dropped_late = 0

    # -- registration ------------------------------------------------------

    def register(self, state: _RequestState) -> None:
        """Called on the loop, with no await since ``generate_async`` returned.

        That matters: ``_on_batch`` also runs on the loop, so it cannot
        interleave between the engine accepting the request and the state being
        visible, and the first iteration's response can never be dropped.
        """
        self.states[state.client_id] = state

    # -- proxy_dispatch_result_thread --------------------------------------

    def dispatch_result_task(self) -> bool:
        """Port of ``proxy.py:532`` with the per-response fan-out removed.

        Structurally identical up to the hand-off: one message off the IPC lane,
        one entry onto the loop's ready deque. What changes is *what* that entry
        carries -- the list, rather than a ``notify_many`` that sets N Events.
        """
        engine = self._engine
        if engine is None:
            return False  # shutdown raced us
        res = engine.result_link.parent.get(timeout=0.25)
        if res is None:
            return not self._stop.is_set()

        iteration = range_("_handle_responses", color="green")
        iteration.__enter__()

        batch = res if isinstance(res, list) else [res]
        keep: List[Response] = []
        for item in batch:
            if item is None:
                iteration.__exit__()
                return False  # shutdown
            self.responses_dispatched += 1
            keep.append(item)

        self.ipc_messages += 1
        self.ipc_times.append(_perf())
        self.ipc_batch_sizes.append(len(keep))

        if keep and self._loop is not None:
            try:
                # ONE ready-deque entry for the whole message -- the same cost
                # _SyncQueue.notify_many pays, and the same count.
                self._loop.call_soon_threadsafe(self._on_batch, keep)
                self.notify_many_calls += 1
            except RuntimeError:
                iteration.__exit__()
                return False
        iteration.__exit__()
        return True

    # -- ON THE LOOP -------------------------------------------------------

    def _on_batch(self, batch: List[Response]) -> None:
        """One pass over an entire engine iteration. Runs on the loop thread."""
        sink = self.sink
        assert sink is not None
        states = self.states
        costs = self.costs
        build_us = costs.scaled(costs.build_response_us)
        # The per-item half of push_send -- depythonize + try_send -- charged
        # per RESPONSE, never amortised. Hoisted out of the loop because it is
        # a constant, which is the only thing "vectorising" legitimately buys.
        send_item_us = sink.per_item_us

        self.loop_batches += 1
        self.loop_batch_items += len(batch)

        outs: List[dict] = []
        records: List[Any] = []
        finished: List[_RequestState] = []

        for response in batch:
            state = states.get(response.client_id)
            if state is None:
                # Late response for an already-finalised request. proxy.py:546
                # drops these too.
                self.dropped_late += 1
                continue

            # ---- handle_response: 23.97 us, per RESPONSE, unchanged --------
            # The real GenerationResult._handle_response, so the cost, the NVTX
            # name and the cumulative token accumulation are the shipped ones.
            # What is gone is only the machinery AROUND it: _aresult_step,
            # AsyncQueue.get, the Event set/clear and the Task wakeup.
            result = state.result
            result._handle_response(response)

            # ---- build_response: 50.65 us, per RESPONSE --------------------
            # Per-request invariants hoisted out of the per-response body --
            # handler_base recomputes num_input_tokens at :1234 on every
            # finishing chunk and re-reads self.disaggregation_mode at :1211 on
            # every response. Hoisting them is what the real batched builder
            # would do; the FULL 50.65 us is still charged per response.
            cursor = state.cursor
            num_input_tokens = state.num_input_tokens
            res_finished = result.finished
            outputs = result.outputs
            record = state.record

            for output in outputs:
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

                    if out.get("finish_reason") or res_finished:
                        if not out.get("finish_reason"):
                            out["finish_reason"] = "unknown"
                        total_completion_tokens = sum(len(o.token_ids) for o in outputs)
                        out["completion_usage"] = {
                            "prompt_tokens": int(num_input_tokens),
                            "completion_tokens": int(total_completion_tokens),
                            "total_tokens": int(
                                num_input_tokens + total_completion_tokens
                            ),
                            "prompt_tokens_details": None,
                        }

                    pad_to(start, build_us)

                cursor[output_idx] = next_total_toks

                # ---- push_send, per-item half: depythonize + try_send ------
                # Charged in full, per response. The loop is finished with the
                # item here, so this is where the meter ticks -- once per
                # RESPONSE, on the loop thread.
                sent = _perf()
                loop_meter.item()
                pad_to(sent, send_item_us)

                outs.append(out)
                records.append(record)

            if res_finished:
                finished.append(state)

        # ---- ONE crossing for the whole batch -----------------------------
        if outs:
            sink.send_batch(outs, records)

        for state in finished:
            self.states.pop(state.client_id, None)
            with self._results_lock:
                self._results.pop(state.client_id, None)
            if not state.done.done():
                state.done.set_result(None)

    # -- reporting ---------------------------------------------------------

    @property
    def responses_per_loop_batch(self) -> float:
        if not self.loop_batches:
            return 0.0
        return self.loop_batch_items / self.loop_batches


class _BatchedHandler:
    """The worker handler with the per-response ``async for`` removed.

    Everything before ``generate_async`` is byte-for-byte the shipped prologue
    and still runs on the loop, once per request. After it, the handler
    registers its state and parks on ONE future for the whole request instead of
    being resumed once per response.

    ``push_egress_capable`` stays OUTERMOST, so this is still driven by the real
    ``drive_push_egress_stream`` and still advanced exactly once per request.
    """

    def __init__(
        self,
        llm: _BatchedLLM,
        costs: Optional[Costs] = None,
        records: Optional[Dict[str, RequestRecord]] = None,
    ) -> None:
        self.llm = llm
        self.costs = costs or Costs()
        self.records: Dict[str, RequestRecord] = records if records is not None else {}
        self.responses_yielded = 0

    @push_egress_capable
    async def generate(self, request: dict, context: Any):
        async for out in self.generate_locally(request, context):
            yield out

    async def generate_locally(self, request: dict, context: Any):
        with range_("trtllm:generate_locally", color="blue"):
            async for out in self._generate_locally_impl(request, context):
                yield out

    async def _generate_locally_impl(self, request: dict, context: Any):
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

        # NO await between generate_async and register(): _on_batch runs on this
        # same loop, so registration cannot lose the first response.
        sender = getattr(context, "response_sender", None)
        done = asyncio.get_running_loop().create_future()
        state = _RequestState(
            client_id=generation_result.client_id,
            request_id=request["id"],
            result=generation_result,
            record=record,
            sender=sender,
            num_input_tokens=len(request.get("token_ids") or []),
            done=done,
        )
        if sender is not None:
            sender.state = state
        self.llm.register(state)

        # One park for the WHOLE request, in place of one wakeup per response.
        await done
        if False:  # pragma: no cover - never runs; makes this an async generator
            yield


class _BatchedPushDriver(Driver):
    """``PythonPushEngine``, with one shared sink instead of one per request."""

    mode = "push"

    def __init__(
        self,
        handler: Any,
        py_loop: asyncio.AbstractEventLoop,
        tokio: TokioRuntime,
        costs: Costs,
    ) -> None:
        super().__init__(handler, py_loop, tokio, costs)
        #: MUST stay 0 -- a yield here is push_forward_yield, which puts the
        #: per-response GIL acquisition straight back.
        self.fallback_yields = 0
        self.senders: List[_BatchSender] = []
        #: The tokio-side receiver of the multiplexed channel.
        self.sink_queue: "asyncio.Queue" = asyncio.Queue()
        self.sink: Optional[_BatchSink] = None
        self._consumer: Optional["asyncio.Future"] = None

    async def run(self, request: dict, record: RequestRecord) -> None:
        if self._consumer is None:
            # Created here because `run` is on the tokio loop, which is where
            # the queue must be awaited from.
            self._consumer = asyncio.ensure_future(self._consume())

        context = FakeContext(request["id"])
        record.accepted_ns = _perf()

        assert self.sink is not None
        sender = _BatchSender(self.sink)
        self.senders.append(sender)
        context.response_sender = sender

        # engine.rs:85-114 / push_egress.rs:475 -- one spawn_blocking GIL
        # acquisition per REQUEST, identical to the shipped push path.
        stream = await self.spawn_blocking(
            functools.partial(
                self.handler.generate, request, context, response_sender=sender
            )
        )
        anext = stream.__anext__

        counter = [0]
        self.loop_handoffs += 1
        pump = asyncio.run_coroutine_threadsafe(push_pump(anext, counter), self.py_loop)
        try:
            await asyncio.wrap_future(pump)
        except asyncio.CancelledError:
            try:
                pump.cancel()
            except RuntimeError:
                pass
            raise
        except Exception as exc:  # pragma: no cover - defensive
            self.errors.append(f"{type(exc).__name__}: {exc}")
            sender.close()
        self.fallback_yields += counter[0]

    async def _consume(self) -> None:
        """The tokio demux task. No GIL on the real worker.

        One hash lookup and one forward per item, which is exactly what the
        multiplexed ``mpsc<(RequestId, Annotated<Resp>)>`` would cost on a tokio
        thread. ``rust_egress_us`` (chunk+encode+publish) is still charged PER
        RESPONSE -- batching does not make the wire work go away.
        """
        queue = self.sink_queue
        on_item = self._on_item
        while True:
            frame = await queue.get()
            if frame[0] == _DATA:
                _, outs, records = frame
                for out, record in zip(outs, records):
                    on_item(out, record)
            else:
                error = frame[1]
                if error:
                    self.errors.append(error)


class BatchedLoop(Architecture):
    """Whole-IPC-batch processing on the loop, with a batched Rust crossing."""

    name = "batched-loop"
    description = "one loop pass per IPC batch + send_batch (2.00 us/call fixed)"
    egress = "push"

    #: Fixed, payload-independent microseconds per ``send`` call that a
    #: ``send_batch`` amortises over the batch. See the module docstring.
    push_send_fixed_us: float = PUSH_SEND_FIXED_US

    def __init__(self) -> None:
        self._llm: Optional[_BatchedLLM] = None
        self._driver: Optional[_BatchedPushDriver] = None
        self._sink: Optional[_BatchSink] = None

    def build_llm(self, engine_config: EngineConfig, costs: Costs) -> FakeLLM:
        self._llm = _BatchedLLM(engine_config, costs)
        return self._llm

    def build_handler(self, llm, costs, records) -> Any:
        return _BatchedHandler(llm, costs=costs, records=records)

    def build_driver(self, handler, py_loop, tokio, costs) -> Driver:
        driver = _BatchedPushDriver(handler, py_loop, tokio, costs)
        assert tokio.loop is not None
        sink = _BatchSink(tokio.loop, driver.sink_queue, costs, self.push_send_fixed_us)
        driver.sink = sink
        self._sink = sink
        self._driver = driver
        assert self._llm is not None
        self._llm.sink = sink
        return driver

    def extra_report(self) -> Dict[str, Any]:
        llm = self._llm
        sink = self._sink
        driver = self._driver
        report: Dict[str, Any] = {
            "push_send_fixed_us": self.push_send_fixed_us,
            "push_send_per_item_us": round(
                max(0.0, Costs().push_send_us - self.push_send_fixed_us), 2
            ),
        }
        if llm is not None:
            report["loop_batches"] = llm.loop_batches
            report["responses_per_loop_batch"] = round(llm.responses_per_loop_batch, 1)
            report["dropped_late"] = llm.dropped_late
        if sink is not None:
            report["send_batch_calls"] = sink.batches
            report["items_per_send_batch"] = round(
                sink.items / sink.batches if sink.batches else 0.0, 1
            )
            off_loop = {k: v for k, v in sink.send_threads.items() if k != "MainThread"}
            if off_loop:
                report["!! send_batch off the loop"] = off_loop
        if driver is not None:
            report["fallback_yields"] = driver.fallback_yields
            # End-to-end: items the tokio side actually got out, over items the
            # loop finished with. Under saturation the shipped path's 240
            # per-request consumer tasks fall behind; one shared consumer
            # draining lists does not. Worth reading next to items/s, which
            # scores the loop only.
            through_loop = len(loop_meter.timestamps())
            report["delivered_end_to_end"] = driver.delivered
            report["drain_ratio"] = round(
                driver.delivered / through_loop if through_loop else 0.0, 3
            )
        return report


class BatchedLoopStrict(BatchedLoop):
    """Same architecture, ZERO amortisation of any modelled stage.

    Every response pays the full ``23.97 + 50.65 + 10.72 = 85.34 us``. Whatever
    this wins is scheduling overhead that the batch pass really does delete --
    N coroutine wakeups, N ``Event`` set/clear pairs, N ``__anext__`` round
    trips through four nested async generators -- measured, not modelled.
    """

    name = "batched-loop-strict"
    description = "one loop pass per IPC batch, NO stage amortisation at all"
    push_send_fixed_us: float = 0.0


register(BatchedLoop)
register(BatchedLoopStrict)
