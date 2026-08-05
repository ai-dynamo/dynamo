# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""COMPOSITION -- do the small wins get bigger once the big one lands?

The hypothesis
--------------
Measured in isolation on ``bench``, the six experiments rank::

    postproc-procs      28,378 items/s   3.01x    loop  28.5 us/item
    baseline-push        9,423            1.00x   loop 100.9
    batched-admission    9,311            0.99x
    batched-loop         9,310            0.99x
    pump-fanout          9,205            0.98x
    two-queue-priority   9,091            0.96x

Each of the "small" ones removes a *fixed number of microseconds* of
scheduling from the loop, and each was measured against a loop that was
spending ~101 us/item::

    pump-fanout    2.75 us/response of Event/Future/Handle/coroutine-resume
                   (pump_fanout.py, isolated at the capture's geometry)
    batched-loop   ~4 us/response of the same thing plus the ``send_batch``
                   amortisation (batched_loop.py, cost-scale 0.25 control)
    batched-admission  a whole ``spawn_blocking`` + cross-thread
                   ``Python::with_gil`` per REQUEST -- invisible at
                   ``max_tokens=1e6``, 1.25x when ``max_tokens=1``

``postproc-procs`` moves ``handle_response`` (23.97) + ``trtllm:build_response``
(50.65) = 74.62 us of the 85.34 into four child PROCESSES, leaving 1.94 +
10.72 = 12.66 us of modelled loop work per response. If the loop's per-item
cost really falls from ~101 to ~28, then 2.75 us stops being 3 % and becomes
10 %, ~4 us becomes 14 %, and ingress -- 213.10 us of loop work per REQUEST --
stops being a rounding error.

So the compositions here are not "stack the wins". They are a test of whether
the isolated ranking is *measuring the wrong denominator*.

What is built
-------------
Nothing new is invented. Every architecture below is the existing experiments'
own classes, imported and subclassed, wired to each other:

``composed-pp-pump``
    ``PostprocLLM`` (offloaded_postproc) delivering into ``PumpLLM``'s single
    MPSC inbox + pump coroutine (pump_fanout) instead of N per-request
    ``AsyncQueue``s. Per chunk on the loop: one dict lookup and one direct
    call, then the 1.94 us residual and ``ResponseSender.send``.

``composed-pp-admit``
    ``PostprocLLM`` + ``PrebuiltChunkHandler`` (offloaded_postproc) driven by
    ``BatchedAdmissionDriver`` (batched_admission): ingress is an MPSC ring the
    loop drains itself, so no request ever costs a cross-thread GIL
    acquisition. Egress is byte-for-byte ``postproc-procs``.

``composed-pp-batchloop``
    ``PostprocLLM`` whose receiver thread hands the loop a whole **chunk list**
    per postproc message, walked in one pass, ending in one
    ``_BatchSink.send_batch`` (batched_loop). ``composed-pp-batchloop-strict``
    is the same with ``push_send_fixed_us = 0``, i.e. zero amortisation of any
    modelled stage -- whatever it wins is scheduling that was really deleted.

``composed-pp-pump-admit``   pump + ring.
``composed-all``             batch pass + ``send_batch`` + ring.
``composed-all-8``           the same with 8 postproc processes, because a
                             faster loop can outrun a 4-process pool
                             (4 x 1e6/74.62 = 53,600 items/s).

Work conservation
-----------------
Identical to the parents', and for the same reasons:

* the 74.62 us of ``handle_response`` + ``build_response`` is in child
  processes and therefore leaves this process's ``spin_ledger``.
  ``extra_report()`` prints ``offloaded_us_per_item`` (~74.6) from the
  children's own ledgers, so the drop in ``all us/item`` is accounted for and
  not deleted;
* ``SERVE_LOOP_US_PER_RESPONSE`` (1.94) is charged on the loop for the
  ``PostprocWorker.Output`` branch of ``_handle_response``
  (``tensorrt_llm/executor/result.py:465-501``);
* ``trtllm:push_send`` (10.72) is charged in full per response, EXCEPT in
  ``composed-pp-batchloop`` / ``composed-all``, which split it exactly as
  ``batched_loop.PUSH_SEND_FIXED_US`` does -- 2.00 us fixed per ``send_batch``
  call, 8.72 us per item -- with the ``-strict`` variant pinning the fixed part
  to zero as the control;
* ingress (213.10 us/request) is untouched everywhere. ``batched-admission``
  removes *scheduling* -- a ThreadPoolExecutor round trip and one ready-deque
  entry -- which ``Costs`` does not model at all.

Measured -- the hypothesis HOLDS, and it is the batch pass that carries it
-------------------------------------------------------------------------
``bench`` at a FIXED rung (``--batch 600``, so the ladder cannot pick a
different geometry per architecture -- see the note above ``main()``), median of
3, one fresh process per measurement, two concurrent 8-core pinned runs::

    architecture                  items/s   vs base   vs pp   loop us   all us   busy
    composed-pp-batchloop          38,256    4.15x    1.32x     11.13    22.67    43%
    composed-all                   37,005    4.02x    1.28x     11.10    22.72    41%
    composed-pp-batchloop-strict   33,851    3.67x    1.17x     13.04    24.73    44%
    composed-all-strict            33,747    3.66x    1.16x     13.10    24.80    44%
    composed-all-8                 33,116    3.59x    1.14x     11.03    22.69    37%
    composed-pp-pump-admit         31,077    3.37x    1.07x     23.61    35.22    73%
    composed-pp-pump               30,562    3.32x    1.05x     26.97    38.57    82%
    composed-pp-admit              29,671    3.22x    1.02x     25.47    37.06    76%
    postproc-procs                 29,001    3.15x    1.00x     26.48    38.26    77%
    postproc-procs-8               27,114    2.94x    0.93x     28.30    39.27    77%
    baseline-push                   9,213    1.00x    0.32x    112.89   113.08   104%

``busy`` is ``items/s x loop us/item`` -- modelled loop work per second of wall
clock. Read it first: it is what settles the whole question.

The marginal value of each small change, alone versus on top of the big one::

    change                  alone on bench   on top of postproc-procs
    pump-fanout                     0.98x    1.05x  (max_tokens 1e6)
    batched-loop                    0.99x    1.32x  (max_tokens 1e6)
    batched-admission               0.99x    1.02x  (max_tokens 1e6)
                                    1.24x    1.28x  (max_tokens 1, request-heavy)

So yes: ``batched-loop`` goes from *losing* 1 % in isolation to *winning* 32 %
once ``postproc-procs`` has taken 74.62 us off the loop, and the denominator is
exactly why -- 4 us against 101 is 4 %, 4 us against a loop that is now paying
~12 us of modelled work plus one cross-thread ``call_soon_threadsafe`` per
response is most of what is left. ``batched-admission`` improves too, at every
``max_tokens``, and by more than it wins alone.

The ``-strict`` controls are the honest floor. They amortise NOTHING -- every
response pays the full ``1.94 + 10.72`` -- and they still land at 1.16-1.17x,
so at least half of the composed win is scheduling that the batch pass really
deletes and not the ``send_batch`` cost split.

...and then it stops mattering, which is the more useful finding
----------------------------------------------------------------
``busy`` falls from **104 %** (``baseline-push``: the loop is the bottleneck,
which is the whole premise of this study) to **77 %** after ``postproc-procs``
and to **41-43 %** after the batch pass. A loop that is idle 57 % of the time is
not the constraint any more, and three controls agree on where the constraint
went::

    npw=8 instead of 4        composed-all-8 33,116 vs composed-all 37,005
                              postproc-procs-8 27,114 vs postproc-procs 29,001
                              -> NOT the postproc pool. More workers halve the
                                 responses per relay message (148.3 -> 74.2),
                                 i.e. twice the IPC messages and twice the
                                 unpickling, and that costs more than the extra
                                 parallelism buys.

    24 cores instead of 8     postproc-procs 29,188 vs 29,001
                              composed-pp-batchloop 35,518 vs 38,256
                              -> NOT CPU. Sixteen more cores change nothing.

    Costs(rust_egress_us=0)   baseline-push  9,310 (unchanged, 1.01x)
                              postproc-procs 44,700 (1.54x)
                              composed-pp-batchloop 51,711 (1.35x)
                              composed-pp-pump      51,569
                              composed-all          50,969
                              -> the Python "tokio" stand-in. Deleting the one
                                 artefact `rust_bridge.py` deviation #1
                                 documents lifts every postproc architecture by
                                 35-54 % and lifts baseline-push by 1 %.

What is left after that is the app process's GIL, shared by the loop, the
dispatch thread (bucket + ``pickle.dumps`` per relay message) and the receiver
thread (``pickle.loads`` per relay message). All four compositions pile up
against the same ~51,000 items/s ceiling regardless of whether their loop costs
10.89 or 15.53 us/item. **That ceiling is largely an artefact of this
simulator**: ``offloaded_postproc.py``'s deviation 1 -- the relay hop exists only
because ``fake_trtllm/engine.py`` is frozen and cannot bucket in the engine
process, where ``base_worker.py:1280-1283`` does it for real. The real change
pays neither the extra ``pickle.dumps`` nor the extra ``pickle.loads``.

The honest summary: the composition is worth 1.28-1.32x over ``postproc-procs``
and 4.0-4.2x over ``baseline-push`` at the decode geometry, and 1.42x / 1.61x
respectively at the prefill geometry -- but the *reason* the numbers stop
climbing is that after edit 1 + edit 2 the event loop is no longer the thing to
optimise.

max_tokens sweep -- prefill versus decode
------------------------------------------
The disagg PREFILL worker emits exactly one response per request
(``handlers.py:211-212``), i.e. ``max_tokens=1``; decode is the ``1e6`` column.
Request-heavy harness (``batched_admission``'s own, closed loop, 1000 in flight,
batch 240, ``--requests 60000``), median of 3, one process per measurement::

    max_tokens                 1        4       16      1e6*     1e6 (bench, batch 600)
    ingress share          71.4 %   38.4 %   13.5 %    0.0 %
    baseline-push          1,985    4,800    7,417    8,779      9,213
    batched-admission      2,458    5,310    7,809       --          --
    postproc-procs         2,243    7,156   15,700   22,307     29,001
    composed-pp-batchloop  2,633    8,413   19,354   22,531     38,256
    composed-pp-admit      2,880    8,816   17,879       --     29,671
    composed-all           3,185    9,830   21,180   22,276     37,005
    ---- composed-all vs baseline-push ----
                           1.61x    2.05x    2.86x        --      4.02x
    ---- composed-all vs postproc-procs ----
                           1.42x    1.37x    1.35x        --      1.28x

\\* The ``1e6`` column of THIS harness is supply-limited and must not be read as
a throughput result: batch 240 at a 10 ms iteration offers only 24,000
responses/s, and every postproc architecture is already within 7 % of that. The
comparable decode number is the ``bench`` column, where batch 600 offers 60,000.

Two things the sweep settles:

* **``batched-admission`` composes.** Alone it is 1.24x at ``max_tokens=1`` and
  0.99x at ``1e6``. On top of ``postproc-procs`` it is 1.28x / 1.23x / 1.14x /
  1.02x at 1 / 4 / 16 / 1e6 -- better at every geometry than it is alone,
  because the loop it is competing with for admission is cheaper.
* **The prefill worker is the case for admission, and decode is the case for
  the batch pass.** At ``max_tokens=1`` the ring is worth +21 % on top of the
  batch pass (3,185 vs 2,633); at ``1e6`` it is worth -3 % (37,005 vs 38,256),
  because its resident polling coroutine is pure overhead when there is nothing
  to admit. Ship the ring on prefill, not on decode -- or make ``idle_spins``
  adaptive.

The same runs under ``bench``'s DEFAULT ladder, and why they are not the table
------------------------------------------------------------------------------
``python3 -m egress_experiments.bench --architecture X``, median of 3, same
machine and pinning::

    composed-pp-batchloop   40,467   4.29x   batch 1500/1500/1500   runs 40,724 38,377 40,467
    composed-all            32,565   3.45x   batch 1500/1500/1500   runs 31,725 34,320 32,565
    composed-pp-admit       29,761   3.15x   batch 10000/4000/4000  runs 27,313 35,989 29,761
    composed-pp-pump        29,180   3.09x   batch  600/600/600     runs 28,105 29,749 29,180
    composed-all-strict     28,727   3.04x   batch 1500/1500/1500   runs 28,727 28,299 29,508
    postproc-procs          26,686   2.83x   batch 4000/600/4000    runs 26,686 29,753 16,434
    composed-pp-pump-admit  26,272   2.78x   batch  600/600/600     runs 26,272 23,906 30,573
    baseline-push            9,440   1.00x   batch  240/240/240     runs  9,188  9,440  9,492

``baseline-push`` reproduces the reference 9,423 to 0.2 %, so the harness is the
same one. Everything else is the ladder: ``postproc-procs`` scored 26,686 /
29,753 / **16,434** on three identical invocations because it landed on batch
4000 / 600 / 4000, and a batch-4000 run charges 4,000 x 213.10 us of ingress
over a tenth as many delivered items. The FIXED-rung table above is the
controlled comparison; this one is here because it is the mandated command and
because the spread is itself a result.

Work conservation, arithmetic
-----------------------------
Per delivered item, ``bench`` batch 600, medians::

                              loop     tokio   offloaded   total
    baseline-push           112.89      0.49        --    113.38
    postproc-procs           26.48     11.77      74.62   112.87
    composed-pp-pump         26.97     11.60      78.17   116.74
    composed-pp-batchloop-s  13.04     11.65      74.95    99.64
    composed-pp-batchloop    11.13     11.79      77.30   100.22

``postproc-procs`` and ``composed-pp-pump`` conserve to within 3 % of the
baseline. The two batch-pass rows come out ~13 us/item LOW, and the reason is
specific and worth stating rather than hiding: ``ResponseSender.send`` does its
cross-thread ``call_soon_threadsafe`` INSIDE ``pad_to``, so when that hop
overruns 10.72 us the ledger is charged what it really cost;
``_BatchSink.send_batch`` makes that hop once per 148 responses and makes it
outside the padded region, so it is charged nothing. Part of the 13 us is
therefore real work removed (147 of every 148 hops genuinely do not happen) and
part is an accounting asymmetry in ``batched_loop._BatchSink``. Neither affects
``items/s``, which is measured from wall-clock delivery timestamps -- but it
does mean ``loop us/item`` is not directly comparable between the per-response
and the batched senders, and the 1.17x of the ``-strict`` control is the number
to trust for "how much scheduling was really deleted".

Real code this composition implies
----------------------------------
See :data:`REAL_CODE_CHANGES`.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

from egress_experiments import architectures, loop_meter
from egress_experiments.architectures.batched_admission import BatchedAdmissionDriver
from egress_experiments.architectures.batched_loop import (
    _DATA,
    PUSH_SEND_FIXED_US,
    _BatchedPushDriver,
    _BatchSender,
    _BatchSink,
)
from egress_experiments.architectures.offloaded_postproc import (
    Chunk,
    OffloadedPostproc,
    PostprocLLM,
    PrebuiltChunkHandler,
    ResultBatch,
)
from egress_experiments.architectures.pump_fanout import PumpLLM
from egress_experiments.costs import SERVE_LOOP_US_PER_RESPONSE, Costs, pad_to, spin
from egress_experiments.dynamo_sim.probes import RequestRecord
from egress_experiments.dynamo_sim.rust_bridge import Driver
from egress_experiments.dynamo_sim.worker import SamplingParams
from egress_experiments.fake_trtllm.engine import EngineConfig
from egress_experiments.fake_trtllm.llm import FakeLLM
from egress_experiments.nvtx_shim import range_

_perf = time.perf_counter_ns

#: Frame tag for a per-request termination on the multiplexed sink. The
#: batched_loop module's own ``_CLOSE`` carries no request identity because its
#: driver learns about completion from the generator instead; the admission-ring
#: composition has no such future to await, so it needs the tag.
_CLOSE_TAGGED = "close-tagged"


REAL_CODE_CHANGES = """
This composition is three independent edits that touch three different files.
None of them depends on the others compiling, which is why they compose at all.

1. POSTPROC (the 74.62 us) -- offloaded_postproc.py's change, unchanged:
   - `tensorrt_llm/executor/worker.py:255-263` / `:305-316` already spawn the
     `ProcessPoolExecutor` and build the per-worker PAIR feed-in queues when
     `postproc_worker_config.enabled`; `base_worker.py:1252-1283`
     (`handle_for_ipc_batched`) already buckets by `client_id % N`
     (`base_worker.py:1434`, sticky per `:1437-1440`); `proxy.py:457-464`
     already flips the result socket to `zmq.PULL`.
   - What is missing is step 3: register dynamo's chunk builder as the
     `postproc_params.post_processor` hook (`postproc_worker.py:184-192`). The
     body of `trtllm:build_response`
     (`components/src/dynamo/trtllm/request_handlers/handler_base.py:1183-1250`)
     is a pure function of `output.token_ids`, the per-choice cursor and
     `res.finished` -- exactly that hook's signature.
   - `components/src/dynamo/trtllm/workers/llm_worker.py:171-193`
     (`_strip_postprocess_workers`, called at `:392`) has to be reverted, but
     LAST: without step 3 it is the correct call.

2. THE LOOP-SIDE FAN-OUT (pump-fanout / batched-loop) -- the change is now to
   the POSTPROC result path, not the raw response path:
   - `tensorrt_llm/executor/proxy.py:532` `dispatch_result_task` is where the
     `PostprocWorker.Output` batches land. Instead of `put_nowait` per output
     into N `_SyncQueue`s + `_SyncQueue.notify_many` (`llmapi/utils.py:475`,
     which sets N `asyncio.Event`s and therefore costs N `call_soon` +
     N Task steps ON the loop), append the list the PULL socket already handed
     us and issue ONE `loop.call_soon_threadsafe`.
   - `tensorrt_llm/executor/result.py:949` -- `GenerationResult.aqueue` becomes
     lazy; `set_response_callback(cb)` is the new fast path (pump-fanout), or
     the whole list is handed to one loop-side callback (batched-loop).
     `__aiter__` / `aresult()` / `result()` keep today's behaviour at today's
     cost by lazily allocating the queue.
   - `handler_base.py:1158`'s `async for res in generation_result` becomes a
     registration plus one `await` on a per-request completion future. With
     postproc on, the body of that loop is already gone (it moved in edit 1),
     so what is left to move is only `response_sender.send(chunk)`.
   - `lib/bindings/python/rust/push_egress.rs:351` grows
     `ResponseSender::send_batch(&self, py, objs: &Bound<PyList>)`; because a
     batch spans N requests it needs one worker-level multiplexed sink
     (`push_egress.rs:385`'s per-request `response_channel` becomes a shared
     `mpsc::Sender<(RequestId, Annotated<Resp>)>` with a tokio demux feeding
     `lib/runtime/src/pipeline/network/ingress/push_handler.rs`).

3. ADMISSION (batched-admission):
   - `lib/bindings/python/rust/engine.rs:85-115` `invoke_generator` --
     `PythonPushEngine::generate` (`push_egress.rs:459-500`) stops awaiting it
     inline and instead pushes `(python_input, ctx, sender)` onto an admission
     channel, returning the `ResponseStream` immediately. It already builds the
     sender before the crossing, so the ordering works.
   - `push_egress.py` gains a module-level `admission_pump()` coroutine started
     once by the worker, which calls `handler.generate(...)` per drained ticket
     under the GIL the loop already holds. `push_egress_capable` is unchanged.

Ordering note: edits 2 and 3 are only worth doing AFTER edit 1, and that is the
finding this module exists to produce, not an assumption it starts from.
"""


# ---------------------------------------------------------------------------
# Shared: the ingress prologue, verbatim from dynamo_sim/worker.py
# ---------------------------------------------------------------------------


def _prologue(handler: Any, request: dict) -> SamplingParams:
    """``_generate_locally_impl`` up to ``generate_async``.

    Identical to ``dynamo_sim/worker.py:114-126`` and to every other
    architecture's copy of it, including the NVTX names, so a capture of any of
    these runs reads back through ``capture_params`` unchanged.
    """
    costs = handler.costs
    for stage_name, stage_us in (
        ("trtllm:normalize_request", costs.normalize_request_us),
        ("trtllm:setup_disagg_params", costs.setup_disagg_params_us),
        ("trtllm:prepare_input", costs.prepare_input_us),
        ("trtllm:sampling_params", costs.sampling_params_us),
    ):
        with range_(stage_name, color="cyan"):
            spin(costs.scaled(stage_us))

    return SamplingParams(
        max_tokens=int(request.get("max_tokens", 64)),
        n=int(request.get("n", 1)),
    )


def _submit(handler: Any, request: dict, sampling_params: SamplingParams):
    """``llm.generate_async`` plus the one field the postproc worker needs.

    ``prompt_lens`` is ``PostprocWorker.Input.sampling_params``
    (``base_worker.py:1426-1429``): carried once per request, because the
    postproc process cannot derive ``num_input_tokens`` from a ``tllm.Response``.
    """
    generation_result = handler.llm.generate_async(
        inputs=request.get("token_ids"),
        sampling_params=sampling_params,
        disaggregated_params=None,
        streaming=True,
        trace_headers=None,
        scheduling_params=None,
        priority=0.5,
        cache_salt=None,
    )
    handler.llm.prompt_lens[generation_result.client_id] = len(
        request.get("token_ids") or []
    )
    return generation_result


# ===========================================================================
# A.  postproc-procs  +  pump-fanout
# ===========================================================================


class _ChunkFanout:
    """What one request registers with the pump instead of parking a coroutine.

    ``pump_fanout._Fanout`` with the two moved stages taken out: by the time a
    :class:`~egress_experiments.architectures.offloaded_postproc.Chunk` reaches
    here, ``handle_response`` and ``trtllm:build_response`` have already run in
    a postproc process. What is left is exactly what
    ``PrebuiltChunkHandler``'s ``async for`` body did -- the 1.94 us residual
    ``PostprocWorker.Output`` branch and ``ResponseSender.send`` -- reached by
    a dict lookup rather than an ``Event.set`` + ``Future`` + ``Handle`` + a
    six-frame coroutine resumption.
    """

    __slots__ = ("sender", "done", "costs", "handler", "finished", "client_id")

    def __init__(
        self,
        *,
        sender: Any,
        done: "asyncio.Future",
        costs: Costs,
        handler: Any,
        client_id: int,
    ) -> None:
        self.sender = sender
        self.done = done
        self.costs = costs
        self.handler = handler
        self.finished = False
        self.client_id = client_id

    def __call__(self, chunk: Chunk) -> None:
        # result.py:465-501 -- the PostprocWorker.Output branch, at the measured
        # trtllm-serve figure. Charged, not assumed free.
        with range_("handle_response", color="red"):
            spin(self.costs.scaled(SERVE_LOOP_US_PER_RESPONSE))
        self.handler.responses_yielded += 1
        # push_egress.py drive_push_egress: response_sender.send(response).
        # Charges trtllm:push_send and ticks loop_meter exactly once, on the
        # loop.
        self.sender.send(chunk.out)
        if chunk.is_final and not self.finished:
            self.close()

    def close(self) -> None:
        self.finished = True
        done = self.done
        if not done.done():
            done.set_result(None)


class _PostprocPumpLLM(PostprocLLM, PumpLLM):
    """Bucket to postproc processes; deliver the results through ONE pump.

    MRO is ``_PostprocPumpLLM -> PostprocLLM -> PumpLLM -> FakeLLM``, so
    ``dispatch_result_task`` is the bucketing one (nothing reaches the loop from
    the dispatch thread at all) and every pump member -- ``_inbox``, ``_table``,
    ``_wake``, ``_pump`` -- is inherited unchanged.

    The only new code is the join: :meth:`_absorb`, which is
    ``PostprocLLM._absorb`` with the ``put_nowait``-per-chunk +
    ``notify_many`` replaced by one ``deque.append`` + one
    ``call_soon_threadsafe``.
    """

    def __init__(
        self,
        engine_config: Optional[EngineConfig] = None,
        costs: Optional[Costs] = None,
        workers: int = 4,
        threaded: bool = False,
    ) -> None:
        super().__init__(engine_config, costs=costs, workers=workers, threaded=threaded)
        self._closing = False

    # -- dispatch thread ---------------------------------------------------

    def dispatch_result_task(self) -> bool:
        """Bucketing, with :attr:`PumpLLM.inbox_high_water` in front of it.

        The bound is not tuning. Both halves of this composition made the
        producer side cheaper -- the dispatch thread now only buckets and
        forwards, and the receiver only appends -- so when the loop falls behind
        it would drain the lane far faster than the loop consumes and loop
        backpressure would silently become host memory. ``PumpLLM`` documents
        the same hazard for the same reason. ``Event.wait`` releases the GIL, so
        a stalled dispatch thread costs the loop nothing.
        """
        high_water = self.inbox_high_water
        if high_water is not None:
            while (self.responses_dispatched - self.pump_items) > high_water:
                if self._stop.wait(0.002):
                    return False
                self.dispatch_stalls += 1
        return PostprocLLM.dispatch_result_task(self)

    # -- receiver thread ---------------------------------------------------

    def _absorb(self, batch: ResultBatch) -> bool:
        """``proxy.dispatch_result_task`` for ``PostprocWorker.Output`` batches.

        One ``deque.append`` of the list the PULL lane already handed us, then
        one ``call_soon_threadsafe``. No per-request ``AsyncQueue``, no
        ``Event.set``, no ``frozenset(queues)``.
        """
        self.child_spin_us[batch.worker] = batch.spin_us
        chunks = batch.chunks
        # NEVER return False here. The receiver thread stops on a False, and a
        # stopped receiver deadlocks teardown: the result lane fills, all N
        # postproc workers block in `results.put`, their feed lanes fill, and
        # `PostprocLLM.shutdown`'s `feed_link.parent.put(None)` -- a blocking
        # `sendall` with no timeout -- never returns. Observed as a 5-minute
        # hang with MainThread and proxy_dispatch_result_thread both parked in
        # `sock_alloc_send_pskb`. Once the loop is going away the right
        # behaviour is to keep draining and drop, exactly as
        # ``proxy.py:546`` drops responses for a finalised request.
        if not chunks or self._closing:
            return True
        self.chunks_received += len(chunks)
        loop = self._loop
        if loop is None or not loop.is_running():
            return True
        self._inbox.append(chunks)
        try:
            loop.call_soon_threadsafe(self._wake)
        except RuntimeError:
            return True
        self.pump_wakeups += 1
        # Keeps SimResult.responses_per_deque_entry meaningful.
        self.notify_many_calls += 1
        return True

    # -- the loop ----------------------------------------------------------

    def _deliver(self, chunk: Chunk) -> None:  # type: ignore[override]
        fanout = self._table.get(chunk.client_id)
        if fanout is None:
            self.pump_orphans += 1
            return
        fanout(chunk)

    def close_down(self) -> None:
        self._closing = True


class _PostprocPumpHandler(PrebuiltChunkHandler):
    """``PrebuiltChunkHandler`` with the ``await aqueue.get()`` loop removed."""

    async def _generate_locally_impl(self, request: dict, context: Any):
        record = self.records.get(request["id"])
        if record is not None and not record.admitted_ns:
            record.admitted_ns = _perf()

        sampling_params = _prologue(self, request)
        generation_result = _submit(self, request, sampling_params)

        sender = getattr(context, "response_sender", None)
        if sender is None:
            raise RuntimeError(
                "composed-pp-pump requires push egress: no response_sender"
            )

        done = asyncio.get_running_loop().create_future()
        client_id = generation_result.client_id
        fanout = _ChunkFanout(
            sender=sender,
            done=done,
            costs=self.costs,
            handler=self,
            client_id=client_id,
        )
        # No await between generate_async and register(): the pump runs on this
        # same loop, so the first iteration's chunk cannot be dropped.
        self.llm.register(client_id, fanout)
        try:
            await done
        finally:
            self.llm.unregister(client_id)

        # Unreachable, load-bearing: makes this an async generator, which
        # drive_push_egress's `async for` requires. Iterates zero times.
        if False:  # pragma: no cover
            yield {}


# ===========================================================================
# B.  postproc-procs  +  batched-loop
# ===========================================================================


class _ChunkState:
    """Per-request state the batch pass needs. ``batched_loop._RequestState``
    minus everything the postproc process now owns (the cumulative outputs and
    the per-choice cursor moved with ``build_response``)."""

    __slots__ = ("client_id", "record", "done")

    def __init__(
        self, client_id: int, record: Optional[RequestRecord], done: "asyncio.Future"
    ) -> None:
        self.client_id = client_id
        self.record = record
        self.done = done


class _PostprocBatchedLLM(PostprocLLM):
    """Bucket to postproc processes; hand the loop the whole chunk LIST."""

    def __init__(
        self,
        engine_config: Optional[EngineConfig] = None,
        costs: Optional[Costs] = None,
        workers: int = 4,
        threaded: bool = False,
    ) -> None:
        super().__init__(engine_config, costs=costs, workers=workers, threaded=threaded)
        self.sink: Optional[_BatchSink] = None
        self.states: Dict[int, _ChunkState] = {}
        self.loop_batches = 0
        self.loop_batch_items = 0
        self.dropped_late = 0
        self._closing = False

    def register(self, state: _ChunkState) -> None:
        self.states[state.client_id] = state

    def close_down(self) -> None:
        self._closing = True

    # -- receiver thread ---------------------------------------------------

    def _absorb(self, batch: ResultBatch) -> bool:
        self.child_spin_us[batch.worker] = batch.spin_us
        chunks = batch.chunks
        # NEVER return False -- see _PostprocPumpLLM._absorb for the deadlock
        # this caused.
        if not chunks or self._closing:
            return True
        self.chunks_received += len(chunks)
        loop = self._loop
        if loop is None or not loop.is_running():
            return True
        try:
            # ONE ready-deque entry for the whole postproc message -- the same
            # cost and the same count as _SyncQueue.notify_many, carrying the
            # list instead of N Event.set calls.
            loop.call_soon_threadsafe(self._on_chunk_batch, chunks)
            self.notify_many_calls += 1
        except RuntimeError:
            return True
        return True

    # -- ON THE LOOP -------------------------------------------------------

    def _on_chunk_batch(self, chunks: List[Chunk]) -> None:
        sink = self.sink
        if sink is None or self._closing:
            return
        costs = self.costs
        residual_us = costs.scaled(SERVE_LOOP_US_PER_RESPONSE)
        # The per-item half of push_send -- depythonize + try_send -- charged
        # per RESPONSE, never amortised. Hoisted because it is a constant.
        send_item_us = sink.per_item_us
        states = self.states

        self.loop_batches += 1
        self.loop_batch_items += len(chunks)

        outs: List[dict] = []
        records: List[Any] = []
        finished: List[_ChunkState] = []

        for chunk in chunks:
            state = states.get(chunk.client_id)
            if state is None:
                self.dropped_late += 1
                continue

            # result.py:465-501, the PostprocWorker.Output branch.
            with range_("handle_response", color="red"):
                spin(residual_us)

            # The loop is finished with this item here, so this is where the
            # meter ticks -- once per RESPONSE, on the loop thread.
            sent = _perf()
            loop_meter.item()
            pad_to(sent, send_item_us)

            outs.append(chunk.out)
            records.append(state.record)
            if chunk.is_final:
                finished.append(state)

        if outs:
            sink.send_batch(outs, records)

        for state in finished:
            states.pop(state.client_id, None)
            with self._results_lock:
                self._results.pop(state.client_id, None)
            if not state.done.done():
                state.done.set_result(None)

    @property
    def responses_per_loop_batch(self) -> float:
        if not self.loop_batches:
            return 0.0
        return self.loop_batch_items / self.loop_batches


class _PostprocBatchedHandler(PrebuiltChunkHandler):
    """Register a state, park once for the whole request."""

    async def _generate_locally_impl(self, request: dict, context: Any):
        record = self.records.get(request["id"])
        if record is not None and not record.admitted_ns:
            record.admitted_ns = _perf()

        sampling_params = _prologue(self, request)
        generation_result = _submit(self, request, sampling_params)

        sender = getattr(context, "response_sender", None)
        done = asyncio.get_running_loop().create_future()
        state = _ChunkState(generation_result.client_id, record, done)
        if sender is not None:
            # batched_loop._BatchSender.send's compatibility path reads this.
            sender.state = state
        self.llm.register(state)

        await done
        if False:  # pragma: no cover
            yield {}


# ===========================================================================
# C.  the multiplexed sink with per-request termination
# ===========================================================================


class _MuxSink(_BatchSink):
    """``_BatchSink`` whose close frame carries the request it belongs to.

    ``batched_loop``'s driver learns that a request finished from the generator
    it is awaiting. The admission-ring composition drives the generator on the
    LOOP, so the tokio side has nothing to await and needs the identity on the
    frame. One extra tuple element; no extra crossing.
    """

    def close(self, error: Optional[str] = None, tag: Optional[str] = None) -> None:
        self._tokio_loop.call_soon_threadsafe(
            self._queue.put_nowait, (_CLOSE_TAGGED, error, tag)
        )


class _MuxSender(_BatchSender):
    """``_BatchSender`` that tags its termination. Same shipped surface."""

    def __init__(self, sink: _MuxSink, tag: str) -> None:
        super().__init__(sink)
        self.tag = tag

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.close_calls += 1
        self._sink.close(None, self.tag)

    def close_with_error(self, message: str) -> None:
        if self._closed:
            return
        self._closed = True
        self.error_calls += 1
        self._sink.close(message, self.tag)


class _AdmitBatchDriver(BatchedAdmissionDriver):
    """Ring admission (batched_admission) + multiplexed sink (batched_loop).

    ``run`` never touches the GIL cross-thread and never issues a
    ``spawn_blocking``: the whole ingress hand-off is one ``deque.append`` onto
    the ring, which the loop's own admission pump drains. Responses come back on
    the single multiplexed sink, drained by ONE tokio-side demux instead of one
    ``_consume`` task per request.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.sink_queue: "asyncio.Queue" = asyncio.Queue()
        self.sink: Optional[_MuxSink] = None
        self._consumer: Optional["asyncio.Future"] = None
        self._completions: Dict[str, "asyncio.Future"] = {}

    async def run(self, request: dict, record: RequestRecord) -> None:
        if self._consumer is None:
            # Created here because `run` is on the tokio loop, which is where
            # the queue has to be awaited from.
            self._consumer = asyncio.ensure_future(self._consume_mux())

        record.accepted_ns = _perf()
        assert self.sink is not None
        tag = request["id"]
        sender = _MuxSender(self.sink, tag)
        self.senders.append(sender)

        done = asyncio.get_running_loop().create_future()
        self._completions[tag] = done
        try:
            self.ring.offer((request, record, sender))
            await done
        except BaseException:
            self._completions.pop(tag, None)
            raise

    async def _consume_mux(self) -> None:
        queue = self.sink_queue
        on_item = self._on_item
        while True:
            frame = await queue.get()
            if frame[0] == _DATA:
                _, outs, records = frame
                for out, record in zip(outs, records):
                    on_item(out, record)
                continue
            _, error, tag = frame
            if error:
                self.errors.append(error)
            future = self._completions.pop(tag, None)
            if future is not None and not future.done():
                future.set_result(None)

    def stop_mux(self) -> None:
        if self._consumer is not None:
            self._consumer.cancel()
            self._consumer = None
        for future in list(self._completions.values()):
            if not future.done():
                future.cancel()
        self._completions.clear()


# ===========================================================================
# Architectures
# ===========================================================================


class _ComposedPostproc(OffloadedPostproc):
    """Base: everything here keeps ``postproc-procs``' pool and its report."""

    workers = 4
    threaded = False

    def __init__(self) -> None:
        super().__init__()
        self._driver: Optional[Driver] = None

    def on_finished(self, llm: FakeLLM, driver: Driver) -> None:
        close_down = getattr(llm, "close_down", None)
        if close_down is not None:
            close_down()


class ComposedPumpPostproc(_ComposedPostproc):
    """postproc-procs + pump-fanout."""

    name = "composed-pp-pump"
    description = "npw=4 postproc PROCESSES + one MPSC pump, fan-out by client_id"
    egress = "push"

    inbox_high_water: Optional[int] = 50_000
    yield_every = 4096

    def build_llm(self, engine_config: EngineConfig, costs: Costs) -> FakeLLM:
        llm = _PostprocPumpLLM(
            engine_config, costs=costs, workers=self.workers, threaded=self.threaded
        )
        llm.inbox_high_water = self.inbox_high_water
        llm.yield_every = self.yield_every
        self._llm = llm
        return llm

    def build_handler(self, llm, costs, records) -> Any:
        return _PostprocPumpHandler(llm, costs=costs, records=records)

    def on_started(self, llm, driver) -> None:
        llm.start_pump()

    def on_finished(self, llm, driver) -> None:
        llm.stop_pump()
        super().on_finished(llm, driver)

    def extra_report(self) -> Dict[str, Any]:
        report = super().extra_report()
        llm = self._llm
        if llm is not None:
            report.update(llm.pump_report())
        return report


class ComposedAdmitPostproc(_ComposedPostproc):
    """postproc-procs + batched-admission. Egress identical to postproc-procs."""

    name = "composed-pp-admit"
    description = "npw=4 postproc PROCESSES + MPSC admission ring (no spawn_blocking)"
    egress = "push"

    def build_driver(self, handler, py_loop, tokio, costs) -> Driver:
        self._driver = BatchedAdmissionDriver(handler, py_loop, tokio, costs)
        return self._driver

    def on_started(self, llm, driver) -> None:
        driver.ring.start()

    def on_finished(self, llm, driver) -> None:
        driver.ring.stop()
        super().on_finished(llm, driver)

    def extra_report(self) -> Dict[str, Any]:
        report = super().extra_report()
        driver = self._driver
        if driver is not None:
            ring = driver.ring
            report["admitted"] = ring.admitted
            report["doorbells"] = ring.doorbells
            report["doorbells_per_request"] = round(
                ring.doorbells / max(1, ring.admitted), 3
            )
        return report


class ComposedPumpAdmitPostproc(ComposedPumpPostproc):
    """postproc-procs + pump-fanout + batched-admission."""

    name = "composed-pp-pump-admit"
    description = "npw=4 PROCESSES + MPSC pump + MPSC admission ring"

    def build_driver(self, handler, py_loop, tokio, costs) -> Driver:
        self._driver = BatchedAdmissionDriver(handler, py_loop, tokio, costs)
        return self._driver

    def on_started(self, llm, driver) -> None:
        llm.start_pump()
        driver.ring.start()

    def on_finished(self, llm, driver) -> None:
        driver.ring.stop()
        super().on_finished(llm, driver)

    def extra_report(self) -> Dict[str, Any]:
        report = super().extra_report()
        driver = self._driver
        if driver is not None:
            ring = driver.ring
            report["admitted"] = ring.admitted
            report["doorbells_per_request"] = round(
                ring.doorbells / max(1, ring.admitted), 3
            )
        return report


class ComposedBatchLoopPostproc(_ComposedPostproc):
    """postproc-procs + batched-loop."""

    name = "composed-pp-batchloop"
    description = "npw=4 PROCESSES + one loop pass per postproc batch + send_batch"
    egress = "push"

    #: See batched_loop.PUSH_SEND_FIXED_US -- deliberately a lower bound.
    push_send_fixed_us: float = PUSH_SEND_FIXED_US

    def __init__(self) -> None:
        super().__init__()
        self._sink: Optional[_BatchSink] = None

    def build_llm(self, engine_config: EngineConfig, costs: Costs) -> FakeLLM:
        llm = _PostprocBatchedLLM(
            engine_config, costs=costs, workers=self.workers, threaded=self.threaded
        )
        self._llm = llm
        return llm

    def build_handler(self, llm, costs, records) -> Any:
        return _PostprocBatchedHandler(llm, costs=costs, records=records)

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

    def on_finished(self, llm, driver) -> None:
        # Rule: cancel everything this architecture started. The shared tokio
        # demux is one task parked on `sink_queue.get()` forever.
        consumer = getattr(driver, "_consumer", None)
        if consumer is not None:
            consumer.cancel()
            driver._consumer = None
        super().on_finished(llm, driver)

    def extra_report(self) -> Dict[str, Any]:
        report = super().extra_report()
        llm = self._llm
        sink = self._sink
        driver = self._driver
        report["push_send_fixed_us"] = self.push_send_fixed_us
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
            through_loop = len(loop_meter.timestamps())
            report["delivered_end_to_end"] = driver.delivered
            report["drain_ratio"] = round(
                driver.delivered / through_loop if through_loop else 0.0, 3
            )
        return report


class ComposedBatchLoopPostprocStrict(ComposedBatchLoopPostproc):
    """The control: ZERO amortisation of any modelled stage.

    Every response pays the full ``1.94 + 10.72`` of loop work, so whatever this
    wins over ``postproc-procs`` is scheduling that the batch pass really does
    delete, measured in real Python on the real thread.
    """

    name = "composed-pp-batchloop-strict"
    description = "same, NO stage amortisation at all (control)"
    push_send_fixed_us: float = 0.0


class ComposedAll(ComposedBatchLoopPostproc):
    """postproc-procs + batched-loop + batched-admission."""

    name = "composed-all"
    description = "npw=4 PROCESSES + batch loop pass + send_batch + admission ring"

    def build_driver(self, handler, py_loop, tokio, costs) -> Driver:
        driver = _AdmitBatchDriver(handler, py_loop, tokio, costs)
        assert tokio.loop is not None
        sink = _MuxSink(tokio.loop, driver.sink_queue, costs, self.push_send_fixed_us)
        driver.sink = sink
        self._sink = sink
        self._driver = driver
        assert self._llm is not None
        self._llm.sink = sink
        return driver

    def on_started(self, llm, driver) -> None:
        driver.ring.start()

    def on_finished(self, llm, driver) -> None:
        driver.ring.stop()
        driver.stop_mux()
        super().on_finished(llm, driver)

    def extra_report(self) -> Dict[str, Any]:
        report = super().extra_report()
        driver = self._driver
        if driver is not None and hasattr(driver, "ring"):
            ring = driver.ring
            report["admitted"] = ring.admitted
            report["doorbells_per_request"] = round(
                ring.doorbells / max(1, ring.admitted), 3
            )
        return report


class ComposedAllStrict(ComposedAll):
    name = "composed-all-strict"
    description = "composed-all with NO stage amortisation at all (control)"
    push_send_fixed_us: float = 0.0


class ComposedAll8(ComposedAll):
    """8 postproc processes: a faster loop can outrun a 4-process pool.

    4 workers absorb ``4 x 1e6 / 74.62 = 53,600`` items/s. If a composition
    pushes the loop near that, the pool is what is being measured.
    """

    name = "composed-all-8"
    description = "composed-all with npw=8 (pool headroom 107,200 items/s)"
    workers = 8


class ComposedPumpPostproc8(ComposedPumpPostproc):
    name = "composed-pp-pump-8"
    description = "composed-pp-pump with npw=8 (pool headroom 107,200 items/s)"
    workers = 8


for _cls in (
    ComposedPumpPostproc,
    ComposedPumpPostproc8,
    ComposedAdmitPostproc,
    ComposedPumpAdmitPostproc,
    ComposedBatchLoopPostproc,
    ComposedBatchLoopPostprocStrict,
    ComposedAll,
    ComposedAllStrict,
    ComposedAll8,
):
    if _cls.name not in architectures.names():
        architectures.register(_cls)


# ---------------------------------------------------------------------------
# ONE measurement, ONE process, at a FIXED batch
# ---------------------------------------------------------------------------
#
# `bench`'s ladder escalates until it observes a growing backlog, and for every
# postproc architecture that detector is unreliable: the pool's feed-in lanes
# are real pipes, so when the pool falls behind the dispatch thread blocks on
# `put`, stops reading the IPC lane, and the ENGINE stalls. `responses_dispatched
# - loop_meter.count()` then stays flat however overloaded the loop is, the
# ladder reads "not saturated" and escalates. Observed on the same binary,
# minutes apart: `postproc-procs` scored batch 600 / window 5.96 s in one run and
# batch 4,000 / window 0.40 s in the next -- and batch 4,000 means 4,000 requests
# of ingress spread over a tenth as many items, i.e. 87.17 loop us/item against
# 36.98 for the SAME architecture.
#
# So the ladder is a confound between architectures, not a property of them.
# This entry point pins the rung. It runs `bench.run_bench` UNMODIFIED with a
# one-element ladder and prints one JSON object: one architecture, one process,
# exactly as the methodology requires. Repetition and medians are the caller's
# job, so nothing accumulates across runs.
#
#     python3 -m egress_experiments.architectures.composed \
#         --architecture composed-all --batch 600 --json
#
# `--rust-egress-us 0` is the tokio-artefact control that `batched_loop.py`
# documents: `rust_bridge.py` deviation #1 says the simulator's tokio side is
# Python and HOLDS the GIL where real tokio does not, so an architecture whose
# consumer KEEPS UP is charged 11.56 us/item of GIL-holding work on a second
# thread that a lagging architecture never pays. Zeroing it deletes exactly that
# artefact and touches nothing that runs on the loop.


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    import json
    import sys

    from egress_experiments import bench

    parser = argparse.ArgumentParser(
        prog="composed",
        description="ONE bench measurement, ONE process, at a FIXED batch",
    )
    parser.add_argument("--architecture", required=True)
    parser.add_argument(
        "--batch",
        type=int,
        default=600,
        help="single ladder rung, so every architecture gets the same geometry",
    )
    parser.add_argument("--cost-scale", type=float, default=1.0)
    parser.add_argument(
        "--rust-egress-us",
        type=float,
        default=None,
        help="override Costs.rust_egress_us; 0 removes the Python-tokio artefact",
    )
    parser.add_argument("--warmup-s", type=float, default=bench.WARMUP_S)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    costs = Costs()
    if args.rust_egress_us is not None:
        costs = Costs(rust_egress_us=args.rust_egress_us)
    costs = costs.with_scale(args.cost_scale)

    result = bench.run_bench(
        args.architecture, costs, ladder=(args.batch,), warmup_s=args.warmup_s
    )
    payload = dict(result.__dict__)
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(
            f"{result.architecture:<28}{result.items_per_s:>11,.0f}"
            f"  loop {result.work_us_per_item_on_loop:>7.2f}"
            f"  all {result.work_us_per_item_total:>7.2f}"
            f"  batch {result.batch}  window {result.window_s:.2f}s"
            f"  saturated {result.saturated}"
        )
        print(f"  by thread: {result.work_us_by_thread}")
        print(f"  arch: {result.arch_report}")
    return 0


if __name__ == "__main__":
    # `python3 -m` re-imports this file as `__main__` AFTER the package's
    # `_discover()` already imported and registered it, so delegate to THAT
    # module object -- the registry and the classes must be one and the same.
    from egress_experiments.architectures import composed as _module

    raise SystemExit(_module.main())
