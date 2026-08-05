# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One response pump, one MPSC queue, fan-out by ``client_id``.

The thing being removed
-----------------------
Today the response path is **one queue and one parked coroutine per in-flight
request**. With a decode batch of ~1000 that is ~1000 ``AsyncQueue`` objects,
~1000 ``asyncio.Event`` objects, and ~1000 coroutines suspended inside
``GenerationResult.__anext__``. Per IPC batch the loop then pays, per response:

``tensorrt_llm/llmapi/utils.py:494`` (``_SyncQueue._notify_many``)
    ``for queue in queues: queue._aq.notify()`` -- so the *batched* notify is
    batched only in the number of ``run_coroutine_threadsafe`` hops. The body
    still runs **N** ``Event.set()`` calls **on the loop**.

``asyncio.Event.set``
    each ``set()`` walks ``_waiters`` and calls ``fut.set_result(True)``, which
    calls ``Future.__schedule_callbacks`` -> ``loop.call_soon(...)``. One
    ``Handle`` allocation and one ready-deque entry **per response**.

``tensorrt_llm/executor/result.py:1035`` (``_aresult_step``) and ``:1104``
(``__anext__``)
    the loop then pops that Handle, runs ``Task.__step``, and resumes a six-deep
    async-generator chain -- ``push_pump`` -> ``drive_push_egress_stream``
    -> ``drive_push_egress`` -> ``Handler.generate`` -> ``generate_locally``
    -> ``_generate_locally_impl`` -> ``__anext__`` -> ``_aresult_step``
    -> ``AsyncQueue.get`` -- to move one response. On the way back down,
    ``AsyncQueue.get`` empties the deque, calls ``_event.clear()``, and the next
    ``await self._event.wait()`` allocates a **fresh Future** and re-parks.

So a batch of N responses costs N Event sets, N Futures, N Handles, N Task
steps and N six-frame coroutine resumptions -- all on the one thread the whole
system is bottlenecked on. ``put_nowait`` being free is the half of the story
the diagram tells; this is the other half.

The replacement
---------------
:class:`PumpLLM` keeps ``put_nowait`` off-loop and keeps exactly one
``call_soon_threadsafe`` per IPC batch -- both of those were already right --
but changes what is on the other end of it:

* responses go into **one** process-wide ``collections.deque`` (the MPSC
  queue), not into N per-request queues,
* the wakeup resolves **one** ``Future`` belonging to **one** long-lived pump
  coroutine,
* the pump then runs a plain ``while inbox:`` drain loop, dispatching each
  response by ``client_id`` through a dict to a plain callable.

Per IPC batch of N: one wakeup, one Future, one Task step, one drain loop, N
dict lookups, N direct calls. No per-response Event, Future, Handle, Task step
or coroutine resumption. Zero ``AsyncQueue``/``Event`` objects at all.

Work is conserved exactly. :class:`_Fanout` calls the same
``GenerationResult._handle_response`` (``handle_response`` 23.97 us), builds the
same response dict under the same ``trtllm:build_response`` range (50.65 us),
and hands it to the same ``ResponseSender.send`` (``trtllm:push_send``
10.72 us, and the one :func:`loop_meter.item` tick, on the loop). Nothing moved
to another thread, nothing amortised, nothing deleted -- only the scheduling
around it is gone.

What still runs unmodified
--------------------------
The real shipped ``push_egress.py`` is still in the path. ``generate`` keeps the
real ``push_egress_capable`` decorator, Rust still gets an async generator, and
``drive_push_egress``'s ``async for response in stream:`` still terminates the
request with ``close()``. It simply iterates zero times, because the responses
now leave through the pump instead of being yielded back up the chain.

What this is worth -- measured, and it is not much
--------------------------------------------------
The machinery above is real and it scales, but at the calibrated cost model it
is small. Isolating it (one loop, no engine, no IPC, trivial per-response body,
the full nine-frame resume chain) gives the per-response SCHEDULING cost::

    in flight / per batch      today    this arch    saved
    240 / 240                  1.52 us   0.15 us     1.38 us
    600 / 600                  1.89      0.15        1.74
    1500 / 1500                2.51      0.18        2.33
    4000 / 4000                3.79      0.20        3.59
    10000 / 10000              9.06      0.23        8.83

    7888 / 197  (capture)      2.88      0.13        2.75

and gen0/gen1/gen2 collections over the same 200k responses drop from
1405/127/11 to 397/36/3.

Against the capture's **85.34 us** of measured per-response loop work, 2.75 us
is **3.2 %**. That is the honest ceiling at the real geometry. The
loop-throughput benchmark runs si=1 at batch 240, where the ceiling is 1.38 us
of 87.8 -- 1.6 %, i.e. inside its own +-5 % noise.

In the full simulation, equal wall time, backlog abort off, 3 reps, median::

    batch   baseline items/s   pump items/s   ratio    marginal us/item
      240             8,987          9,165    1.020x   87.90 -> 87.18
      600             8,885          9,205    1.036x   88.02 -> 87.19
     1500             8,061          8,849    1.098x   88.03 -> 87.36

The ratio grows with the batch exactly as the isolated numbers predict, and the
marginal per-item service time -- the part that is not supply or GC -- is a
flat ~0.8 % better at every size.

Under ``bench``'s own stop condition it comes out at **0.95-0.97x**, and the
reason is worth knowing rather than hiding. ``bench`` scores loop-exit items;
``--max-backlog`` is ``responses_dispatched - driver.delivered``, and
``driver.delivered`` is the TOKIO-side consumer. Baseline runs much further
ahead of that consumer (it delivers 80 % of its loop items to the client, this
architecture delivers 100 %), and the simulator's tokio stand-in is Python and
**holds the GIL** where real tokio does not -- a deviation ``rust_bridge.py``
documents. So an architecture whose consumer keeps up is charged 11.2 us/item
of GIL-holding work on a second thread where baseline is charged 4.1, and it
pays for that on the loop. Same runs, same reps: this architecture's *system*
throughput is 8,796/s against 7,678/s, **1.15x**.

The lever is not here. It is ``trtllm:build_response`` (50.65) plus
``handle_response`` (23.97), which are 87 % of the per-response loop budget and
which this change deliberately does not touch.

Does ``async for`` over ``GenerationResult`` survive?
-----------------------------------------------------
**As a public API: yes, unchanged.** As dynamo's hot path: no, and that is the
whole point -- ``handler_base.py:1158``'s ``async for res in generation_result``
*is* the parked coroutine being removed.

The compatible shape is to make the queue **lazy and opt-in**:

* ``GenerationResult.__init__`` stops constructing ``AsyncQueue`` eagerly
  (``result.py:949``);
* ``__aiter__`` / ``aresult()`` / ``__await__`` / ``result()`` / ``__next__``
  allocate it on first use and register the callback ``lambda r:
  self.aqueue.put(r)`` with the pump. Behaviour and cost are then **exactly**
  today's -- that is the ``pump-wakeup`` ablation, measured;
* ``set_response_callback(cb)`` is the new fast path and is mutually exclusive
  with iteration.

Three semantics change for a caller that opts into the callback, and they are
why the queue has to stay for everyone else:

1. **No per-request backpressure.** Today a slow consumer lets its own deque
   grow. A callback runs inside the pump, so it cannot defer; a consumer that
   wants to ``await`` between responses must keep the queue.
2. **Sync consumers.** ``result()`` / ``__next__`` read ``self.queue`` from a
   non-loop thread (``result.py:1017``). A single loop-side pump cannot serve
   them; they keep the queue.
3. **Head-of-line blocking.** One slow callback now stalls the whole pump, not
   just its own request. Acceptable for dynamo (the decode callback never
   awaits) and unacceptable in general.

Error and cancellation paths are straightforward: ``EngineDeadError``, which
today is enqueued onto every pending result, instead invokes every registered
callback (or completes every ``done`` future exceptionally), and ``abort()`` /
``_cancellation_monitor`` are untouched because they never went through the
queue.

Real-code delta
---------------
See the module-level ``REAL_CODE_CHANGES`` string.
"""

from __future__ import annotations

import asyncio
import collections
import time
from typing import Any, Dict, Optional

from egress_experiments import architectures
from egress_experiments.costs import Costs, pad_to, spin
from egress_experiments.dynamo_sim.rust_bridge import Driver, PushDriver
from egress_experiments.dynamo_sim.worker import SamplingParams, TrtllmWorkerHandler
from egress_experiments.fake_trtllm.engine import EngineConfig
from egress_experiments.fake_trtllm.llm import FakeLLM
from egress_experiments.fake_trtllm.result import GenerationResult, Response
from egress_experiments.nvtx_shim import range_

_perf = time.perf_counter_ns


REAL_CODE_CHANGES = """
tensorrt_llm/executor/proxy.py:532  dispatch_result_task
    The per-response `queue.put_nowait(res)` into N `_SyncQueue`s, the
    `async_queues` list and the `_SyncQueue.notify_many(event_loop,
    async_queues)` call all go. In their place: `self._response_inbox.append(
    res)` -- ONE append of the list the lane already handed us -- followed by a
    single `loop.call_soon_threadsafe(self._pump.wake)`. `frozenset(queues)`
    (utils.py:503), which is O(N) hashing on the dispatch thread per batch,
    goes with it. The `self._results.pop(client_id)` on a final response moves
    into the pump, which is where finality is now observed.

    NEW, and load-bearing: the dispatch thread must not read the lane when the
    inbox is over a high-water mark. Today the unbounded per-request deques are
    protected only by this thread being slow; make it fast and loop
    backpressure silently becomes host memory. See PumpLLM.inbox_high_water.

tensorrt_llm/executor/proxy.py  (new) _ResponsePump
    One `asyncio.Task` created in `_start_dispatch_threads`. Parks on a Future,
    drains the inbox one IPC batch at a time, and per response does
    `cb = self._callbacks.get(res.client_id)` then `cb(res)`.

tensorrt_llm/executor/result.py:949  GenerationResult.__init__
    `self.aqueue = AsyncQueue()` becomes lazy. A result that nobody iterates
    never allocates a queue or an asyncio.Event -- which, for dynamo, is every
    result.

tensorrt_llm/executor/result.py  (new) GenerationResult.set_response_callback
    Registers a plain callable with the proxy's pump. Mutually exclusive with
    `__aiter__`/`aresult()`/`result()`, which lazily allocate the queue and
    register the "put it on the queue" callback instead -- so the old API keeps
    working, at the old cost, unchanged.

tensorrt_llm/executor/result.py:1035/1104  _aresult_step / __anext__
    UNCHANGED. They just stop being on dynamo's hot path.

tensorrt_llm/llmapi/utils.py:388/475  AsyncQueue / _SyncQueue
    unchanged, but no longer instantiated per request on the dynamo path.
    `_SyncQueue.notify_many` loses its only caller.

components/src/dynamo/trtllm/request_handlers/handler_base.py:1158
    `async for res in generation_result:` -> `generation_result
    .set_response_callback(cb)` plus `await done`, where `cb` is the body of
    the loop (the `for output in res.outputs:` block, verbatim) with `yield
    out` replaced by `response_sender.send(out)`.

components/src/dynamo/trtllm/request_handlers/push_egress.py:186
    `drive_push_egress`'s `async for response in stream:` keeps its job
    (termination, close/close_with_error, cancellation) but iterates zero times
    for the decode handler. It is still needed verbatim for the pull path and
    for handlers that still yield.

    NOTE: this is push-only. On the pull path there is nowhere to put `out`
    except back into a per-request queue for the generator to yield from, which
    is the thing being removed.
"""


# ---------------------------------------------------------------------------
# The per-request fan-out target: a plain callable, not a coroutine
# ---------------------------------------------------------------------------


class _Fanout:
    """What one request registers with the pump, instead of parking a coroutine.

    Called directly from the pump's drain loop, on the event loop thread, once
    per response. Everything it does is what ``_generate_locally_impl``'s
    ``async for`` body did -- the only difference is that reaching it costs a
    dict lookup rather than an ``Event.set`` + ``Future`` + ``Handle`` + a
    six-frame coroutine resumption.
    """

    __slots__ = (
        "result",
        "sender",
        "done",
        "costs",
        "handler",
        "num_input_tokens",
        "cursor",
        "finished",
        "client_id",
    )

    def __init__(
        self,
        *,
        result: GenerationResult,
        sender: Any,
        done: "asyncio.Future",
        costs: Costs,
        handler: Any,
        num_input_tokens: int,
    ) -> None:
        self.result = result
        self.sender = sender
        self.done = done
        self.costs = costs
        self.handler = handler
        self.num_input_tokens = num_input_tokens
        #: ``output_tokens_per_choice`` from handler_base.py:1018 -- the
        #: per-choice cursor into the CUMULATIVE token_ids TRT-LLM streams.
        self.cursor: Dict[int, int] = {}
        self.finished = False
        self.client_id = result.client_id

    def __call__(self, response: Response) -> None:
        result = self.result
        # result.py:454 _handle_response -- IDENTICAL work, identical cost, and
        # still on the loop. It just is not reached through a coroutine now.
        result._handle_response(response)

        costs = self.costs
        cursor = self.cursor
        sender = self.sender
        handler = self.handler

        for output in result.outputs:
            # handler_base.py:1187 trtllm:build_response -- verbatim from
            # dynamo_sim/worker.py so the only delta versus baseline-push is
            # scheduling.
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

                if out.get("finish_reason") or result.finished:
                    if not out.get("finish_reason"):
                        out["finish_reason"] = "unknown"
                    total_completion_tokens = sum(
                        len(o.token_ids) for o in result.outputs
                    )
                    out["completion_usage"] = {
                        "prompt_tokens": int(self.num_input_tokens),
                        "completion_tokens": int(total_completion_tokens),
                        "total_tokens": int(
                            self.num_input_tokens + total_completion_tokens
                        ),
                        "prompt_tokens_details": None,
                    }

                pad_to(start, costs.scaled(costs.build_response_us))

            handler.responses_yielded += 1
            # push_egress.py drive_push_egress: `response_sender.send(response)`.
            # Still on the loop, still under the GIL we already hold, still
            # charged trtllm:push_send -- and still exactly one loop_meter tick.
            sender.send(out)
            cursor[output_idx] = next_total_toks

        if result.finished and not self.finished:
            self.close()

    def close(self) -> None:
        """End of stream: release the one coroutine this request ever parked."""
        self.finished = True
        done = self.done
        if not done.done():
            done.set_result(None)


# ---------------------------------------------------------------------------
# The proxy side: one MPSC queue, one pump coroutine
# ---------------------------------------------------------------------------


class PumpLLM(FakeLLM):
    """``FakeLLM`` with ``dispatch_result_task`` rewired onto a single queue.

    Only the egress half changes. ``generate_async``, the engine, the IPC lane
    and every counter the harness reads are the base class's.
    """

    #: Safety cap on how many responses the drain runs before handing the loop
    #: back, INSIDE one IPC batch. The natural yield point is the batch
    #: boundary -- which is exactly baseline's granularity, because
    #: ``_notify_many`` schedules the whole batch into ``_ready`` and
    #: ``BaseEventLoop._run_once`` then runs all of them back to back without a
    #: ``select()``. This only bites for batches larger than itself.
    #:
    #: It cannot be removed: with no yield at all the loop never re-enters
    #: ``_run_once``, and the harness's own ``call_soon_threadsafe(done.set)``
    #: never runs -- a 12 s run took 432 s before this was bounded.
    yield_every = 4096

    #: Responses allowed to sit in the MPSC queue before the dispatch thread
    #: stops reading the IPC lane.
    #:
    #: This is NOT tuning, it is the thing that makes the redesign safe. The
    #: dispatch thread got dramatically cheaper -- one ``deque.append`` per
    #: MESSAGE instead of N ``put_nowait`` plus a ``frozenset(queues)`` build
    #: per batch -- so when the loop falls behind it now drains the lane far
    #: faster than the loop consumes, the engine stops blocking on its
    #: ``zmq``/socket ``send``, and loop backpressure silently becomes host
    #: memory. Measured at batch 1500: the unbounded pump pulled 66,006
    #: responses/s off the lane against a loop doing ~7,900/s, and the
    #: resulting live-object count cost more in gen2 GC than the redesign
    #: saved -- 6,452 items/s against baseline's 9,436.
    #:
    #: Today's code has the same unbounded per-request deques and is protected
    #: only by its dispatch thread being slow. Making the bound explicit is
    #: what you would ship. Set to ``None`` to measure without it.
    inbox_high_water: Optional[int] = 50_000

    def __init__(
        self,
        engine_config: Optional[EngineConfig] = None,
        costs: Optional[Costs] = None,
    ) -> None:
        super().__init__(engine_config, costs)
        #: THE MPSC queue. One deque for the whole process, holding IPC
        #: batches -- i.e. exactly the ``list`` that came off the lane, which
        #: ``handle_for_ipc_batched`` (base_worker.py:1252) already built. One
        #: ``append`` per MESSAGE from the dispatch thread, atomic under the
        #: GIL, replacing today's N ``put_nowait`` calls into N deques.
        self._inbox: collections.deque = collections.deque()
        #: client_id -> _Fanout. Loop-thread only, so no lock.
        self._table: Dict[int, Any] = {}
        self._waiter: Optional["asyncio.Future"] = None
        self._pump_task: Optional["asyncio.Task"] = None
        self._pump_stop = False

        # -- pump counters, reported by the architecture -------------------
        #: ``call_soon_threadsafe`` wakeups issued by the dispatch thread.
        self.pump_wakeups = 0
        #: Times the pump actually had to park on a Future.
        self.pump_parks = 0
        #: Times the drain loop ran.
        self.pump_drains = 0
        #: Responses dispatched by the pump.
        self.pump_items = 0
        #: Voluntary ``await asyncio.sleep(0)`` yields inside the drain.
        self.pump_yields = 0
        #: Responses arriving for a client_id no longer in the table
        #: (proxy.py:546's "drop late responses").
        self.pump_orphans = 0
        #: 2 ms sleeps the dispatch thread took at the high-water mark.
        self.dispatch_stalls = 0

    # -- registration ------------------------------------------------------

    def register(self, client_id: int, fanout: Any) -> None:
        self._table[client_id] = fanout

    def unregister(self, client_id: int) -> None:
        self._table.pop(client_id, None)
        with self._results_lock:
            self._results.pop(client_id, None)

    # -- proxy_dispatch_result_thread --------------------------------------

    def dispatch_result_task(self) -> bool:
        """``proxy.py:532`` with the per-request queues taken out.

        Per response: one ``deque.append`` -- same as ``put_nowait`` today.
        Per batch: one ``call_soon_threadsafe`` -- same count as ``notify_many``
        today, minus the ``frozenset(queues)`` build and the ``_notify_many``
        coroutine.
        """
        engine = self._engine
        if engine is None:
            return False

        # Backpressure, BEFORE reading the lane. Not reading is what makes the
        # engine block on its own send, which is exactly the mechanism that
        # bounds today's design -- it is just implicit there. `Event.wait`
        # releases the GIL, so a stalled dispatch thread costs the loop
        # nothing.
        high_water = self.inbox_high_water
        if high_water is not None:
            while (self.responses_dispatched - self.pump_items) > high_water:
                if self._stop.wait(0.002):
                    return False
                self.dispatch_stalls += 1

        res = engine.result_link.parent.get(timeout=0.25)
        if res is None:
            return not self._stop.is_set()

        iteration = range_("_handle_responses", color="green")
        iteration.__enter__()

        batch = res if isinstance(res, list) else [res]
        if batch and batch[-1] is None:
            # Shutdown sentinel: the engine's trailing None (engine.py:367).
            iteration.__exit__()
            return False
        count = len(batch)
        # ONE append for the whole message. The list is already the unit the
        # engine shipped, so nothing is copied and the deque stays short even
        # when the loop is tens of thousands of responses behind.
        if count:
            self._inbox.append(batch)
        self.responses_dispatched += count
        self.ipc_messages += 1
        self.ipc_times.append(_perf())
        self.ipc_batch_sizes.append(len(batch))

        if count:
            loop = self._loop
            if loop is None or not loop.is_running():
                iteration.__exit__()
                return False
            try:
                loop.call_soon_threadsafe(self._wake)
            except RuntimeError:
                iteration.__exit__()
                return False
            self.pump_wakeups += 1
            # Keeps SimResult.responses_per_deque_entry meaningful: this is the
            # one ready-deque entry the whole batch costs.
            self.notify_many_calls += 1

        iteration.__exit__()
        return True

    # -- the loop side -----------------------------------------------------

    def _wake(self) -> None:
        """Runs ON the loop. One Future resolution for the whole batch."""
        waiter = self._waiter
        if waiter is not None and not waiter.done():
            waiter.set_result(None)

    def _deliver(self, response: Response) -> None:
        """Dispatch one response. Overridden by the wakeup ablation."""
        fanout = self._table.get(response.client_id)
        if fanout is None:
            self.pump_orphans += 1
            return
        fanout(response)

    async def _pump(self) -> None:
        """THE response pump. One coroutine for the whole process.

        One drain per IPC batch, one dict lookup and one direct call per
        response. The only suspension is at a batch boundary, which is where
        ``BaseEventLoop._run_once`` would have returned to ``select()`` on the
        baseline path too -- so the two architectures hand the GIL back at the
        same points and the comparison is not an artefact of scheduling
        cadence.
        """
        loop = self._loop
        assert loop is not None
        inbox = self._inbox
        yield_every = self.yield_every
        deliver = self._deliver

        while not self._pump_stop:
            if inbox:
                self.pump_drains += 1
                batch = inbox.popleft()
                n = 0
                for response in batch:
                    deliver(response)
                    n += 1
                    if n >= yield_every:
                        # Counted here, not after the loop: under saturation
                        # the drain can run for a long time and a post-loop
                        # update would never land.
                        self.pump_items += n
                        n = 0
                        self.pump_yields += 1
                        await asyncio.sleep(0)
                        if self._pump_stop:
                            return
                self.pump_items += n
                self.pump_yields += 1
                await asyncio.sleep(0)
                continue

            waiter = loop.create_future()
            self._waiter = waiter
            self.pump_parks += 1
            try:
                await waiter
            except asyncio.CancelledError:
                return
            finally:
                self._waiter = None

    def start_pump(self) -> None:
        if self._pump_task is None:
            self._pump_task = asyncio.get_running_loop().create_task(self._pump())

    def stop_pump(self) -> None:
        self._pump_stop = True
        waiter = self._waiter
        if waiter is not None and not waiter.done():
            waiter.set_result(None)
        if self._pump_task is not None:
            self._pump_task.cancel()
            self._pump_task = None

    # -- reporting ---------------------------------------------------------

    def pump_report(self) -> Dict[str, Any]:
        wakeups = max(1, self.pump_wakeups)
        return {
            "wakeups_per_ipc_batch": round(
                self.pump_wakeups / max(1, self.ipc_messages), 3
            ),
            "pump_parks": self.pump_parks,
            "pump_drains": self.pump_drains,
            "pump_items": self.pump_items,
            "items_per_wakeup": round(self.pump_items / wakeups, 1),
            "loop_yields_per_item": round(
                self.pump_yields / max(1, self.pump_items), 4
            ),
            "dispatch_stalls": self.dispatch_stalls,
            "high_water": self.inbox_high_water,
            "orphans": self.pump_orphans,
            "async_queues_alive": 0,
        }


class WakeupPumpLLM(PumpLLM):
    """Ablation: same MPSC transport, but keep the per-request wakeups.

    The pump drains the single queue exactly as :class:`PumpLLM` does and then,
    per response, does ``result.aqueue.put(response)`` -- a deque append plus
    ``Event.set()``, which schedules that request's parked coroutine. Every
    ``AsyncQueue``, every ``Event``, every per-response ``Future``/``Handle``
    and the whole six-frame resumption chain stay exactly as they are today.

    So ``baseline-push`` -> ``pump-wakeup`` isolates the *transport* change
    (one deque instead of N, no ``frozenset``, no ``_notify_many`` coroutine),
    and ``pump-wakeup`` -> ``pump-fanout`` isolates the *wakeup* change
    (N ``Event.set`` + N coroutine resumptions vs one drain loop).
    """

    def _deliver(self, response: Response) -> None:
        result = self._results.get(response.client_id)
        if result is None:
            self.pump_orphans += 1
            return
        # llmapi/utils.py:415 AsyncQueue.put -- append + Event.set. The set
        # walks _waiters and schedules the per-request coroutine.
        result.aqueue.put(response)
        if response.has_error() or (
            response.result is not None and response.result.is_final
        ):
            self._results.pop(response.client_id, None)

    def pump_report(self) -> Dict[str, Any]:
        report = super().pump_report()
        report["async_queues_alive"] = len(self._results)
        return report


# ---------------------------------------------------------------------------
# The worker side: no per-request consumer coroutine
# ---------------------------------------------------------------------------


class PumpHandler(TrtllmWorkerHandler):
    """``_generate_locally_impl`` with the ``async for`` replaced by a
    registration plus a single ``await``.

    Everything above it is untouched: ``generate`` still carries the real
    ``push_egress_capable``, so Rust still receives an async generator from
    ``drive_push_egress_stream``, ``drive_push_egress`` still owns termination,
    and ``generate_locally``'s NVTX range still spans the request.
    """

    def __init__(self, llm: PumpLLM, costs=None, records=None) -> None:
        super().__init__(llm, costs=costs, records=records)
        self.pump_llm = llm

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

        # push_egress.rs delivers the sender both as a kwarg and on the context;
        # push_egress_capable consumes the kwarg, so read the context copy.
        sender = getattr(context, "response_sender", None)
        if sender is None:
            raise RuntimeError(
                "pump-fanout requires push egress: no response_sender on the context"
            )

        done = asyncio.get_running_loop().create_future()
        fanout = _Fanout(
            result=generation_result,
            sender=sender,
            done=done,
            costs=self.costs,
            handler=self,
            num_input_tokens=len(request.get("token_ids") or []),
        )
        client_id = generation_result.client_id
        self.pump_llm.register(client_id, fanout)
        try:
            # THE only suspension this request performs. Resolved once, by the
            # pump, on the final response -- not once per token.
            await done
        finally:
            self.pump_llm.unregister(client_id)

        # Unreachable, and load-bearing: it is what makes this an async
        # generator, which drive_push_egress's `async for` requires. It
        # iterates zero times -- responses left through the pump.
        if False:  # pragma: no cover
            yield {}


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------


class PumpFanout(architectures.Architecture):
    name = "pump-fanout"
    description = "one pump coroutine + one MPSC queue, fan-out by client_id"
    egress = "push"

    llm_class = PumpLLM
    handler_class = PumpHandler
    #: Safety cap inside one IPC batch; the natural yield is the batch boundary.
    yield_every = 4096
    #: See :attr:`PumpLLM.inbox_high_water`.
    inbox_high_water: Optional[int] = 50_000

    def __init__(self) -> None:
        self._llm: Optional[PumpLLM] = None

    def build_llm(self, engine_config: EngineConfig, costs: Costs) -> FakeLLM:
        llm = self.llm_class(engine_config, costs=costs)
        llm.yield_every = self.yield_every
        llm.inbox_high_water = self.inbox_high_water
        self._llm = llm
        return llm

    def build_handler(self, llm, costs, records):
        return self.handler_class(llm, costs=costs, records=records)

    def build_driver(self, handler, py_loop, tokio, costs) -> Driver:
        # Unchanged: Rust still advances the generator once per REQUEST and the
        # sender is still the only response channel.
        return PushDriver(handler, py_loop, tokio, costs)

    def on_started(self, llm, driver) -> None:
        llm.start_pump()

    def on_finished(self, llm, driver) -> None:
        llm.stop_pump()

    def extra_report(self) -> Dict[str, Any]:
        return self._llm.pump_report() if self._llm is not None else {}


class PumpWakeup(PumpFanout):
    name = "pump-wakeup"
    description = "ABLATION: MPSC pump, but still N Event.set + N coroutines"
    llm_class = WakeupPumpLLM
    handler_class = TrtllmWorkerHandler  # the baseline per-request consumer


architectures.register(PumpFanout)
architectures.register(PumpWakeup)
