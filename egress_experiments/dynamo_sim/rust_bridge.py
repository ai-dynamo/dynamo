# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Rust half of the bridge: ingress, and the two egress directions.

Modelled on ``lib/bindings/python/rust/engine.rs`` and
``lib/bindings/python/rust/push_egress.rs``.

INGRESS -- ``invoke_generator`` (engine.rs:85-114). Every request, on BOTH
paths, crosses into Python inside ``tokio::task::spawn_blocking`` under
``Python::with_gil``: the request object is built and ``generate`` is called,
returning the async generator without running its body. :class:`Driver`.\
``spawn_blocking`` reproduces this on a real
:class:`~concurrent.futures.ThreadPoolExecutor`, so the GIL is genuinely
acquired from a thread that is not the event loop.

PULL -- ``demand_driven_python_stream`` (engine.rs:122-149)::

    let anext = generator.getattr("__anext__")?.unbind();
    ... Python::with_gil(|py| into_future_with_locals(&locals, anext.bind(py).call0()?))

``push_egress.rs:8-16`` enumerates what this costs per RESPONSE -- two
independent GIL acquisitions on tokio threads:

1. ``pybridge.anext_call`` -- a tokio worker takes the GIL only to call
   ``__anext__`` and hand the work to the loop via ``call_soon_threadsafe``,
   then drops the GIL and parks.
2. ``pybridge.decode_response`` -- a ``spawn_blocking`` thread takes the GIL
   again to depythonize the yielded object.

:class:`PullDriver` reproduces both: one ``run_coroutine_threadsafe`` per
response issued from the tokio thread (1), and one :meth:`Driver.spawn_blocking`
per response (2).

PUSH -- ``PythonPushEngine`` (push_egress.rs) with
``DYN_TRTLLM_PUSH_EGRESS=1``. Rust hands the handler a ``ResponseSender`` and
advances the returned async generator ONCE per REQUEST; responses travel out of
band on the sender, converted under the GIL the handler already holds.
:class:`PushDriver` reproduces that: one ``run_coroutine_threadsafe`` per
request, and :class:`ResponseSender` never touches the Python loop.

Known deviations
----------------
1. **The "tokio" side is Python**, so it holds the GIL where real tokio does
   not. That inflates the push path's apparent cost (its Rust-side
   chunk/encode/publish is charged against the same GIL as the loop) and never
   the pull path's, so any push win reported here is a lower bound.
2. **One tokio worker, not eight.** ``push_egress.rs:18-19`` -- "which tokio
   worker polls the stream is arbitrary, so over a run essentially every worker
   thread becomes a GIL contender". Here a single loop thread polls every
   stream, so the *number* of distinct contenders is understated even though
   the per-response acquisition count is exact.
3. **The GIL is far less contended than the real worker's.** The capture holds
   the GIL 7.263 s of a 7.357 s span -- 98.7 % -- while the three instrumented
   stages account for only 32.8 %. The other two thirds is Python this
   simulation does not model at all, which is why ``--gil-noise`` exists.
4. **``invoke_generator`` is charged no intrinsic cost** (``Costs.
   invoke_generator_us`` defaults to 0). The diagram's 1.05 ms for that box is
   *latency* -- overwhelmingly the wait for a GIL the loop holds 98.7 % of the
   time -- not work. Modelling it as work would double-count the queueing the
   simulation is supposed to produce on its own.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from egress_experiments import loop_meter
from egress_experiments.costs import Costs, pad_to, spin
from egress_experiments.dynamo_sim.probes import RequestRecord

_perf = time.perf_counter_ns


class FakeContext:
    """``dynamo._core.Context`` as the handler uses it."""

    def __init__(self, request_id: str) -> None:
        self._id = request_id
        self._stopped = False
        #: Rust delivers the sender both as a kwarg and here (push_egress.rs).
        #: push_egress_capable reads this as its safety net.
        self.response_sender: Any = None

    def id(self) -> str:
        return self._id

    def is_stopped(self) -> bool:
        return self._stopped

    def stop_generating(self) -> None:
        self._stopped = True

    def trace_headers(self) -> Dict[str, str]:
        return {}

    def async_killed_or_stopped(self) -> "asyncio.Future":
        return asyncio.get_running_loop().create_future()


class _Closed:
    __slots__ = ("error",)

    def __init__(self, error: Optional[str] = None) -> None:
        self.error = error


class ResponseSender:
    """Port of ``push_egress.rs::ResponseSender``.

    Exposes exactly ``send`` / ``close`` / ``close_with_error``. All three are
    called from the event loop; none of them enqueue onto it. Delivery is a
    hand-off to the tokio side, standing in for the Rust mpsc channel.
    """

    def __init__(
        self,
        tokio_loop: asyncio.AbstractEventLoop,
        sink: "asyncio.Queue",
        costs: Costs,
    ) -> None:
        self._tokio_loop = tokio_loop
        self._sink = sink
        self._costs = costs
        self._closed = False
        self.sends = 0
        self.close_calls = 0
        self.error_calls = 0
        self.send_threads: List[str] = []

    def _deliver(self, item: Any) -> None:
        self._tokio_loop.call_soon_threadsafe(self._sink.put_nowait, item)

    def send(self, obj: Any) -> None:
        """One call per response. Converts under the GIL we already hold.

        This is where the loop is finished with the item, so this is where the
        benchmark's tick goes -- NOT after the tokio consumer has drained it.
        """
        start = _perf()
        loop_meter.item()
        self.sends += 1
        self.send_threads.append(threading.current_thread().name)
        self._deliver(obj)
        # trtllm:push_send -- the pythonize/enqueue cost, on the loop.
        pad_to(start, self._costs.scaled(self._costs.push_send_us))

    def close(self) -> None:
        if self._closed:
            return  # idempotent, as on the Rust side
        self._closed = True
        self.close_calls += 1
        self._deliver(_Closed())

    def close_with_error(self, message: str) -> None:
        if self._closed:
            return
        self._closed = True
        self.error_calls += 1
        self._deliver(_Closed(message))


async def anext_call(anext: Callable[[], Any]) -> Any:
    """One ``__anext__`` advance, scheduled onto the Python loop.

    Named to match the nsys range ``pybridge.anext_call``. Exists because
    ``run_coroutine_threadsafe`` requires an actual coroutine and
    ``agen.__anext__()`` returns an ``async_generator_asend``.
    """
    item = await anext()
    # The loop is done with this item; StopAsyncIteration skips the tick,
    # which is correct -- it is not an item.
    loop_meter.item()
    return item


async def push_pump(anext: Callable[[], Any], counter: List[int]) -> None:
    """Drive the push-mode generator to exhaustion. Advances ONCE per request.

    ``drive_push_egress_stream`` yields nothing, so the first ``__anext__``
    runs the whole request and then raises ``StopAsyncIteration``. Anything it
    *does* yield would be the Rust driver's fallback arm
    (``pybridge.push_forward_yield``), which is exactly the per-response GIL
    acquisition push exists to remove -- so the count is asserted on in tests.
    """
    while True:
        try:
            await anext()
        except StopAsyncIteration:
            return
        counter[0] += 1


class TokioRuntime:
    """A second event loop plus a blocking pool, standing in for tokio.

    The pool matters. ``engine.rs:85`` wraps the whole Rust->Python crossing in
    ``tokio::task::spawn_blocking``, so ``invoke_generator``'s GIL acquisition
    happens on a *blocking-pool* thread, not on the worker that polls the
    stream -- and on the pull path ``pybridge.decode_response`` takes the GIL on
    one of those threads again, per response. Running either inline would put
    the acquisition on a thread that is already GIL-adjacent and understate the
    cross-thread cost push exists to remove.
    """

    def __init__(
        self, name: str = "tokio-runtime-worker", blocking_threads: int = 8
    ) -> None:
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._name = name
        #: Stand-in for tokio's spawn_blocking pool.
        self.blocking = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, blocking_threads), thread_name_prefix="tokio-blocking"
        )

    def start(self) -> asyncio.AbstractEventLoop:
        def run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.loop = loop
            self._ready.set()
            loop.run_forever()

        self._thread = threading.Thread(target=run, name=self._name, daemon=True)
        self._thread.start()
        self._ready.wait()
        assert self.loop is not None
        return self.loop

    def submit(self, coro) -> "asyncio.Future":
        assert self.loop is not None
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def stop(self) -> None:
        """Cancel outstanding tasks, let them unwind, then stop the loop.

        Stopping the loop outright strands whatever was still in flight -- and
        after an aborted overload run that is thousands of driver tasks, each
        printing "Task was destroyed but it is pending" on GC.
        """
        if self.loop is not None:

            async def drain() -> None:
                pending = [
                    task
                    for task in asyncio.all_tasks()
                    if task is not asyncio.current_task()
                ]
                for task in pending:
                    task.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)

            try:
                asyncio.run_coroutine_threadsafe(drain(), self.loop).result(
                    timeout=10.0
                )
            except Exception:
                pass
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self.blocking.shutdown(wait=False, cancel_futures=True)


class Driver:
    """Common state for the two egress paths."""

    def __init__(
        self,
        handler: Any,
        py_loop: asyncio.AbstractEventLoop,
        tokio: TokioRuntime,
        costs: Costs,
    ) -> None:
        self.handler = handler
        self.py_loop = py_loop
        self.tokio = tokio
        self.costs = costs
        #: Responses that reached the "client" (i.e. left Rust).
        self.delivered = 0
        #: `run_coroutine_threadsafe` calls this driver made onto the Python
        #: loop. Pull: one per response. Push: one per request.
        self.loop_handoffs = 0
        self.errors: List[str] = []
        #: Delivery timestamp per response, so aggregates can be restricted to
        #: a steady-state window -- the nsys capture this models is itself a
        #: 5.169 s window at max batch, not a whole run.
        self.response_times: List[int] = []
        #: GIL acquisitions this driver made from the blocking pool. Pull pays
        #: one per REQUEST (invoke_generator) plus one per RESPONSE
        #: (decode_response); push pays only the former.
        self.blocking_gil_acquisitions = 0

    async def spawn_blocking(self, fn: Callable[..., Any], *args: Any) -> Any:
        """``tokio::task::spawn_blocking`` + ``Python::with_gil``.

        Runs on a pool thread, so the GIL really is acquired cross-thread --
        the sim does not fake this, it pays it.
        """
        self.blocking_gil_acquisitions += 1
        loop = asyncio.get_running_loop()
        if self.costs.invoke_generator_us:
            spin(self.costs.scaled(self.costs.invoke_generator_us))
        return await loop.run_in_executor(self.tokio.blocking, fn, *args)

    def _on_item(self, item: Any, record: RequestRecord) -> None:
        start = _perf()
        self.delivered += 1
        self.response_times.append(start)
        record.responses += 1
        now = _perf()
        if not record.first_response_ns:
            record.first_response_ns = now
        record.last_response_ns = now
        # Rust egress: chunk 6.56 + encode 3.31 + publish 1.69 us. Off the
        # Python loop on both paths; on the pull path it is preceded by the
        # depythonize charged in PullDriver.
        pad_to(start, self.costs.scaled(self.costs.rust_egress_us))


class PullDriver(Driver):
    """``demand_driven_python_stream``: Rust advances the generator per response."""

    mode = "pull"

    async def run(self, request: dict, record: RequestRecord) -> None:
        context = FakeContext(request["id"])
        record.accepted_ns = _perf()

        # engine.rs:85-114 -- invoke_generator: spawn_blocking + with_gil,
        # building the request object and calling `generate`. No body runs yet.
        generator = await self.spawn_blocking(
            functools.partial(self.handler.generate, request, context)
        )
        anext = generator.__anext__

        while True:
            # Python::with_gil { into_future_with_locals(anext.call0()) }
            # -> call_soon_threadsafe -> ONE ready-deque entry, per RESPONSE.
            self.loop_handoffs += 1
            future = asyncio.run_coroutine_threadsafe(anext_call(anext), self.py_loop)
            try:
                item = await asyncio.wrap_future(future)
            except StopAsyncIteration:
                break
            except Exception as exc:  # pragma: no cover - defensive
                self.errors.append(f"{type(exc).__name__}: {exc}")
                break
            # engine.rs:348-352 -- pybridge.decode_response: a spawn_blocking
            # thread takes the GIL AGAIN to depythonize the yielded object.
            # This is the acquisition push removes, and the reason it is run on
            # the pool rather than inline: the cost that matters is contending
            # for the GIL from a thread that does not already hold it.
            await self.spawn_blocking(
                spin, self.costs.scaled(self.costs.pull_bridge_us)
            )
            self._on_item(item, record)


class PushDriver(Driver):
    """``PythonPushEngine``: Rust advances the generator once per request."""

    mode = "push"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        #: Items the push generator yielded anyway. MUST stay 0 -- see
        #: push_forward_yield in the module docstring.
        self.fallback_yields = 0
        self.senders: List[ResponseSender] = []

    async def run(self, request: dict, record: RequestRecord) -> None:
        context = FakeContext(request["id"])
        record.accepted_ns = _perf()

        sink: asyncio.Queue = asyncio.Queue()
        sender = ResponseSender(self.tokio.loop, sink, self.costs)
        self.senders.append(sender)
        # Rust delivers the sender BOTH ways; push_egress_capable prefers the
        # kwarg and falls back to the context.
        context.response_sender = sender

        # push_egress.rs:475 calls the SAME engine::invoke_generator, so push
        # pays this spawn_blocking GIL acquisition too -- once per request.
        # What it does not pay is a second one per response.
        stream = await self.spawn_blocking(
            functools.partial(
                self.handler.generate, request, context, response_sender=sender
            )
        )
        anext = stream.__anext__

        counter = [0]
        # ONE call_soon_threadsafe for the WHOLE request.
        self.loop_handoffs += 1
        pump = asyncio.run_coroutine_threadsafe(push_pump(anext, counter), self.py_loop)
        consumer = asyncio.ensure_future(self._consume(sink, record))

        try:
            try:
                await asyncio.wrap_future(pump)
            except Exception as exc:  # pragma: no cover - defensive
                self.errors.append(f"{type(exc).__name__}: {exc}")
                sender.close()
            self.fallback_yields += counter[0]
            await consumer
        except BaseException:
            # Includes CancelledError, which is how --max-backlog stops an
            # overloaded run. The consumer is parked on `sink.get()` and would
            # otherwise be left pending, producing one "Task was destroyed but
            # it is pending" per in-flight request -- thousands of them, right
            # where the interesting output is.
            for pending in (pump, consumer):
                try:
                    pending.cancel()
                except RuntimeError:
                    # Loop already closed: teardown beat us to it, which is
                    # fine -- the task dies with the loop either way.
                    pass
            raise

    async def _consume(self, sink: asyncio.Queue, record: RequestRecord) -> None:
        """The tokio task draining the sender. No GIL on the real worker."""
        while True:
            item = await sink.get()
            if isinstance(item, _Closed):
                if item.error:
                    self.errors.append(item.error)
                return
            self._on_item(item, record)
