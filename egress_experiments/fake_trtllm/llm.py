# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The worker/engine boundary: ``generate_async`` in, dispatch thread out.

This is the mock the goal asks for. The dynamo worker calls
``llm.generate_async(...)``; the engine eats it. Responses come back over the
IPC (zmq) lane and re-enter Python on ``proxy_dispatch_result_thread``, which
is a plain OS thread -- *not* the event loop.

``_Proxy.dispatch_result_task`` is a line-for-line port of
``tensorrt_llm/executor/proxy.py:532``, because its exact shape is the claim
the diagram rests on:

* ``res`` off the IPC lane is a **list** (one engine iteration),
* every element is ``put_nowait``-ed into its own per-request ``AsyncQueue``
  -- a deque append, invisible to the loop,
* and then **one** ``_SyncQueue.notify_many(event_loop, async_queues)`` is
  issued for the entire batch.

So N responses cost N deque appends off-loop and exactly ONE ready-deque entry
on-loop. The counters on :class:`FakeLLM` expose that ratio directly, which is
what ``queue_probe`` measured at ~132:1 on the real worker.
"""

from __future__ import annotations

import asyncio
import collections
import itertools
import threading
import time
from typing import Any, Dict, List, Optional

from egress_experiments.costs import Costs, spin
from egress_experiments.fake_trtllm.aqueue import AsyncQueue, SyncQueue
from egress_experiments.fake_trtllm.engine import EngineConfig, spawn_engine
from egress_experiments.fake_trtllm.result import GenerationResult, Response
from egress_experiments.nvtx_shim import range_

_perf = time.perf_counter_ns


class FakeLLM:
    """Stands in for ``tensorrt_llm.llmapi.LLM`` at the one method dynamo calls.

    Usage mirrors the real worker: construct it, :meth:`start` it from the
    event loop that will consume responses, then call :meth:`generate_async`
    from that same loop.
    """

    def __init__(
        self,
        engine_config: Optional[EngineConfig] = None,
        costs: Optional[Costs] = None,
    ) -> None:
        self.engine_config = engine_config or EngineConfig()
        self.costs = costs or Costs()

        self._client_ids = itertools.count(1)
        self._results: Dict[int, GenerationResult] = {}
        self._results_lock = threading.Lock()
        self._engine = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._dispatch_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

        # Keep a bounded sample of completed results for structural checks.
        # The objects are appended by the dispatch thread before their final
        # response is consumed; their loop-side diagnostics are populated by
        # the time the request generator finishes.
        self.completed_results = collections.deque(maxlen=1024)

        # -- observability the tests assert on ------------------------------
        #: IPC messages received (== engine iterations).
        self.ipc_messages = 0
        #: Individual responses unpacked from those messages.
        self.responses_dispatched = 0
        #: notify_many calls == ready-deque entries the response path costs.
        self.notify_many_calls = 0
        #: Name of the thread that ran the dispatch loop.
        self.dispatch_thread_name: Optional[str] = None
        #: Name of the loop thread, for comparison.
        self.loop_thread_name: Optional[str] = None
        #: submit() calls that reached the engine.
        self.submitted = 0
        #: One event-loop wake for each native batch containing final requests.
        self.native_completion_notify_calls = 0
        #: Arrival timestamp per IPC message, so the observed batch can be
        #: restricted to a steady-state window.
        self.ipc_times: List[int] = []
        #: Responses carried by each IPC message. Measured at the boundary, so
        #: it reports what the engine EMITTED even when the loop is too far
        #: behind to deliver it -- which is exactly the case worth seeing.
        self.ipc_batch_sizes: List[int] = []

    # -- lifecycle ---------------------------------------------------------

    def start(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        self._loop = loop or asyncio.get_event_loop()
        self.loop_thread_name = threading.current_thread().name
        self._engine = spawn_engine(self.engine_config)
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop, name="proxy_dispatch_result_thread", daemon=True
        )
        self._dispatch_thread.start()

    def shutdown(self) -> None:
        self._stop.set()
        if self._engine is not None:
            self._engine.shutdown()
            self._engine = None
        if self._dispatch_thread is not None:
            self._dispatch_thread.join(timeout=5.0)
            self._dispatch_thread = None

    # -- the boundary ------------------------------------------------------

    def generate_async(
        self,
        inputs: Any = None,
        sampling_params: Any = None,
        *,
        streaming: bool = True,
        response_processor: Any = None,
        response_sender: Any = None,
        prompt_tokens: int = 0,
        calibrated_work_us: float = 0.0,
        **kwargs: Any,
    ) -> GenerationResult:
        """Submit and return immediately. Called ON the event loop.

        Accepts and ignores the rest of the real signature
        (``disaggregated_params``, ``trace_headers``, ``scheduling_params``,
        ``priority``, ``cache_salt``) so the real
        ``handler_base._generate_locally_impl`` can call this unmodified.
        """
        if self._engine is None:
            raise RuntimeError("FakeLLM.start() must be called before generate_async()")

        client_id = next(self._client_ids)
        max_tokens = getattr(sampling_params, "max_tokens", None) or (
            self.engine_config.max_tokens
        )
        n = getattr(sampling_params, "n", None) or 1

        result = GenerationResult(
            client_id,
            n=n,
            streaming=streaming,
            costs=self.costs,
            loop=self._loop,
            response_processor=response_processor,
        )
        if response_processor is not None:
            if response_sender is None:
                raise ValueError("native response processing requires response_sender")
            response_processor.register(
                client_id,
                int(prompt_tokens),
                int(n),
                response_sender,
                float(calibrated_work_us),
            )
        # Register BEFORE submitting: the dispatch thread drops responses for
        # unknown client ids (proxy.py:550), so a late registration would lose
        # the first iteration's response.
        with self._results_lock:
            self._results[client_id] = result

        # RpcWorker.submit. Range name matches handler_base's, so a capture of
        # this simulation reads back through capture_params unchanged.
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

    def _dispatch_loop(self) -> None:
        self.dispatch_thread_name = threading.current_thread().name
        while not self._stop.is_set():
            if not self.dispatch_result_task():
                break

    def dispatch_result_task(self) -> bool:
        """Port of ``proxy.py:532``. Returns False to stop the thread."""
        engine = self._engine
        if engine is None:
            return False  # shutdown raced us
        res = engine.result_link.parent.get(timeout=0.25)
        if res is None:
            # timeout or EOF; EOF is signalled by the engine's trailing None,
            # handled in the loop below.
            return not self._stop.is_set()

        # One engine iteration, as the app process observes it. The real
        # capture gets this range from the executor, which shares rank 0's
        # process with the proxy; here the engine is deliberately a separate
        # process, so the marker goes where the iteration lands -- the
        # dispatch thread. capture_params counts iterations by this name and
        # buckets responses that follow it, which holds either way.
        iteration = range_("_handle_responses", color="green")
        iteration.__enter__()
        async_queues: List[SyncQueue] = []
        native_batches: Dict[int, tuple[Any, list[dict[str, Any]]]] = {}
        native_completed: list[GenerationResult] = []
        event_loop: Optional[asyncio.AbstractEventLoop] = None

        def process_res(response: Response) -> None:
            nonlocal event_loop
            with self._results_lock:
                result = self._results.get(response.client_id)
            if result is None:
                # Late response for an already-finalised request (proxy.py:546).
                return
            if result.response_processor is not None:
                payload = response.result
                processor_key = id(result.response_processor)
                _, native_batch = native_batches.setdefault(
                    processor_key, (result.response_processor, [])
                )
                native_batch.append(
                    {
                        "client_id": response.client_id,
                        "new_token_ids": (
                            payload.new_token_ids if payload is not None else []
                        ),
                        "is_final": payload.is_final if payload is not None else True,
                        "finish_reasons": (
                            payload.finish_reasons if payload is not None else None
                        ),
                        "stop_reasons": None,
                        "error_msg": response.error_msg,
                    }
                )
                return
            queue = result.queue
            queue.put_nowait(response)  # deque append -- the loop is untouched
            async_queues.append(queue)
            event_loop = event_loop or queue.loop

            if response.has_error() or (
                response.result is not None and response.result.is_final
            ):
                with self._results_lock:
                    completed = self._results.pop(response.client_id, None)
                if completed is not None:
                    self.completed_results.append(completed)

        batch = res if isinstance(res, list) else [res]
        for item in batch:
            if item is None:
                iteration.__exit__()
                return False  # shutdown
            self.responses_dispatched += 1
            process_res(item)
        # Counted only once the whole message is consumed, so the trailing
        # shutdown sentinel never inflates the responses-per-entry ratio.
        self.ipc_messages += 1
        self.ipc_times.append(_perf())
        self.ipc_batch_sizes.append(len(batch))

        for processor, native_batch in native_batches.values():
            completed_client_ids = processor.process_mock_batch(native_batch)
            for client_id in completed_client_ids:
                with self._results_lock:
                    completed = self._results.pop(client_id, None)
                if completed is not None:
                    self.completed_results.append(completed)
                    native_completed.append(completed)
                    event_loop = event_loop or completed.queue.loop

        if native_completed and event_loop is not None:
            if event_loop.is_running():
                event_loop.call_soon_threadsafe(
                    self._mark_native_done_many, tuple(native_completed)
                )
                self.native_completion_notify_calls += 1
            else:
                iteration.__exit__()
                return False

        if async_queues:
            try:
                SyncQueue.notify_many(event_loop, async_queues)
                self.notify_many_calls += 1
            except AsyncQueue.EventLoopShutdownError:
                iteration.__exit__()
                return False
        iteration.__exit__()
        return True

    @staticmethod
    def _mark_native_done_many(results: tuple[GenerationResult, ...]) -> None:
        for result in results:
            result.mark_native_done()

    # -- reporting ---------------------------------------------------------

    @property
    def responses_per_deque_entry(self) -> float:
        """The ~132:1 the diagram corrects itself with."""
        if not self.notify_many_calls:
            return 0.0
        return self.responses_dispatched / self.notify_many_calls
