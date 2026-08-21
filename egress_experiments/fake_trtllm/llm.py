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


def _build_native_event(
    result: GenerationResult, response: Response
) -> Optional[dict[str, Any]]:
    request_key = result.response_request_key
    if request_key is None:
        raise RuntimeError("native response request was not registered")
    if response.generation != request_key.generation:
        return None

    payload = response.result
    token_ids = payload.new_token_ids if payload is not None else []
    finish_reasons = payload.finish_reasons if payload is not None else None
    event = {
        "client_id": response.client_id,
        "generation": response.generation,
        "sequence": result.response_sequence,
        "outputs": [
            {
                "index": index,
                "new_token_ids": choice_tokens,
                "finish_reason": (
                    finish_reasons[index]
                    if finish_reasons is not None and index < len(finish_reasons)
                    else None
                ),
                "stop_reason": None,
            }
            for index, choice_tokens in enumerate(token_ids)
        ],
        "is_final": payload.is_final if payload is not None else True,
        "error_msg": response.error_msg,
    }
    result.response_sequence += 1
    return event


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
        dispatch_stopped = True
        if self._dispatch_thread is not None:
            self._dispatch_thread.join(timeout=5.0)
            dispatch_stopped = not self._dispatch_thread.is_alive()
            if dispatch_stopped:
                self._dispatch_thread = None

        with self._results_lock:
            pending = list(self._results.values())
            self._results.clear()
        message = "native response processing stopped: worker shut down"
        native_pending = []
        for result in pending:
            if (
                result.response_processor is None
                or result.response_request_key is None
            ):
                continue
            result.response_processor.cancel(result.response_request_key)
            if result.response_sender is not None:
                result.response_sender.close_with_error(message)
            native_pending.append(result)
        if native_pending:
            def fail_pending() -> None:
                self._mark_native_error_many(tuple(native_pending), message)

            if (
                self._loop is not None
                and self._loop.is_running()
                and threading.current_thread().name != self.loop_thread_name
            ):
                self._loop.call_soon_threadsafe(fail_pending)
            else:
                fail_pending()
        if not dispatch_stopped:
            raise RuntimeError("response dispatch thread did not stop within 5 seconds")

    def cancel_native(self, result: GenerationResult) -> None:
        """Remove native request state before late engine responses arrive."""
        result.abort()
        with self._results_lock:
            if self._results.get(result.client_id) is result:
                self._results.pop(result.client_id)
        if (
            result.response_processor is not None
            and result.response_request_key is not None
        ):
            result.response_processor.cancel(result.response_request_key)

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
            result.response_sender = response_sender
            result.response_request_key = response_processor.register(
                client_id,
                int(prompt_tokens),
                int(n),
                response_sender,
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
                    "generation": (
                        result.response_request_key.generation
                        if result.response_request_key is not None
                        else None
                    ),
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
                event = _build_native_event(result, response)
                if event is None:
                    return
                processor_key = id(result.response_processor)
                _, native_batch = native_batches.setdefault(
                    processor_key, (result.response_processor, [])
                )
                native_batch.append(event)
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
            try:
                completed_requests = processor.process_batch(native_batch)
            except Exception as error:
                message = f"native response processing failed: {error}"
                native_failed: list[GenerationResult] = []
                for event in native_batch:
                    client_id = event["client_id"]
                    generation = event["generation"]
                    with self._results_lock:
                        current = self._results.get(client_id)
                        if (
                            current is not None
                            and current.response_request_key is not None
                            and current.response_request_key.generation == generation
                        ):
                            failed = self._results.pop(client_id)
                        else:
                            failed = None
                    if failed is None:
                        continue
                    processor.cancel(failed.response_request_key)
                    if failed.response_sender is not None:
                        failed.response_sender.close_with_error(message)
                    native_failed.append(failed)
                    event_loop = event_loop or failed.queue.loop
                if native_failed and event_loop is not None and event_loop.is_running():
                    event_loop.call_soon_threadsafe(
                        self._mark_native_error_many,
                        tuple(native_failed),
                        message,
                    )
                continue
            for request_key in completed_requests:
                client_id = request_key.client_id
                with self._results_lock:
                    current = self._results.get(client_id)
                    if (
                        current is not None
                        and current.response_request_key is not None
                        and current.response_request_key.generation
                        == request_key.generation
                    ):
                        completed = self._results.pop(client_id)
                    else:
                        completed = None
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

    @staticmethod
    def _mark_native_error_many(
        results: tuple[GenerationResult, ...], message: str
    ) -> None:
        for result in results:
            result.mark_native_error(message)

    # -- reporting ---------------------------------------------------------

    @property
    def responses_per_deque_entry(self) -> float:
        """The ~132:1 the diagram corrects itself with."""
        if not self.notify_many_calls:
            return 0.0
        return self.responses_dispatched / self.notify_many_calls
