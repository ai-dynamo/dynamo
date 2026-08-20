# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``GenerationResult`` and the response types that cross the IPC boundary.

Ported from ``tensorrt_llm/executor/result.py``. The shape that matters to the
diagram is the async consumption path::

    __anext__          result.py:1104   ->  _aresult_step
      _aresult_step    result.py:1035   ->  await self.aqueue.get()
                                        ->  self._handle_response(response)

``_handle_response`` therefore runs **on the event loop**, inside the coroutine
the proxy dispatch thread woke -- which is why the diagram puts its 23.97 us in
the "ON THE ASYNCIO LOOP" box and not on the dispatch thread. The dispatch
thread's only per-response work is ``put_nowait`` (a deque append).

``Response`` / ``CompletionOutput`` carry only the fields the dynamo worker
reads in ``handler_base._generate_locally_impl``: cumulative ``token_ids`` per
choice, ``index``, ``finish_reason``, ``stop_reason``, plus ``res.finished``
and ``res.outputs``.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

from egress_experiments.costs import Costs, pad_to
from egress_experiments.fake_trtllm.aqueue import AsyncQueue, SyncQueue
from egress_experiments.nvtx_shim import range_

_perf = time.perf_counter_ns


@dataclass
class CompletionOutput:
    """Mirrors ``tensorrt_llm.executor.result.CompletionOutput``.

    ``token_ids`` is CUMULATIVE. The dynamo worker keeps its own per-choice
    cursor and emits only the new slice -- see ``output_tokens_per_choice`` in
    ``handler_base._generate_locally_impl``.
    """

    index: int = 0
    token_ids: List[int] = field(default_factory=list)
    finish_reason: Optional[str] = None
    stop_reason: Optional[str] = None
    request_perf_metrics: Any = None


@dataclass
class ResultPayload:
    """The ``response.result`` half of ``tllm.Response``."""

    #: New tokens for this iteration, one list per choice.
    new_token_ids: List[List[int]]
    is_final: bool = False
    finish_reasons: Optional[List[Optional[str]]] = None


@dataclass
class Response:
    """Mirrors ``tllm.Response``: what one request gets out of one engine
    iteration. A whole iteration's worth of these travels as ONE IPC message.
    """

    client_id: int
    result: Optional[ResultPayload] = None
    error_msg: Optional[str] = None
    #: Engine-side timestamp, used to attribute observed latency between the
    #: engine and the loop queue.
    emitted_ns: int = 0

    def has_error(self) -> bool:
        return self.error_msg is not None


class GenerationResult:
    """What ``generate_async`` hands back to the dynamo worker."""

    def __init__(
        self,
        client_id: int,
        *,
        n: int = 1,
        streaming: bool = True,
        costs: Optional[Costs] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        response_processor: Any = None,
    ) -> None:
        self.client_id = client_id
        self.request_id = client_id
        self._streaming = streaming
        self._costs = costs or Costs()

        self.aqueue = AsyncQueue()
        self.queue: SyncQueue = self.aqueue.sync_q
        if loop is not None:
            self.queue.bind_loop(loop)

        self._outputs: List[CompletionOutput] = [
            CompletionOutput(index=i) for i in range(n)
        ]
        self._done = False
        self._aborted = False
        self._error_msg: Optional[str] = None
        self.response_processor = response_processor
        self._native_done = asyncio.Event() if response_processor is not None else None

        #: Diagnostics for the tests: which thread ran _handle_response, and
        #: how many times. The diagram's claim is that this is the loop thread.
        self.handle_response_threads: List[str] = []

    # -- surface the dynamo worker reads -----------------------------------

    @property
    def outputs(self) -> List[CompletionOutput]:
        return self._outputs

    @property
    def finished(self) -> bool:
        return self._done

    @property
    def disaggregated_params(self):
        return None

    def abort(self) -> None:
        """``GenerationResult.abort`` -- what ``_cancellation_monitor`` calls."""
        self._aborted = True

    @property
    def aborted(self) -> bool:
        return self._aborted

    # -- the response path -------------------------------------------------

    def _handle_response(self, response: Response) -> None:
        """``result.py:454``. Runs ON the event loop, via ``_aresult_step``.

        The range name matches the real ``GenerationResultBase._handle_response``
        decorator, so a capture of this simulation reads back through
        ``capture_params`` exactly as the real capture does.
        """
        with range_("handle_response", color="red"):
            self._handle_response_impl(response)

    def _handle_response_impl(self, response: Response) -> None:
        import threading

        start = _perf()
        self.handle_response_threads.append(threading.current_thread().name)

        if response.has_error():
            self._error_msg = response.error_msg
            self._done = True
            pad_to(start, self._costs.scaled(self._costs.handle_response_us))
            return

        payload = response.result
        assert payload is not None
        for idx, new_tokens in enumerate(payload.new_token_ids):
            if idx >= len(self._outputs):
                continue
            # Cumulative, exactly as TRT-LLM accumulates into CompletionOutput.
            self._outputs[idx].token_ids.extend(new_tokens)
            if payload.finish_reasons and payload.finish_reasons[idx]:
                self._outputs[idx].finish_reason = payload.finish_reasons[idx]
        self._done = payload.is_final

        pad_to(start, self._costs.scaled(self._costs.handle_response_us))

    async def _aresult_step(self) -> None:
        response = await self.aqueue.get()
        self._handle_response(response)

    def __aiter__(self) -> "GenerationResult":
        return self

    async def __anext__(self) -> "GenerationResult":
        if self._done:
            raise StopAsyncIteration
        await self._aresult_step()
        return self

    async def aresult(self) -> "GenerationResult":
        while not self._done:
            await self._aresult_step()
        return self

    async def wait_native(self) -> None:
        if self._native_done is None:
            raise RuntimeError("wait_native requires a native response processor")
        await self._native_done.wait()

    def mark_native_done(self) -> None:
        if self._native_done is None:
            raise RuntimeError("mark_native_done requires a native response processor")
        self._native_done.set()
