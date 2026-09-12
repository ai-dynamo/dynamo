# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cancellation behaviour of the disaggregated SGLang prefill handler.

A prefill worker's first output only arrives after prefill and the KV handoff,
so anything that learns the engine request ID from that output cannot abort
during prefill at all -- which is the whole window a disconnecting client cares
about.
"""

import asyncio
from types import SimpleNamespace

import pytest

from dynamo.sglang.request_handlers.llm.prefill_handler import PrefillWorkerHandler

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
    pytest.mark.timeout(30),
]


class _CancelableContext:
    """Minimal Context stand-in whose cancellation is level-triggered.

    The real binding is sticky: a request cancelled before the monitor starts
    still fires. Modelling that matters, otherwise a test can pass against an
    implementation that would miss an early disconnect in production.
    """

    def __init__(self, request_id: str, *, trace_id: str | None = None):
        self._request_id = request_id
        self.trace_id = trace_id
        self._cancelled = asyncio.Event()

    def id(self) -> str:
        return self._request_id

    def trace_headers(self) -> dict[str, str]:
        return {}

    def async_killed_or_stopped(self) -> asyncio.Task[bool]:
        return asyncio.create_task(self._cancelled.wait())

    def is_stopped(self) -> bool:
        return self._cancelled.is_set()

    def is_killed(self) -> bool:
        return False

    def cancel(self) -> None:
        self._cancelled.set()


def _handler(engine) -> PrefillWorkerHandler:
    handler = PrefillWorkerHandler.__new__(PrefillWorkerHandler)
    handler.engine = engine
    handler.shutdown_event = None
    return handler


@pytest.mark.asyncio
async def test_prefill_submits_under_a_known_request_id():
    captured: dict = {}

    async def empty_results():
        if False:  # pragma: no cover - an async generator that yields nothing
            yield {}

    class _Engine:
        async def async_generate(self, **kwargs):
            captured.update(kwargs)
            return empty_results()

    handler = _handler(_Engine())
    handler.bootstrap_host = "127.0.0.1"
    handler.bootstrap_port = 1234
    handler.enable_trace = False
    handler._consume_tasks = set()
    handler._generate_bootstrap_room = lambda: 17
    handler._get_input_param = lambda request: {"input_ids": request["token_ids"]}
    handler._resolve_lora = lambda request: None
    handler._priority_kwargs = lambda priority: {}

    stream = handler.generate(
        {"request": {"token_ids": [1, 2, 3], "routing": {}}, "sampling_params": {}},
        _CancelableContext("request-id"),
    )
    await anext(stream)
    await stream.aclose()

    # Without this the handler would submit with rid=None and have nothing to
    # abort until the engine echoed an ID back after the handoff.
    assert captured["rid"] == "request-id"


@pytest.mark.asyncio
async def test_prefill_aborts_during_prefill_before_any_output():
    rid = "request-id"
    aborts: list[tuple[str, bool]] = []
    consuming = asyncio.Event()
    aborted = asyncio.Event()

    class _TokenizerManager:
        def abort_request(self, *, rid, abort_all):
            aborts.append((rid, abort_all))
            aborted.set()

    async def results():
        # Models a long prefill: the generator is live but produces nothing
        # until the abort lands.
        consuming.set()
        await aborted.wait()
        return
        yield {}  # pragma: no cover - unreachable, marks this an async generator

    handler = _handler(SimpleNamespace(tokenizer_manager=_TokenizerManager()))
    context = _CancelableContext(rid)

    consumer = asyncio.create_task(handler._consume_results(results(), rid, context))
    await asyncio.wait_for(consuming.wait(), timeout=1)

    context.cancel()
    await asyncio.wait_for(consumer, timeout=2)

    assert aborts == [(rid, False)]


@pytest.mark.asyncio
async def test_prefill_not_submitted_when_the_client_has_already_gone():
    """The abort must not be armed before the engine has registered the rid.

    SGLang registers a request on the first iteration of its generator, not
    when the generator is built, and drops an abort naming an rid it has not
    registered. Arming the monitor at submission therefore let an already
    disconnected client produce an abort that went nowhere, followed by a
    prefill that ran to completion regardless.
    """
    started = False
    aborts: list[str] = []

    class _TokenizerManager:
        def abort_request(self, *, rid, abort_all):
            aborts.append(rid)

    async def results():
        nonlocal started
        started = True
        yield {}

    handler = _handler(SimpleNamespace(tokenizer_manager=_TokenizerManager()))
    context = _CancelableContext("request-id")
    context.cancel()

    await asyncio.wait_for(
        handler._consume_results(results(), "request-id", context), timeout=2
    )

    assert not started, "the engine stream was started for a client already gone"
    assert not aborts, "aborted an rid the engine had never registered"


@pytest.mark.asyncio
async def test_parallel_sampling_aborts_every_sample_id():
    """Parallel sampling replaces the submitted ID with one per sample.

    ``GenerateReqInput._normalize_rid`` turns a string ``rid`` into
    ``rid_0``, ``rid_1``, ..., and ``abort_request`` drops an ID it has not
    registered, so aborting the ID the request was submitted under cancels
    nothing and every sample keeps running after the client disconnects.
    """
    aborts: list[str] = []

    class _TokenizerManager:
        def abort_request(self, *, rid, abort_all):
            aborts.append(rid)

    handler = _handler(SimpleNamespace(tokenizer_manager=_TokenizerManager()))
    context = _CancelableContext("request-id")
    future: asyncio.Future = asyncio.Future()

    assert handler._arm_cancellation(future, context, "request-id", 3)

    monitor = asyncio.create_task(handler._handle_cancellation(future, context))
    await asyncio.sleep(0)
    context.cancel()
    await asyncio.wait_for(monitor, timeout=2)

    assert aborts == ["request-id_0", "request-id_1", "request-id_2"]


@pytest.mark.asyncio
async def test_single_sample_aborts_the_submitted_id_unchanged():
    aborts: list[str] = []

    class _TokenizerManager:
        def abort_request(self, *, rid, abort_all):
            aborts.append(rid)

    handler = _handler(SimpleNamespace(tokenizer_manager=_TokenizerManager()))
    context = _CancelableContext("request-id")
    future: asyncio.Future = asyncio.Future()

    assert handler._arm_cancellation(future, context, "request-id")

    monitor = asyncio.create_task(handler._handle_cancellation(future, context))
    await asyncio.sleep(0)
    context.cancel()
    await asyncio.wait_for(monitor, timeout=2)

    assert aborts == ["request-id"]
