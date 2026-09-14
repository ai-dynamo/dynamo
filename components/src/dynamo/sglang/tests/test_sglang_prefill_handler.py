# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace

import pytest

from dynamo.sglang.request_handlers import cancellation
from dynamo.sglang.request_handlers.llm.prefill_handler import PrefillWorkerHandler

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("ordered_cancellation", "expected_id"),
    [(True, "internal-request-id"), (False, "internal-request-id")],
)
async def test_prefill_selects_request_id_for_runtime_capability(
    monkeypatch: pytest.MonkeyPatch,
    ordered_cancellation: bool,
    expected_id: str,
):
    captured_kwargs = {}

    async def empty_results():
        if False:
            yield {}

    class _Engine:
        async def async_generate(self, **kwargs):
            captured_kwargs.update(kwargs)
            return empty_results()

    handler = PrefillWorkerHandler.__new__(PrefillWorkerHandler)
    handler.engine = _Engine()
    handler.bootstrap_host = "127.0.0.1"
    handler.bootstrap_port = 1234
    handler.enable_trace = False
    handler._consume_tasks = set()
    handler._supports_ordered_cancellation = ordered_cancellation
    handler._generate_bootstrap_room = lambda: 17
    handler._get_input_param = lambda request: {"input_ids": request["token_ids"]}
    handler._resolve_lora = lambda request: None
    handler._priority_kwargs = lambda priority: {}
    monkeypatch.setattr(
        "dynamo.sglang.request_handlers.llm.prefill_handler.require_reasoning_kwargs",
        lambda engine, request: {},
    )
    monkeypatch.setattr(
        "dynamo.sglang.request_handlers.llm.prefill_handler.new_sglang_request_id",
        lambda: "internal-request-id",
    )

    context = SimpleNamespace(
        id=lambda: "request-id",
        trace_id=None,
        trace_headers=lambda: {},
    )
    stream = handler.generate(
        {
            "request": {"token_ids": [1, 2, 3], "routing": {}},
            "sampling_params": {},
        },
        context,
    )
    await anext(stream)

    assert captured_kwargs["rid"] == expected_id
    await stream.aclose()


@pytest.mark.asyncio
@pytest.mark.timeout(5)
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize(
    "phase", ["before_registration", "before_dispatch", "after_dispatch"]
)
async def test_prefill_cancellation_waits_for_dispatch_and_drains(
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    native: bool,
):
    request_id = "native-request-id" if native else "request-id"
    engine_request_id = "internal-request-id"
    registry = {}
    abort_calls = []
    started = asyncio.Event()
    registered = asyncio.Event()
    dispatched = asyncio.Event()
    aborted = asyncio.Event()
    drained = asyncio.Event()
    cancelled = asyncio.Event()
    polling = asyncio.Event()
    allow_registration = asyncio.Event()
    allow_dispatch = asyncio.Event()
    if phase != "before_registration":
        allow_registration.set()
    if phase == "after_dispatch":
        allow_dispatch.set()

    async def results(rid):
        started.set()
        try:
            await allow_registration.wait()
            state = SimpleNamespace(
                time_stats=SimpleNamespace(api_server_dispatch_finish_time=0.0)
            )
            registry[rid] = state
            registered.set()
            await allow_dispatch.wait()
            state.time_stats.api_server_dispatch_finish_time = 1.0
            dispatched.set()
            await aborted.wait()
        finally:
            registry.pop(rid, None)
            drained.set()
        if False:
            yield {}

    def abort_request(*, rid, abort_all):
        abort_calls.append((rid, abort_all))
        if rid in registry and dispatched.is_set():
            aborted.set()

    def generate_request(request, _context):
        return results(request.rid)

    class _Engine:
        tokenizer_manager = SimpleNamespace(
            rid_to_state=registry,
            abort_request=abort_request,
            generate_request=generate_request,
        )

        async def async_generate(self, **kwargs):
            return results(kwargs["rid"])

    handler = PrefillWorkerHandler.__new__(PrefillWorkerHandler)
    handler.engine = _Engine()
    handler.config = SimpleNamespace(
        server_args=SimpleNamespace(pp_size=1, enable_dp_attention=False)
    )
    handler.shutdown_event = None
    handler.bootstrap_host = "127.0.0.1"
    handler.bootstrap_port = 1234
    handler.enable_trace = False
    handler._consume_tasks = set()
    handler._supports_ordered_cancellation = True
    handler._generate_bootstrap_room = lambda: 17
    handler._get_input_param = lambda request: {"input_ids": request["token_ids"]}
    handler._resolve_lora = lambda request: None
    handler._priority_kwargs = lambda priority: {}
    monkeypatch.setattr(
        "dynamo.sglang.request_handlers.llm.prefill_handler.require_reasoning_kwargs",
        lambda engine, request: {},
    )
    monkeypatch.setattr(
        cancellation, "resolved_server_args", lambda server_args: server_args
    )
    monkeypatch.setattr(
        "dynamo.sglang.request_handlers.llm.prefill_handler.new_sglang_request_id",
        lambda: engine_request_id,
    )
    real_sleep = asyncio.sleep

    async def observed_sleep(delay):
        polling.set()
        await real_sleep(0)

    monkeypatch.setattr(
        cancellation,
        "asyncio",
        SimpleNamespace(**{**vars(asyncio), "sleep": observed_sleep}),
    )
    context = SimpleNamespace(
        id=lambda: request_id,
        trace_id=None,
        trace_headers=lambda: {},
        async_killed_or_stopped=lambda: asyncio.create_task(cancelled.wait()),
    )
    inner_request = {"token_ids": [1, 2, 3], "routing": {}}
    if native:
        inner_request["extra_args"] = {"sglang_tito": {"rid": request_id}}
    stream = handler.generate(
        {
            "request": inner_request,
            "sampling_params": {},
        },
        context,
    )
    await anext(stream)
    consumer = asyncio.create_task(_drain(stream))
    await asyncio.wait_for(started.wait(), timeout=1)
    if phase != "before_registration":
        await asyncio.wait_for(registered.wait(), timeout=1)
    if phase == "after_dispatch":
        await asyncio.wait_for(dispatched.wait(), timeout=1)

    cancelled.set()
    if phase != "after_dispatch":
        await asyncio.wait_for(polling.wait(), timeout=1)
        assert not abort_calls
        allow_registration.set()
        allow_dispatch.set()

    assert await asyncio.wait_for(consumer, timeout=1) == []
    assert abort_calls == [(engine_request_id, False)]
    assert dispatched.is_set()
    assert aborted.is_set()
    assert drained.is_set()
    assert not registry


async def _drain(stream):
    return [item async for item in stream]
