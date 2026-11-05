# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST

from dynamo.health_check import HEALTH_CHECK_KEY
from dynamo.llm import HttpError
from dynamo.llm.exceptions import InvalidArgument
from dynamo.sglang.health_check import SglangPrefillHealthCheckPayload
from dynamo.sglang.request_handlers import cancellation
from dynamo.sglang.request_handlers.llm.prefill_handler import PrefillWorkerHandler

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.fault_tolerance,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]


@pytest.mark.asyncio
@pytest.mark.timeout(5)
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("ordered_cancellation", [False, True])
@pytest.mark.parametrize("error_bearing_abort", [False, True])
@pytest.mark.parametrize(
    "phase", ["before_registration", "before_dispatch", "after_dispatch"]
)
async def test_prefill_cancellation_waits_for_dispatch_and_drains(
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    native: bool,
    ordered_cancellation: bool,
    error_bearing_abort: bool,
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
            response = {"meta_info": {"id": rid}}
            yield response
            await aborted.wait()
            if error_bearing_abort:
                yield {
                    "meta_info": {
                        "id": rid,
                        "finish_reason": {
                            "type": "abort",
                            "status_code": 500,
                            "message": "Prefill bootstrap failed: Aborted by AbortReq.",
                        },
                    }
                }
        finally:
            registry.pop(rid, None)
            drained.set()

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
    handler._abort_tasks = set()
    handler._consume_tasks = set()
    handler._supports_ordered_cancellation = ordered_cancellation
    handler._generate_bootstrap_room = lambda: 17
    handler._get_input_param = lambda request: {"input_ids": request["token_ids"]}
    handler._resolve_lora = lambda request: None
    handler._priority_kwargs = lambda priority: {}
    monkeypatch.setattr(
        "dynamo.sglang.request_handlers.llm.prefill_handler.require_reasoning_kwargs",
        lambda *args, **kwargs: {},
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
        if ordered_cancellation:
            await asyncio.wait_for(polling.wait(), timeout=1)
        else:
            await asyncio.sleep(0)
        assert not abort_calls
        allow_registration.set()
        allow_dispatch.set()

    assert await asyncio.wait_for(consumer, timeout=1) == []
    assert abort_calls == [(engine_request_id, False)]
    assert dispatched.is_set()
    assert aborted.is_set()
    assert drained.is_set()
    assert not registry


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_native_batched_prefill_disables_ordered_cancellation(monkeypatch):
    captured_submitted_id = None
    captured_native_request = None

    async def native_results():
        if False:
            yield None

    handler = PrefillWorkerHandler.__new__(PrefillWorkerHandler)
    handler.engine = SimpleNamespace()
    handler.bootstrap_host = "127.0.0.1"
    handler.bootstrap_port = 1234
    handler.enable_trace = False
    handler._consume_tasks = set()
    handler._supports_ordered_cancellation = True
    handler._generate_bootstrap_room = lambda: 17
    handler._get_input_param = lambda request: {"input_ids": request["token_ids"]}
    handler._resolve_lora = lambda request: None
    handler._priority_kwargs = lambda priority: {}

    async def capture_consume(_results, submitted_request_id, _context):
        nonlocal captured_submitted_id
        captured_submitted_id = submitted_request_id

    def capture_native_stream(_engine, request):
        nonlocal captured_native_request
        captured_native_request = request
        return native_results()

    handler._consume_results = capture_consume
    monkeypatch.setattr(
        "dynamo.sglang.request_handlers.llm.prefill_handler.new_sglang_request_id",
        lambda: "internal-request-id",
    )
    monkeypatch.setattr(
        "dynamo.sglang.request_handlers.llm.prefill_handler.native_generate_stream",
        capture_native_stream,
    )
    context = SimpleNamespace(
        id=lambda: "request-id", trace_id=None, trace_headers=lambda: {}
    )
    stream = handler.generate(
        {
            "request": {
                "token_ids": [[1], [2]],
                "routing": {},
                "extra_args": {"sglang_tito": {}},
            },
            "sampling_params": {},
        },
        context,
    )

    await anext(stream)
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    assert captured_submitted_id is None
    assert captured_native_request.input_ids == [[1], [2]]


def _kv_hit_prefill_handler(monkeypatch, responses, *, cancelled):
    async def results():
        for response in responses:
            yield response

    class _Engine:
        async def async_generate(self, **kwargs):
            return results()

    handler = PrefillWorkerHandler.__new__(PrefillWorkerHandler)
    handler.engine = _Engine()
    handler.config = SimpleNamespace(server_args=SimpleNamespace())
    handler.shutdown_event = None
    handler.bootstrap_host = "127.0.0.1"
    handler.bootstrap_port = 1234
    handler.enable_trace = False
    handler._consume_tasks = set()
    handler._generate_bootstrap_room = lambda: 17
    handler._get_input_param = lambda request: {"input_ids": request["token_ids"]}
    handler._resolve_lora = lambda request: None
    handler._priority_kwargs = lambda priority: {}

    @asynccontextmanager
    async def cancellation_monitor(*args, **kwargs):
        async def fire_or_wait():
            if not cancelled:
                await asyncio.Future()

        task = asyncio.create_task(fire_or_wait())
        if cancelled:
            await task
        try:
            yield task
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    handler._cancellation_monitor = cancellation_monitor
    monkeypatch.setattr(
        "dynamo.sglang.request_handlers.llm.prefill_handler.require_reasoning_kwargs",
        lambda engine, request, **kwargs: {},
    )
    return handler


@pytest.fixture
def prefill_responses():
    return [
        {
            "meta_info": {
                "id": "request-id",
                "finish_reason": {"type": "length"},
                "prompt_tokens": 3,
                "cached_tokens": 2,
            }
        }
    ]


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_prefill_reports_kv_cache_hit_after_bootstrap(
    monkeypatch, prefill_responses
):
    handler = _kv_hit_prefill_handler(monkeypatch, prefill_responses, cancelled=False)
    context = SimpleNamespace(
        id=lambda: "request-id", trace_id=None, trace_headers=lambda: {}
    )

    chunks = await _drain(
        handler.generate(
            {"request": {"token_ids": [1, 2, 3], "routing": {}}, "sampling_params": {}},
            context,
        )
    )

    assert len(chunks) == 2
    assert "disaggregated_params" in chunks[0]
    assert chunks[1] == {
        "token_ids": [],
        "engine_data": {"kv_cache_hit": {"prompt_tokens": 3, "reused_tokens": 2}},
    }


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_prefill_omits_kv_cache_hit_after_cancellation(
    monkeypatch, prefill_responses
):
    handler = _kv_hit_prefill_handler(monkeypatch, prefill_responses, cancelled=True)
    context = SimpleNamespace(
        id=lambda: "request-id", trace_id=None, trace_headers=lambda: {}
    )

    chunks = await _drain(
        handler.generate(
            {"request": {"token_ids": [1, 2, 3], "routing": {}}, "sampling_params": {}},
            context,
        )
    )

    assert len(chunks) == 1
    assert "disaggregated_params" in chunks[0]


@pytest.mark.asyncio
@pytest.mark.timeout(5)
@pytest.mark.parametrize("native_metadata", [False, True])
@pytest.mark.parametrize(
    "status_code,message,exception",
    [
        (400, "Input length exceeds the maximum allowed length", InvalidArgument),
        (500, "Prefill bootstrap failed", HttpError),
        (500, "Prefill transfer failed", HttpError),
        (500, "Grammar accept_token failed", HttpError),
    ],
)
async def test_prefill_propagates_backend_abort(
    monkeypatch, native_metadata, status_code, message, exception
):
    meta_info = {
        "id": "request-id",
        "finish_reason": {
            "type": "abort",
            "status_code": status_code,
            "message": message,
        },
        "prompt_tokens": 3,
        "cached_tokens": 2,
    }
    response = {"meta_info": meta_info}
    if native_metadata:
        response = {"engine_data": {"sglang_response": response}}
    handler = _kv_hit_prefill_handler(monkeypatch, [response], cancelled=False)
    context = SimpleNamespace(
        id=lambda: "request-id", trace_id=None, trace_headers=lambda: {}
    )
    stream = handler.generate(
        {"request": {"token_ids": [1, 2, 3], "routing": {}}, "sampling_params": {}},
        context,
    )

    assert "disaggregated_params" in await anext(stream)
    with pytest.raises(exception, match=message):
        await anext(stream)


@pytest.mark.asyncio
@pytest.mark.timeout(5)
@pytest.mark.parametrize("native_metadata", [False, True])
@pytest.mark.parametrize("cancelled", [False, True])
async def test_prefill_bootstrap_abort_after_router_cancellation(
    monkeypatch, native_metadata, cancelled
):
    """Only an already cancelled request may suppress SGLang's bootstrap 500."""
    message = (
        "Prefill bootstrap failed for request rank=0 req.rid='request-id' "
        "req.bootstrap_room=17 with exception Aborted by AbortReq."
    )
    response = {
        "meta_info": {
            "id": "request-id",
            "finish_reason": {
                "type": "abort",
                "status_code": 500,
                "message": message,
            },
            "prompt_tokens": 3,
            "cached_tokens": 2,
        }
    }
    if native_metadata:
        response = {"engine_data": {"sglang_response": response}}
    handler = _kv_hit_prefill_handler(monkeypatch, [response], cancelled=cancelled)
    context = SimpleNamespace(
        id=lambda: "request-id", trace_id=None, trace_headers=lambda: {}
    )
    stream = handler.generate(
        {"request": {"token_ids": [1, 2, 3], "routing": {}}, "sampling_params": {}},
        context,
    )

    assert "disaggregated_params" in await anext(stream)
    if cancelled:
        assert await _drain(stream) == []
    else:
        with pytest.raises(HttpError, match="Aborted by AbortReq"):
            await anext(stream)


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_prefill_reports_backend_cancellation_without_success_cache_stats(
    monkeypatch,
):
    handler = _kv_hit_prefill_handler(
        monkeypatch,
        [
            {
                "meta_info": {
                    "id": "request-id",
                    "finish_reason": {"type": "abort", "message": "Aborted"},
                    "prompt_tokens": 3,
                    "cached_tokens": 2,
                }
            }
        ],
        cancelled=False,
    )
    context = SimpleNamespace(
        id=lambda: "request-id", trace_id=None, trace_headers=lambda: {}
    )
    chunks = await _drain(
        handler.generate(
            {"request": {"token_ids": [1, 2, 3], "routing": {}}, "sampling_params": {}},
            context,
        )
    )
    assert len(chunks) == 2
    assert chunks[1] == {"token_ids": [], "finish_reason": "cancelled"}


async def _drain(stream):
    return [item async for item in stream]


def _bootstrap_capture_handler(monkeypatch, *, native):
    """Prefill handler whose engine records the request it is given."""
    calls = []

    async def results():
        yield {"meta_info": {"id": "request-id", "finish_reason": {"type": "length"}}}

    class _Engine:
        async def async_generate(self, **kwargs):
            calls.append(kwargs)
            return results()

    def capture_native_stream(_engine, request):
        calls.append(
            {
                "bootstrap_host": request.bootstrap_host,
                "bootstrap_port": request.bootstrap_port,
                "bootstrap_room": request.bootstrap_room,
                "input_ids": request.input_ids,
            }
        )
        return results()

    handler = PrefillWorkerHandler.__new__(PrefillWorkerHandler)
    handler.engine = _Engine()
    handler.config = SimpleNamespace(
        server_args=SimpleNamespace(), dynamo_args=SimpleNamespace(enable_rl=False)
    )
    handler.shutdown_event = None
    handler.bootstrap_host = "10.0.0.5"
    handler.bootstrap_port = 1234
    handler.enable_trace = False
    handler._consume_tasks = set()
    handler._default_sampling_params = {}
    handler._generate_bootstrap_room = lambda: 17
    handler._get_input_param = lambda request: {"input_ids": request["token_ids"]}
    handler._resolve_lora = lambda request: None
    handler._priority_kwargs = lambda priority: {}

    @asynccontextmanager
    async def cancellation_monitor(*args, **kwargs):
        # Never fires: these requests are not cancelled.
        never_cancelled = asyncio.get_running_loop().create_future()
        try:
            yield never_cancelled
        finally:
            never_cancelled.cancel()

    handler._cancellation_monitor = cancellation_monitor
    monkeypatch.setattr(
        "dynamo.sglang.request_handlers.llm.prefill_handler.require_reasoning_kwargs",
        lambda engine, request, **kwargs: {},
    )
    if native:
        monkeypatch.setattr(
            "dynamo.sglang.request_handlers.llm.prefill_handler.native_generate_stream",
            capture_native_stream,
        )
    return handler, calls


def _flat_request(**fields):
    return {
        "token_ids": [7, 8],
        "sampling_options": {"temperature": 0.0},
        "stop_conditions": {"max_tokens": 1},
        **fields,
    }


# (request, expected engine-facing bootstrap (host, port, room), is a probe)
_PROBE_HOST_CASES = {
    "default_probe": (
        lambda: SglangPrefillHealthCheckPayload().to_dict(),
        (FAKE_BOOTSTRAP_HOST, 0, 0),
        True,
    ),
    "probe_without_bootstrap_info": (
        lambda: _flat_request(**{HEALTH_CHECK_KEY: True}),
        (FAKE_BOOTSTRAP_HOST, 1234, 17),
        True,
    ),
    "probe_with_real_host": (
        lambda: _flat_request(
            **{
                HEALTH_CHECK_KEY: True,
                "bootstrap_info": {"bootstrap_host": "10.0.0.9", "bootstrap_room": 5},
            }
        ),
        (FAKE_BOOTSTRAP_HOST, 1234, 5),
        True,
    ),
    "wrapped_probe": (
        lambda: {
            HEALTH_CHECK_KEY: True,
            "request": {"token_ids": [7, 8], "routing": {}},
            "sampling_params": {},
        },
        (FAKE_BOOTSTRAP_HOST, 1234, 17),
        True,
    ),
    "ordinary_router_request": (
        lambda: {
            "request": {
                "token_ids": [7, 8],
                "routing": {},
                "bootstrap_info": {
                    "bootstrap_host": "10.0.0.9",
                    "bootstrap_port": 4321,
                    "bootstrap_room": 5,
                },
            },
            "sampling_params": {},
        },
        ("10.0.0.9", 4321, 5),
        False,
    ),
    "ordinary_local_request": (
        lambda: _flat_request(),
        ("10.0.0.5", 1234, 17),
        False,
    ),
    "non_bool_marker": (
        lambda: _flat_request(**{HEALTH_CHECK_KEY: 1}),
        ("10.0.0.5", 1234, 17),
        False,
    ),
}


@pytest.mark.asyncio
@pytest.mark.timeout(5)
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("case", list(_PROBE_HOST_CASES))
async def test_prefill_probe_forces_fake_bootstrap_host(monkeypatch, case, native):
    """A marked probe reaches the engine with FAKE_BOOTSTRAP_HOST, whatever
    host its payload names; ordinary requests keep their real handoff."""
    monkeypatch.delenv("DYN_HEALTH_CHECK_PAYLOAD", raising=False)
    make_request, (host, port, room), probe = _PROBE_HOST_CASES[case]
    request = make_request()
    if native:
        inner = request.get("request", request)
        inner["extra_args"] = {"sglang_tito": {}}
    handler, calls = _bootstrap_capture_handler(monkeypatch, native=native)
    context = SimpleNamespace(
        id=lambda: "request-id", trace_id=None, trace_headers=lambda: {}
    )

    chunks = await _drain(handler.generate(request, context))

    assert len(calls) == 1
    call = calls[0]
    assert (
        call["bootstrap_host"],
        call["bootstrap_port"],
        call["bootstrap_room"],
    ) == (host, port, room)
    if probe:
        # The probe relays engine output instead of a handoff.
        assert all("disaggregated_params" not in chunk for chunk in chunks)
    else:
        assert chunks[0]["disaggregated_params"] == {
            "bootstrap_host": host,
            "bootstrap_port": port,
            "bootstrap_room": room,
        }
