# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from types import SimpleNamespace

import pytest

from dynamo.common.constants import DisaggregationMode
from dynamo.llm.exceptions import EngineShutdown
from dynamo.sglang.request_handlers.llm.decode_handler import DecodeWorkerHandler

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]


async def _collect(stream):
    return [item async for item in stream]


@pytest.mark.asyncio
async def test_cancellation_monitor_rechecks_shutdown_after_cleanup():
    handler = DecodeWorkerHandler.__new__(DecodeWorkerHandler)
    handler.shutdown_event = asyncio.Event()

    async def set_shutdown_when_cancelled(*_args):
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            handler.shutdown_event.set()
            raise

    handler._handle_cancellation = set_shutdown_when_cancelled
    request_id_future = asyncio.get_running_loop().create_future()
    request_id_future.set_result("sglang-request-id")
    context = SimpleNamespace(id=lambda: "request-id")

    with pytest.raises(EngineShutdown, match="shut down during token generation"):
        async with handler._cancellation_monitor(request_id_future, context):
            await asyncio.sleep(0)


@pytest.fixture
def decode_cancellation_case(monkeypatch):
    handler = DecodeWorkerHandler.__new__(DecodeWorkerHandler)
    handler.shutdown_event = None
    handler.use_sglang_tokenizer = False
    handler.config = SimpleNamespace(
        server_args=SimpleNamespace(served_model_name="test-model"),
        dynamo_args=SimpleNamespace(enable_rl=False),
    )
    handler._first_token_source = None
    handler._supports_ordered_cancellation = True
    monkeypatch.setattr(
        "dynamo.sglang.request_handlers.llm.decode_handler.new_sglang_request_id",
        lambda: "internal-request-id",
    )
    handler.shutdown_event = asyncio.Event()
    handler.serving_mode = DisaggregationMode.DECODE
    handler.enable_trace = False
    handler._routed_experts_kwargs = {}
    handler._enable_frontend_decoding = False
    handler._mm_hashes_supported = False
    handler._get_input_param = lambda request: {"input_ids": [1]}
    handler._build_sampling_params = lambda request: {"max_new_tokens": 1}
    handler._build_logprob_kwargs = lambda request: {}
    handler._resolve_lora = lambda request: None
    handler._priority_kwargs = lambda priority: {}
    monkeypatch.setattr(
        "dynamo.sglang.request_handlers.llm.decode_handler.require_reasoning_kwargs",
        lambda *args: {},
    )

    monkeypatch.setattr(
        "dynamo.sglang.request_handlers.cancellation.resolved_server_args",
        lambda server_args: server_args,
    )
    polling = asyncio.Queue()

    async def observed_sleep(delay):
        polling.put_nowait(delay)
        await asyncio.sleep(delay)

    monkeypatch.setattr(
        "dynamo.sglang.request_handlers.cancellation.asyncio",
        SimpleNamespace(**{**vars(asyncio), "sleep": observed_sleep}),
    )
    started = asyncio.Event()
    allow_registration = asyncio.Event()
    registered = asyncio.Event()
    allow_dispatch = asyncio.Event()
    allow_dispatch.set()
    dispatched = asyncio.Event()
    aborted = asyncio.Event()
    cancelled = asyncio.Event()
    drained = asyncio.Event()
    abort_calls = []
    registry = {}

    def cancellation_future():
        return asyncio.create_task(cancelled.wait())

    context = SimpleNamespace(
        id=lambda: "dynamo-request",
        trace_id="trace-request",
        async_killed_or_stopped=cancellation_future,
        is_stopped=cancelled.is_set,
        notify_first_token=lambda: None,
    )
    case = SimpleNamespace(
        handler=handler,
        context=context,
        started=started,
        allow_registration=allow_registration,
        registered=registered,
        allow_dispatch=allow_dispatch,
        dispatched=dispatched,
        aborted=aborted,
        cancelled=cancelled,
        drained=drained,
        abort_calls=abort_calls,
        registry=registry,
        polling=polling,
        finish_without_cancel=False,
        first_response=False,
        cancel_before_response=False,
        fail_registration=False,
        fail_dispatch=False,
        first_response_consumed=asyncio.Event(),
        native_request=None,
        request={
            "token_ids": [1],
            # No network is used by this engine double.
            "bootstrap_info": {
                "bootstrap_host": "localhost",
                "bootstrap_port": 0,
                "bootstrap_room": 1,
            },
        },
    )

    async def stream(rid):
        started.set()
        try:
            await allow_registration.wait()
            if case.fail_registration:
                raise ValueError("registration failed")
            state = SimpleNamespace(
                time_stats=SimpleNamespace(api_server_dispatch_finish_time=0.0)
            )
            registry[rid] = state
            registered.set()
            await allow_dispatch.wait()
            if case.fail_dispatch:
                raise ValueError("dispatch failed")
            state.time_stats.api_server_dispatch_finish_time = 1.0
            dispatched.set()
            if case.finish_without_cancel:
                return
            if case.first_response:
                if case.cancel_before_response:
                    cancelled.set()
                yield {
                    "output_ids": [],
                    "meta_info": {"id": rid, "finish_reason": None},
                }
                case.first_response_consumed.set()
            await aborted.wait()
        finally:
            registry.pop(rid, None)
            drained.set()

    def abort_request(*, rid, abort_all):
        abort_calls.append((rid, abort_all))
        # Registration alone does not make a request visible to the scheduler.
        if rid in registry and dispatched.is_set():
            aborted.set()

    async def async_generate(**kwargs):
        return stream(kwargs["rid"])

    def generate_request(request, context):
        case.native_request = request
        return stream(request.rid)

    handler.engine = SimpleNamespace(
        async_generate=async_generate,
        tokenizer_manager=SimpleNamespace(
            rid_to_state=registry,
            abort_request=abort_request,
            generate_request=generate_request,
        ),
    )
    return case


@pytest.mark.asyncio
@pytest.mark.timeout(5)
@pytest.mark.parametrize("disaggregated", [False, True])
@pytest.mark.parametrize("output_mode", ["tokens", "text", "native"])
@pytest.mark.parametrize(
    "phase", ["before_registration", "before_dispatch", "after_dispatch"]
)
@pytest.mark.parametrize("signal", ["cancel", "shutdown"])
async def test_decode_cancels_before_first_response(
    decode_cancellation_case, output_mode, phase, signal, disaggregated, caplog
):
    caplog.set_level(logging.INFO)
    case = decode_cancellation_case
    abort_message = f"Aborted Request ID: {case.context.id()}"
    if phase == "before_dispatch":
        case.allow_dispatch.clear()
    expected_id = "internal-request-id"
    if not disaggregated:
        case.handler.serving_mode = DisaggregationMode.AGGREGATED
    case.handler.use_sglang_tokenizer = output_mode == "text"
    if output_mode == "native":
        case.request["extra_args"] = {"sglang_tito": {"rid": "caller-supplied-id"}}

    consumer = asyncio.create_task(
        _collect(case.handler.generate(case.request, case.context))
    )
    try:
        await asyncio.wait_for(case.started.wait(), timeout=1)
        if phase != "before_registration":
            case.allow_registration.set()
            await asyncio.wait_for(case.registered.wait(), timeout=1)
        if phase == "after_dispatch":
            await asyncio.wait_for(case.dispatched.wait(), timeout=1)
        if signal == "cancel":
            case.cancelled.set()
        else:
            case.handler.shutdown_event.set()
        if phase == "before_registration":
            await asyncio.wait_for(case.polling.get(), timeout=1)
            assert not case.abort_calls
            assert abort_message not in caplog.messages
            case.allow_registration.set()
        elif phase == "before_dispatch":
            await asyncio.wait_for(case.polling.get(), timeout=1)
            assert not case.abort_calls
            assert abort_message not in caplog.messages
            assert not case.aborted.is_set()
            assert case.registry
            case.allow_dispatch.set()

        if signal == "shutdown":
            with pytest.raises(EngineShutdown):
                await asyncio.wait_for(consumer, timeout=1)
        else:
            assert await asyncio.wait_for(consumer, timeout=1) == []
        assert case.abort_calls == [(expected_id, False)]
        assert caplog.messages.count(abort_message) == 1
        assert case.dispatched.is_set()
        assert case.aborted.is_set()
        assert case.drained.is_set()
        assert not case.registry
        if output_mode == "native":
            assert case.native_request is not None
            expected_bootstrap = (
                ("localhost", 0, 1) if disaggregated else (None, None, None)
            )
            assert (
                case.native_request.bootstrap_host,
                case.native_request.bootstrap_port,
                case.native_request.bootstrap_room,
            ) == expected_bootstrap
        else:
            assert case.native_request is None
    finally:
        consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.timeout(5)
@pytest.mark.parametrize(
    "outcome", ["complete", "registration_error", "cancel_after_response"]
)
async def test_decode_cancellation_preserves_stream_lifetime(
    decode_cancellation_case, outcome
):
    case = decode_cancellation_case
    case.finish_without_cancel = outcome == "complete"
    case.fail_registration = outcome == "registration_error"
    case.first_response = outcome == "cancel_after_response"
    case.allow_registration.set()
    consumer = asyncio.create_task(
        _collect(case.handler.generate(case.request, case.context))
    )
    try:
        if outcome == "registration_error":
            with pytest.raises(ValueError, match="registration failed"):
                await asyncio.wait_for(consumer, timeout=1)
        else:
            if case.first_response:
                await asyncio.wait_for(case.first_response_consumed.wait(), timeout=1)
                case.cancelled.set()
            assert await asyncio.wait_for(consumer, timeout=1) == []
        expected = [("internal-request-id", False)] if case.first_response else []
        assert case.abort_calls == expected
        assert case.drained.is_set()
        assert not case.registry
    finally:
        consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.timeout(5)
@pytest.mark.parametrize("native", [False, True])
async def test_unsupported_runtime_uses_response_id_cancellation(
    decode_cancellation_case, native
):
    case = decode_cancellation_case
    case.handler._supports_ordered_cancellation = False
    case.first_response = True
    case.allow_registration.set()
    if native:
        case.request["extra_args"] = {"sglang_tito": {"rid": "caller-supplied-id"}}

    consumer = asyncio.create_task(
        _collect(case.handler.generate(case.request, case.context))
    )
    try:
        await asyncio.wait_for(case.first_response_consumed.wait(), timeout=1)
        assert not case.abort_calls

        case.cancelled.set()

        outputs = await asyncio.wait_for(consumer, timeout=1)
        assert case.abort_calls == [("internal-request-id", False)]
        if native:
            assert case.native_request.rid == "internal-request-id"
            assert (
                outputs[0]["engine_data"]["sglang_response"]["meta_info"]["id"]
                == "caller-supplied-id"
            )
        else:
            assert outputs == []
        assert case.drained.is_set()
        assert not case.registry
    finally:
        consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_cancelled_stream_drain_is_bounded(
    decode_cancellation_case, monkeypatch, caplog
):
    caplog.set_level(logging.WARNING)
    monkeypatch.setattr(
        "dynamo.sglang.request_handlers.cancellation._CANCELLATION_DRAIN_TIMEOUT_S",
        0,
    )

    async def stalled_stream():
        await asyncio.Future()
        yield {}

    cancellation_task = asyncio.create_task(asyncio.sleep(0))
    await cancellation_task
    outputs = await _collect(
        decode_cancellation_case.handler._stream_until_cancelled(
            stalled_stream(), cancellation_task
        )
    )

    assert outputs == []
    assert "Timed out draining SGLang stream after cancellation" in caplog.messages


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_cancelled_stream_drain_does_not_wait_for_resistant_iterator(
    decode_cancellation_case, monkeypatch, caplog
):
    caplog.set_level(logging.WARNING)
    monkeypatch.setattr(
        "dynamo.sglang.request_handlers.cancellation._CANCELLATION_DRAIN_TIMEOUT_S",
        0.01,
    )
    cancellation_seen = asyncio.Event()
    iterator_finished = asyncio.Event()

    async def resistant_stream():
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            cancellation_seen.set()
            await asyncio.sleep(0.2)
        iterator_finished.set()
        yield {}

    cancellation_task = asyncio.create_task(asyncio.sleep(0))
    await cancellation_task
    started = asyncio.get_running_loop().time()
    outputs = await _collect(
        decode_cancellation_case.handler._stream_until_cancelled(
            resistant_stream(), cancellation_task
        )
    )
    elapsed = asyncio.get_running_loop().time() - started

    assert outputs == []
    assert elapsed < 0.1
    assert "Timed out draining SGLang stream after cancellation" in caplog.messages
    await asyncio.wait_for(iterator_finished.wait(), timeout=1)
    assert cancellation_seen.is_set()


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_decode_cancellation_drains_buffered_empty_chunk(
    decode_cancellation_case,
):
    case = decode_cancellation_case
    case.first_response = True
    case.cancel_before_response = True
    case.allow_registration.set()

    # The buffered chunk arrives after stop, without yielding to the abort monitor.
    outputs = await asyncio.wait_for(
        _collect(case.handler.generate(case.request, case.context)), timeout=1
    )

    assert outputs == []
    assert case.abort_calls == [("internal-request-id", False)]
    assert case.first_response_consumed.is_set()
    assert case.drained.is_set()
    assert not case.registry


@pytest.mark.asyncio
@pytest.mark.timeout(5)
@pytest.mark.parametrize(
    "state_kind", ["no_registry", "missing", "no_stats", "null_stats"]
)
@pytest.mark.parametrize("abort_fails", [False, True])
async def test_cancellation_monitor_logs_only_submitted_abort(
    decode_cancellation_case, caplog, state_kind, abort_fails
):
    """Fallback paths log the context ID only after a successful abort call."""
    caplog.set_level(logging.INFO)
    case = decode_cancellation_case
    rid = case.context.trace_id
    registry = None if state_kind == "no_registry" else case.registry
    if state_kind == "no_stats":
        case.registry[rid] = object()
    elif state_kind == "null_stats":
        case.registry[rid] = SimpleNamespace(time_stats=None)

    def abort_request(*, rid, abort_all):
        assert f"Aborted Request ID: {case.context.id()}" not in caplog.messages
        if abort_fails:
            raise RuntimeError("abort failed")
        case.abort_calls.append((rid, abort_all))
        case.registry.pop(rid, None)

    manager = case.handler.engine.tokenizer_manager
    manager.abort_request = abort_request
    abort = case.handler._abort_sglang_request(
        manager, rid, registry, case.context.id()
    )
    if abort_fails:
        with pytest.raises(RuntimeError, match="abort failed"):
            await abort
    else:
        await abort
    assert case.abort_calls == ([] if abort_fails else [(rid, False)])
    assert caplog.messages.count(f"Aborted Request ID: {case.context.id()}") == (
        0 if abort_fails else 1
    )


@pytest.mark.asyncio
@pytest.mark.timeout(5)
@pytest.mark.parametrize("reuse_rid", [False, True])
@pytest.mark.parametrize("waiting_for", ["dispatch", "retry"])
async def test_cancellation_monitor_stops_when_request_finishes(
    decode_cancellation_case, monkeypatch, reuse_rid, waiting_for, caplog
):
    """Neither a dispatch wait nor a retry may abort a replacement request."""
    caplog.set_level(logging.INFO)
    case = decode_cancellation_case
    rid = case.context.trace_id
    case.registry[rid] = (
        SimpleNamespace(time_stats=SimpleNamespace(api_server_dispatch_finish_time=0.0))
        if waiting_for == "dispatch"
        else object()
    )
    sleeping = asyncio.Event()
    resume = asyncio.Event()

    async def controlled_sleep(delay):
        sleeping.set()
        await resume.wait()

    monkeypatch.setattr(
        "dynamo.sglang.request_handlers.cancellation.asyncio",
        SimpleNamespace(**{**vars(asyncio), "sleep": controlled_sleep}),
    )
    request_id_future = asyncio.get_running_loop().create_future()
    monitor = asyncio.create_task(
        case.handler._handle_cancellation(request_id_future, case.context, rid)
    )
    try:
        case.cancelled.set()
        await asyncio.wait_for(sleeping.wait(), timeout=1)
        # Model stream completion while the monitor is asleep.
        case.registry.pop(rid)
        if reuse_rid:
            case.registry[rid] = object()
        resume.set()
        await asyncio.wait_for(monitor, timeout=1)
        expected = [(rid, False)] if waiting_for == "retry" else []
        assert case.abort_calls == expected
        assert caplog.messages.count(f"Aborted Request ID: {case.context.id()}") == len(
            expected
        )
        assert not request_id_future.done()
    finally:
        monitor.cancel()
        await asyncio.gather(monitor, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.timeout(5)
@pytest.mark.parametrize("outcome", ["complete", "error"])
async def test_decode_stream_exit_cleans_up_pending_dispatch_wait(
    decode_cancellation_case, monkeypatch, outcome
):
    """Completion or dispatch failure must cancel the pending monitor."""
    case = decode_cancellation_case
    case.allow_registration.set()
    case.allow_dispatch.clear()
    case.finish_without_cancel = True
    sleeping = asyncio.Event()
    wait_cancelled = asyncio.Event()

    async def controlled_sleep(delay):
        sleeping.set()
        try:
            await asyncio.Future()
        finally:
            wait_cancelled.set()

    monkeypatch.setattr(
        "dynamo.sglang.request_handlers.cancellation.asyncio",
        SimpleNamespace(**{**vars(asyncio), "sleep": controlled_sleep}),
    )
    case.fail_dispatch = outcome == "error"
    consumer = asyncio.create_task(
        _collect(case.handler.generate(case.request, case.context))
    )
    try:
        await asyncio.wait_for(case.registered.wait(), timeout=1)
        case.cancelled.set()
        await asyncio.wait_for(sleeping.wait(), timeout=1)
        case.allow_dispatch.set()
        if outcome == "error":
            with pytest.raises(ValueError, match="dispatch failed"):
                await asyncio.wait_for(consumer, timeout=1)
        else:
            assert await asyncio.wait_for(consumer, timeout=1) == []
        assert not case.abort_calls
        assert case.drained.is_set()
        assert not case.registry
        assert wait_cancelled.is_set()
    finally:
        consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.timeout(5)
@pytest.mark.parametrize(
    "engine", [None, SimpleNamespace(), SimpleNamespace(tokenizer_manager=None)]
)
@pytest.mark.parametrize("signal", ["cancel", "shutdown"])
async def test_cancellation_monitor_without_tokenizer_manager(
    decode_cancellation_case, engine, signal
):
    """Engine-less handlers retain cancellation and shutdown cleanup behavior."""
    case = decode_cancellation_case
    case.handler.engine = engine
    if signal == "cancel":
        case.cancelled.set()
    else:
        case.handler.shutdown_event.set()
    monitor = case.handler._handle_cancellation(
        asyncio.get_running_loop().create_future(), case.context, case.context.trace_id
    )
    if signal == "shutdown":
        with pytest.raises(EngineShutdown):
            await asyncio.wait_for(monitor, timeout=1)
    else:
        await asyncio.wait_for(monitor, timeout=1)


@pytest.mark.asyncio
@pytest.mark.timeout(5)
@pytest.mark.parametrize("registry", ["missing", "non_mapping"])
async def test_cancellation_monitor_without_request_registry(
    decode_cancellation_case, registry
):
    case = decode_cancellation_case
    abort_calls = []

    def abort_request(*, rid, abort_all):
        abort_calls.append((rid, abort_all))

    tokenizer_manager = SimpleNamespace(abort_request=abort_request)
    if registry == "non_mapping":
        tokenizer_manager.rid_to_state = []
    case.handler.engine = SimpleNamespace(tokenizer_manager=tokenizer_manager)
    case.cancelled.set()

    await asyncio.wait_for(
        case.handler._handle_cancellation(
            asyncio.get_running_loop().create_future(),
            case.context,
            case.context.trace_id,
        ),
        timeout=1,
    )

    assert abort_calls == [(case.context.trace_id, False)]


@pytest.mark.asyncio
@pytest.mark.timeout(5)
@pytest.mark.parametrize(
    "server_args, has_dispatch_time, expect_retry",
    [
        ({}, True, False),
        ({"dp_size": 2}, True, False),
        ({"enable_dp_attention": True}, True, True),
        (
            {
                "enable_dp_attention": True,
                "enable_dp_attention_local_control_broadcast": True,
            },
            True,
            False,
        ),
        ({"pp_size": 2}, True, True),
        ({}, False, True),
    ],
    ids=["tp", "dp", "dp-attention-global", "dp-attention-local", "pp", "legacy"],
)
async def test_cancellation_monitor_retries_only_without_ordered_dispatch(
    decode_cancellation_case, monkeypatch, server_args, has_dispatch_time, expect_retry
):
    case = decode_cancellation_case
    # SGLang may keep raw and effective server arguments separately.
    monkeypatch.setattr(
        "dynamo.sglang.request_handlers.cancellation.resolved_server_args",
        lambda raw_args: SimpleNamespace(**server_args),
    )
    rid = case.context.trace_id
    case.registry[rid] = (
        SimpleNamespace(time_stats=SimpleNamespace(api_server_dispatch_finish_time=1.0))
        if has_dispatch_time
        else object()
    )

    def abort_request(*, rid, abort_all):
        case.abort_calls.append((rid, abort_all))
        # Leave the state registered after the first abort. A single-abort path
        # must exit on its own, independently of the consumer draining the stream.
        if len(case.abort_calls) == 2:
            case.registry.pop(rid)

    case.handler.engine.tokenizer_manager.abort_request = abort_request
    case.cancelled.set()
    await asyncio.wait_for(
        case.handler._handle_cancellation(
            asyncio.get_running_loop().create_future(), case.context, rid
        ),
        timeout=1,
    )
    assert case.abort_calls == [(rid, False)] * (2 if expect_retry else 1)


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_cancellation_monitor_bounds_abort_retries(
    decode_cancellation_case, monkeypatch, caplog
):
    caplog.set_level(logging.WARNING)
    case = decode_cancellation_case
    rid = "internal-request-id"
    case.registry[rid] = object()

    async def immediate_sleep(_delay):
        await asyncio.sleep(0)

    monkeypatch.setattr(
        "dynamo.sglang.request_handlers.cancellation.asyncio",
        SimpleNamespace(**{**vars(asyncio), "sleep": immediate_sleep}),
    )
    monkeypatch.setattr(
        "dynamo.sglang.request_handlers.cancellation.resolved_server_args",
        lambda raw_args: SimpleNamespace(),
    )

    await asyncio.wait_for(
        case.handler._abort_sglang_request(
            case.handler.engine.tokenizer_manager,
            rid,
            case.registry,
            case.context.id(),
        ),
        timeout=1,
    )

    assert case.abort_calls == [(rid, False)] * 9
    assert (
        "SGLang request internal-request-id remained registered after 8 abort retries"
        in caplog.messages
    )


@pytest.mark.asyncio
@pytest.mark.timeout(5)
@pytest.mark.parametrize("phase", ["registration", "dispatch"])
async def test_cancellation_monitor_preserves_abort_ordering(
    decode_cancellation_case, phase
):
    case = decode_cancellation_case
    rid = "internal-request-id"
    state = SimpleNamespace(
        time_stats=SimpleNamespace(api_server_dispatch_finish_time=0.0)
    )

    if phase == "registration":
        case.cancelled.set()
        operation = asyncio.create_task(
            case.handler._handle_cancellation(
                asyncio.get_running_loop().create_future(),
                case.context,
                rid,
            )
        )
    else:
        case.registry[rid] = state
        operation = asyncio.create_task(
            case.handler._abort_sglang_request(
                case.handler.engine.tokenizer_manager,
                rid,
                case.registry,
                case.context.id(),
            )
        )

    await asyncio.wait_for(case.polling.get(), timeout=1)
    assert not case.abort_calls
    case.registry[rid] = state
    state.time_stats.api_server_dispatch_finish_time = 1.0

    await asyncio.wait_for(operation, timeout=1)
    assert case.abort_calls == [(rid, False)]
