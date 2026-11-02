# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned native overloads retain capacity classification without engine details."""

import asyncio
from unittest.mock import Mock

import pytest

from dynamo.llm.exceptions import EngineShutdown, HttpError, WorkerOverloaded
from dynamo.vllm import decision_native_lifecycle as native_lifecycle
from dynamo.vllm.decision_native_errors import NativeExecutionError
from dynamo.vllm.tests.decision_head_test_support import DecisionContext
from dynamo.vllm.tests.decision_head_test_support import Engine as HeadEngine
from dynamo.vllm.tests.decision_head_test_support import NativeApi as HeadNativeApi
from dynamo.vllm.tests.decision_head_test_support import adapter as head_adapter
from dynamo.vllm.tests.decision_head_test_support import question as head_question
from dynamo.vllm.tests.decision_head_test_support import wire as head_wire
from dynamo.vllm.tests.decision_read_contract_support import adapter as read_adapter
from dynamo.vllm.tests.decision_read_contract_support import question as read_question
from dynamo.vllm.tests.decision_read_contract_support import wire as read_wire
from dynamo.vllm.tests.decision_read_test_support import Engine

pytest_plugins = ["dynamo.vllm.tests.decision_pinned_fixtures"]
pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.timeout(30),
]


@pytest.fixture
def native_errors(pinned_errors):
    return (
        pinned_errors.QueueOverflowError,
        pinned_errors.MaxQueuedTokensError,
    ), WorkerOverloaded


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["head", "read"])
@pytest.mark.parametrize("phase", ["admission", "engine"])
@pytest.mark.parametrize("error_index", [0, 1])
async def test_actual_native_overload_preserves_capacity_and_drains_children(
    native_reads, native_errors, monkeypatch, kind, phase, error_index
):
    error_types, overload_error = native_errors
    if kind == "head":
        api, engine = HeadNativeApi(), HeadEngine()
        monkeypatch.setattr(native_lifecycle, "load_native", lambda: api)
        instance = await head_adapter(engine)
        request = head_wire(questions=[head_question(), head_question(1)])
        method = "encode"
    else:
        api, engine = native_reads, Engine()
        instance = await read_adapter(engine)
        request = read_wire(questions=[read_question(), read_question(1)])
        method = "generate"
    api.overload_errors, api.overload_error = error_types, overload_error
    error = error_types[error_index]()
    error.message = "private engine diagnostic must not escape"
    if phase == "admission":
        engine.check_admission = Mock(side_effect=error)
    else:
        engine.block = asyncio.Event()
        original = getattr(engine, method)

        async def fail_one_child(*args, **kwargs):
            request_id = kwargs["request_id"] if "request_id" in kwargs else args[2]
            if request_id.endswith("-0"):
                await engine.started.wait()
                raise error
            async for output in original(*args, **kwargs):
                yield output

        monkeypatch.setattr(engine, method, fail_one_child)
    with pytest.raises(overload_error) as caught:
        await instance.execute(
            request, DecisionContext(), request_id="overloaded", cancelled=lambda: False
        )
    assert str(caught.value) == "Native decision engine capacity is exhausted"
    assert engine.active == 0
    if phase == "admission":
        assert engine.calls == []
    else:
        assert engine.admissions == [2]
        assert len(engine.aborted) == 2
        assert instance.semaphore._value == instance.concurrency


@pytest.mark.asyncio
async def test_untrusted_status_like_errors_cannot_select_an_http_status(
    native_reads, native_errors
):
    class UnknownError(RuntimeError):
        http_status = 503
        message = "unknown"

    error_types, overload_error = native_errors
    native_reads.overload_errors, native_reads.overload_error = (
        error_types,
        overload_error,
    )
    engine = Engine()
    cause = UnknownError("original failure")
    engine.check_admission = Mock(side_effect=cause)
    instance = await read_adapter(engine)
    with pytest.raises(HttpError) as failure:
        await instance.execute(
            read_wire(),
            DecisionContext(),
            request_id="unknown",
            cancelled=lambda: False,
        )
    assert failure.value.code == 500
    assert failure.value.__cause__ is cause
    assert "original failure" not in failure.value.message


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["head", "read"])
@pytest.mark.parametrize(
    "fault", ["generate", "dead", "validation", "retryable", "generation"]
)
async def test_actual_engine_failures_preserve_availability_and_cleanup(
    native_reads, pinned_engine_errors, pinned_errors, monkeypatch, kind, fault
):
    callback = Mock()
    if kind == "head":
        api, engine = HeadNativeApi(), HeadEngine()
        monkeypatch.setattr(native_lifecycle, "load_native", lambda: api)
        instance = await head_adapter(engine, on_engine_dead=callback)
        request = head_wire(questions=[head_question(), head_question(1)])
        method = "encode"
    else:
        api, engine = native_reads, Engine()
        instance = await read_adapter(engine, on_engine_dead=callback)
        request = read_wire(questions=[read_question(), read_question(1)])
        method = "generate"
    api.engine_generate_error = pinned_engine_errors.EngineGenerateError
    api.engine_dead_error = pinned_engine_errors.EngineDeadError
    api.retryable_request_error = pinned_errors.RetryableRequestError
    api.validation_error = pinned_errors.VLLMValidationError
    cause = {
        "generate": api.engine_generate_error,
        "dead": api.engine_dead_error,
        "validation": api.validation_error,
        "retryable": api.retryable_request_error,
        "generation": pinned_errors.GenerationError,
    }[fault]("private engine diagnostic")
    engine.block = asyncio.Event()
    original = getattr(engine, method)
    closed = []

    async def fail_one_child(*args, **kwargs):
        request_id = kwargs["request_id"] if "request_id" in kwargs else args[2]
        assert instance.semaphore._value < instance.concurrency
        try:
            if request_id.endswith("-0"):
                await engine.started.wait()
                raise cause
            async for output in original(*args, **kwargs):
                yield output
        finally:
            closed.append(request_id)

    def shutdown():
        assert engine.active == 0
        assert len(engine.aborted) == 2 and len(closed) == 2

    callback.side_effect = shutdown
    assert instance.on_engine_dead is callback
    monkeypatch.setattr(engine, method, fail_one_child)
    expected = {
        "generate": HttpError,
        "dead": EngineShutdown,
        "validation": NativeExecutionError,
        "retryable": HttpError,
        "generation": HttpError,
    }[fault]
    with pytest.raises(expected) as failure:
        await instance.execute(
            request,
            DecisionContext(),
            request_id="engine-failed",
            cancelled=lambda: False,
        )
    assert failure.value.__cause__ is cause
    if fault in ("generate", "retryable", "generation"):
        assert failure.value.code == (500 if fault == "generation" else 503)
    assert "private engine diagnostic" not in str(failure.value)
    assert engine.active == 0 and len(engine.aborted) == 2 and len(closed) == 2
    assert instance.semaphore._value == instance.concurrency
    assert callback.call_count == int(fault == "dead")
