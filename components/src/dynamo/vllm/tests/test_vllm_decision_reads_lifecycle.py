# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python request cancellation, not proof of physical GPU abort settlement."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dynamo.llm.exceptions import (
    HttpError,
    InvalidArgument,
    WorkerOverloaded,
    WorkerUnavailable,
)
from dynamo.vllm import decision_native_lifecycle as native_lifecycle
from dynamo.vllm.decision_native_errors import (
    NativeExecutionError,
    NativeValidationError,
)
from dynamo.vllm.decision_native_lifecycle import run_with_context_cancellation
from dynamo.vllm.decision_native_wire import MAX_NATIVE_QUESTIONS
from dynamo.vllm.tests.decision_head_test_support import DecisionContext
from dynamo.vllm.tests.decision_head_test_support import Engine as HeadEngine
from dynamo.vllm.tests.decision_head_test_support import NativeApi as HeadNativeApi
from dynamo.vllm.tests.decision_head_test_support import adapter as head_adapter
from dynamo.vllm.tests.decision_head_test_support import wire as head_wire
from dynamo.vllm.tests.decision_read_contract_support import adapter as read_adapter
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


async def native_adapter(kind, monkeypatch):
    if kind == "head":
        monkeypatch.setattr(native_lifecycle, "load_native", lambda: HeadNativeApi())
        engine = HeadEngine()
        return await head_adapter(engine), engine, head_wire()
    engine = Engine()
    return await read_adapter(engine), engine, read_wire()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["head", "read"])
@pytest.mark.parametrize("phase", ["prepare", "respond"])
@pytest.mark.parametrize("error_type", [TypeError, ValueError])
async def test_unexpected_adapter_failures_are_sanitized_internal_errors(
    native_reads, monkeypatch, kind, phase, error_type
):
    instance, engine, request = await native_adapter(kind, monkeypatch)
    cause = error_type("private adapter failure")

    def fail(*args):
        raise cause

    async def prepare(*args):
        fail()

    monkeypatch.setattr(
        instance,
        "_prepare" if phase == "prepare" else "_respond",
        prepare if phase == "prepare" else fail,
    )
    with pytest.raises(HttpError) as failure:
        await instance.execute(
            request, DecisionContext(), request_id="failure", cancelled=lambda: False
        )
    assert failure.value.code == 500
    assert failure.value.__cause__ is cause
    assert "private adapter failure" not in failure.value.message
    if phase == "prepare":
        assert engine.admissions == [] and engine.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["head", "read"])
@pytest.mark.parametrize(
    "error_type",
    [
        NativeValidationError,
        NativeExecutionError,
        InvalidArgument,
        WorkerUnavailable,
        WorkerOverloaded,
        asyncio.CancelledError,
    ],
)
async def test_expected_adapter_failures_keep_their_type_and_identity(
    native_reads, monkeypatch, kind, error_type
):
    instance, engine, request = await native_adapter(kind, monkeypatch)
    cause = error_type("expected failure")

    async def prepare(*args):
        raise cause

    monkeypatch.setattr(instance, "_prepare", prepare)
    with pytest.raises(error_type) as failure:
        await instance.execute(
            request, DecisionContext(), request_id="failure", cancelled=lambda: False
        )
    assert failure.value is cause
    assert engine.admissions == [] and engine.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["head", "read"])
async def test_explicit_http_error_status_is_preserved(native_reads, monkeypatch, kind):
    instance, _, request = await native_adapter(kind, monkeypatch)
    cause = HttpError(429, "capacity exceeded")

    async def prepare(*args):
        raise cause

    monkeypatch.setattr(instance, "_prepare", prepare)
    with pytest.raises(HttpError) as failure:
        await instance.execute(
            request, DecisionContext(), request_id="failure", cancelled=lambda: False
        )
    assert failure.value is cause and failure.value.code == 429


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["head", "read"])
async def test_request_token_budget_remains_an_explicit_client_error(
    native_reads, monkeypatch, kind
):
    instance, engine, request = await native_adapter(kind, monkeypatch)
    request[f"decision_{kind}"]["max_input_tokens"] = 1
    with pytest.raises(NativeValidationError) as failure:
        await instance.execute(
            request, DecisionContext(), request_id="budget", cancelled=lambda: False
        )
    assert isinstance(failure.value, InvalidArgument)
    assert engine.admissions == [] and engine.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["head", "read"])
@pytest.mark.parametrize("stop_after", [None, "preflight", "children"])
async def test_base_lifecycle_orders_preflight_admission_children_and_response(
    native_reads, monkeypatch, kind, stop_after
):
    instance, engine, request = await native_adapter(kind, monkeypatch)
    events = []
    prepared = SimpleNamespace(admission_count=2)
    stopped = False

    async def prepare(wire):
        nonlocal stopped
        assert wire is request
        events.append("preflight")
        stopped = stop_after == "preflight"
        return prepared

    def admit(count):
        assert count == prepared.admission_count
        events.append("admission")

    def children(plan, context, **kwargs):
        assert plan is prepared
        events.append("children")

        async def child(index):
            nonlocal stopped
            stopped = stop_after == "children"
            return index

        return (native_lifecycle.NativeChildBatch(2, child),)

    def respond(plan, results):
        assert plan is prepared and results == [0, 1]
        events.append("response")
        return {"complete": True}

    monkeypatch.setattr(instance, "_prepare", prepare)
    monkeypatch.setattr(engine, "check_admission", admit)
    monkeypatch.setattr(instance, "_children", children)
    monkeypatch.setattr(instance, "_respond", respond)
    operation = instance.execute(
        request, DecisionContext(), request_id="order", cancelled=lambda: stopped
    )
    if stop_after is None:
        assert await operation == {"complete": True}
        assert events == ["preflight", "admission", "children", "response"]
    else:
        with pytest.raises(asyncio.CancelledError):
            await operation
        expected = ["preflight"]
        if stop_after == "children":
            expected += ["admission", "children"]
        assert events == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["head", "read"])
@pytest.mark.parametrize("extra", [0, 1])
async def test_native_parent_id_reserves_the_longest_child_suffix(
    native_reads, monkeypatch, kind, extra
):
    instance, engine, request = await native_adapter(kind, monkeypatch)
    longest_suffix = max(len(f"-head-{MAX_NATIVE_QUESTIONS - 1}"), len("-prefix"))
    parent_limit = 256 - longest_suffix
    context = DecisionContext()
    operation = instance.execute(
        request,
        context,
        request_id="r" * (parent_limit + extra),
        cancelled=lambda: False,
    )
    if extra:
        with pytest.raises(NativeExecutionError, match="request ID") as failure:
            await operation
        assert failure.value.code == 502
        assert engine.admissions == [] and engine.calls == []
    else:
        assert await operation
        assert context and all(len(child_id) <= 256 for child_id in context)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["head", "read"])
async def test_context_stop_cancels_semaphore_wait_without_borrowing_slot(
    native_reads, monkeypatch, kind
):
    instance, engine, request = await native_adapter(kind, monkeypatch)
    for _ in range(instance.concurrency):
        await instance.semaphore.acquire()
    prepared = asyncio.Event()
    original = instance._prepare

    async def prepare(wire):
        result = await original(wire)
        prepared.set()
        return result

    monkeypatch.setattr(instance, "_prepare", prepare)
    signal = asyncio.get_running_loop().create_future()
    context = SimpleNamespace(async_killed_or_stopped=lambda: signal)
    stopped = False
    task = asyncio.create_task(
        instance.execute(
            request, context, request_id="queued", cancelled=lambda: stopped
        )
    )
    await prepared.wait()
    stopped = True
    signal.set_result(None)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=0.2)
    assert instance.semaphore._value == 0
    assert engine.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["head", "read"])
async def test_context_stop_interrupts_async_preflight_before_admission(
    native_reads, monkeypatch, kind
):
    instance, engine, request = await native_adapter(kind, monkeypatch)
    entered = asyncio.Event()
    released = asyncio.Event()

    async def prepare(wire):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            released.set()

    monkeypatch.setattr(instance, "_prepare", prepare)
    signal = asyncio.get_running_loop().create_future()
    context = SimpleNamespace(async_killed_or_stopped=lambda: signal, append=Mock())
    task = asyncio.create_task(
        instance.execute(
            request, context, request_id="preflight", cancelled=lambda: False
        )
    )
    await entered.wait()
    signal.set_result(None)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=0.2)
    assert released.is_set()
    assert engine.calls == []
    assert engine.admissions == []


@pytest.mark.asyncio
async def test_waker_failure_is_not_silenced_and_owned_operation_is_drained():
    entered, drained = asyncio.Event(), asyncio.Event()

    async def operation():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            drained.set()

    signal = asyncio.get_running_loop().create_future()
    context = SimpleNamespace(async_killed_or_stopped=lambda: signal)
    task = asyncio.create_task(
        run_with_context_cancellation(operation(), context, lambda: False)
    )
    await entered.wait()
    signal.set_exception(ValueError("broken context waker"))
    with pytest.raises(ValueError, match="broken context waker"):
        await task
    assert drained.is_set()


@pytest.mark.asyncio
async def test_context_without_runtime_waker_fails_instead_of_polling():
    async def operation():
        await asyncio.Event().wait()

    with pytest.raises(AttributeError, match="async_killed_or_stopped"):
        await run_with_context_cancellation(operation(), object(), lambda: False)


@pytest.mark.asyncio
async def test_completed_operation_wins_a_simultaneously_ready_context_signal():
    loop = asyncio.get_running_loop()
    operation, signal = loop.create_future(), loop.create_future()
    operation.set_result("complete")
    signal.set_result(None)
    context = SimpleNamespace(async_killed_or_stopped=lambda: signal)
    assert (
        await run_with_context_cancellation(operation, context, lambda: False)
        == "complete"
    )


@pytest.mark.asyncio
async def test_operation_failure_is_preserved_and_runtime_waker_is_closed():
    async def operation():
        raise ValueError("engine failure")

    signal = asyncio.get_running_loop().create_future()
    context = SimpleNamespace(async_killed_or_stopped=lambda: signal)
    with pytest.raises(ValueError, match="engine failure"):
        await run_with_context_cancellation(operation(), context, lambda: False)
    assert signal.cancelled()
