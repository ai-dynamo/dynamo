# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python request cancellation, not proof of physical GPU abort settlement."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dynamo.vllm import decision_heads as heads
from dynamo.vllm.decision_lifecycle import run_with_context_cancellation
from dynamo.vllm.tests.test_vllm_decision_heads import Engine as HeadEngine
from dynamo.vllm.tests.test_vllm_decision_heads import NativeApi as HeadNativeApi
from dynamo.vllm.tests.test_vllm_decision_heads import adapter as head_adapter
from dynamo.vllm.tests.test_vllm_decision_heads import wire as head_wire
from dynamo.vllm.tests.test_vllm_decision_reads import adapter as read_adapter
from dynamo.vllm.tests.test_vllm_decision_reads import wire as read_wire
from dynamo.vllm.tests.test_vllm_decision_reads_native import Engine

pytest_plugins = ["dynamo.vllm.tests.test_vllm_decision_reads_native"]

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.timeout(30),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["head", "read"])
async def test_context_stop_cancels_semaphore_wait_without_borrowing_slot(
    native_reads, monkeypatch, kind
):
    if kind == "head":
        monkeypatch.setattr(heads, "_load_native", lambda _: HeadNativeApi())
        engine = HeadEngine()
        instance = await head_adapter(engine)
        request = head_wire()
    else:
        engine = Engine()
        instance = await read_adapter(engine)
        request = read_wire()
    for _ in range(instance.concurrency):
        await instance.semaphore.acquire()
    prepared = asyncio.Event()
    original = instance.prepare
    if kind == "head":

        def prepare(wire):
            result = original(wire)
            prepared.set()
            return result

    else:

        async def prepare(wire):
            result = await original(wire)
            prepared.set()
            return result

    monkeypatch.setattr(instance, "prepare", prepare)
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
async def test_context_stop_interrupts_async_preflight_before_admission(
    native_reads, monkeypatch
):
    engine = Engine()
    instance = await read_adapter(engine)
    entered = asyncio.Event()
    released = asyncio.Event()

    async def prepare(wire):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            released.set()

    monkeypatch.setattr(instance, "prepare", prepare)
    signal = asyncio.get_running_loop().create_future()
    context = SimpleNamespace(async_killed_or_stopped=lambda: signal, append=Mock())
    task = asyncio.create_task(
        instance.execute(
            read_wire(), context, request_id="preflight", cancelled=lambda: False
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
async def test_operation_failure_is_preserved_and_runtime_waker_is_closed():
    async def operation():
        raise ValueError("engine failure")

    signal = asyncio.get_running_loop().create_future()
    context = SimpleNamespace(async_killed_or_stopped=lambda: signal)
    with pytest.raises(ValueError, match="engine failure"):
        await run_with_context_cancellation(operation(), context, lambda: False)
    assert signal.cancelled()
