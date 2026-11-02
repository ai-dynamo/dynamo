# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine health is independent of optional decision admission capacity."""

import asyncio

import pytest

from dynamo.health_check import HEALTH_CHECK_KEY
from dynamo.runtime import Context
from dynamo.vllm.decision_surface import NativeDecisionSurface, PoolingProbe
from dynamo.vllm.tests.decision_head_test_support import Engine, adapter, wire

pytest_plugins = ["dynamo.vllm.tests.decision_pinned_fixtures"]
pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
]


async def _health(surface, context):
    payload = {**surface.health_payload("test"), HEALTH_CHECK_KEY: True}
    return [row async for row in surface.maybe_handle(payload, context)]


@pytest.mark.asyncio
async def test_health_runs_while_decision_children_hold_every_permit(
    native, monkeypatch
):
    engine = Engine()
    instance = await adapter(engine, max_concurrency=2)
    read = (await instance._prepare(wire())).reads[0]
    surface = NativeDecisionSurface("head", instance, PoolingProbe(read, instance))
    original = engine.encode
    saturated = asyncio.Event()
    blocked, closed = [], []

    async def encode(**kwargs):
        request_id = kwargs["request_id"]
        try:
            if request_id.startswith("blocked-"):
                blocked.append(request_id)
                if len(blocked) == instance.concurrency:
                    saturated.set()
                await asyncio.Event().wait()
            async for output in original(**kwargs):
                yield output
        finally:
            closed.append(request_id)

    monkeypatch.setattr(engine, "encode", encode)
    children = [
        asyncio.create_task(
            instance._read(
                read,
                "isolated",
                Context(),
                f"blocked-{index}",
                lambda: False,
                None,
                lambda final: None,
            )
        )
        for index in range(instance.concurrency)
    ]
    try:
        await asyncio.wait_for(saturated.wait(), 1)
        assert instance.semaphore.locked()
        assert await asyncio.wait_for(_health(surface, Context()), 0.5) == [
            {"status": "ok"}
        ]
        assert instance.semaphore.locked() and all(
            not child.done() for child in children
        )
    finally:
        for child in children:
            child.cancel()
        await asyncio.gather(*children, return_exceptions=True)
    assert sorted(engine.aborted) == sorted(blocked)
    assert set(blocked) <= set(closed)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", ["caller", "context"])
async def test_health_cancellation_aborts_and_closes_its_engine_child(
    native, monkeypatch, cancel
):
    engine = Engine()
    instance = await adapter(engine)
    read = (await instance._prepare(wire())).reads[0]
    surface = NativeDecisionSurface("head", instance, PoolingProbe(read, instance))
    original = engine.encode
    closed = []

    async def encode(**kwargs):
        try:
            async for output in original(**kwargs):
                yield output
        finally:
            closed.append(kwargs["request_id"])

    monkeypatch.setattr(engine, "encode", encode)
    engine.block = asyncio.Event()
    for _ in range(instance.concurrency):
        await instance.semaphore.acquire()
    assert instance.semaphore.locked()
    context = Context()
    task = asyncio.create_task(_health(surface, context))
    await asyncio.wait_for(engine.started.wait(), 1)
    if cancel == "caller":
        task.cancel()
    else:
        context.stop_generating()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert engine.active == 0
    assert engine.aborted == closed == [f"{context.id()}-head-0"]
    assert instance.semaphore.locked()
    for _ in range(instance.concurrency):
        instance.semaphore.release()
