# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from contextlib import aclosing
from types import SimpleNamespace

import pytest

from dynamo.sglang.engine_generate import native_generate_stream

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
    pytest.mark.asyncio,
]


def _engine(responses):
    """Provide a tokenizer manager that returns the retained response iterator."""
    return SimpleNamespace(
        tokenizer_manager=SimpleNamespace(
            generate_request=lambda request, http: responses
        )
    )


async def test_early_close_awaits_inner_cleanup():
    """Closing the outer generator must finish cleanup before returning."""
    closed = False

    async def responses():
        """Keep a suspended generator whose cleanup can be observed."""
        nonlocal closed
        try:
            yield {"output_ids": [1]}
            yield {"output_ids": [2]}
        finally:
            closed = True

    async with aclosing(responses()) as inner:
        outer = native_generate_stream(_engine(inner), None)
        async with aclosing(outer):
            assert (await anext(outer))["token_ids"] == [1]
        assert closed


async def test_normal_completion_preserves_responses():
    """Iterator ownership must not change response envelopes or token IDs."""
    payloads = [{"output_ids": [1]}, {"output_ids": [], "meta_info": {"done": True}}]

    async def responses():
        """Yield native response payloads without modifying them."""
        for payload in payloads:
            yield payload

    outputs = [
        item async for item in native_generate_stream(_engine(responses()), None)
    ]
    assert [item["engine_data"]["sglang_response"] for item in outputs] == payloads
    assert [item["token_ids"] for item in outputs] == [[1], []]


async def test_task_cancellation_closes_inner_iterator():
    """Cancelling a pending read must await the inner generator's cleanup."""
    entered = asyncio.Event()
    closed = False

    async def responses():
        """Signal entry before blocking so cancellation needs no sleep."""
        nonlocal closed
        try:
            entered.set()
            await asyncio.Event().wait()
            yield {}
        finally:
            closed = True

    async with aclosing(responses()) as inner:
        async with aclosing(native_generate_stream(_engine(inner), None)) as outer:
            task = asyncio.create_task(anext(outer))
            try:
                await asyncio.wait_for(entered.wait(), timeout=1)
            finally:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=1)
            assert closed


@pytest.mark.parametrize("early_close", [False, True])
async def test_cleanup_failure_is_reported(early_close):
    """Surface cleanup-only failures on exhaustion and explicit outer close."""

    class Responses:
        def __aiter__(self):
            """Return the owned iterator."""
            return self

        async def __anext__(self):
            """Yield a chunk for early close, or complete immediately."""
            if early_close:
                return {"output_ids": [1]}
            raise StopAsyncIteration

        async def aclose(self):
            """Fail cleanup independently of iteration."""
            raise RuntimeError("cleanup failed")

    outer = native_generate_stream(_engine(Responses()), None)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        if early_close:
            await anext(outer)
            await outer.aclose()
        else:
            await anext(outer)


@pytest.mark.parametrize(
    "failure", [ValueError("engine failed"), asyncio.CancelledError()]
)
async def test_cleanup_failure_does_not_replace_primary_error(failure, caplog):
    """Keep the original failure and log a secondary cleanup failure."""

    class Responses:
        closed = False

        def __aiter__(self):
            """Return the owned iterator."""
            return self

        async def __anext__(self):
            """Raise the original engine error or cancellation."""
            raise failure

        async def aclose(self):
            """Record attempted cleanup before raising a secondary failure."""
            self.closed = True
            raise RuntimeError("cleanup failed")

    inner = Responses()
    with pytest.raises(type(failure)) as caught:
        await anext(native_generate_stream(_engine(inner), None))
    assert caught.value is failure
    assert inner.closed
    assert "Failed to close the native SGLang response iterator" in caplog.text
