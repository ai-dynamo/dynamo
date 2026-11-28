# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from contextlib import aclosing, suppress
from types import SimpleNamespace

import pytest
from sglang.srt.managers.io_struct import GenerateReqInput

from dynamo.sglang.engine_generate import native_generate_stream

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
    pytest.mark.asyncio,
    pytest.mark.timeout(5),
]


def _stream_for(responses):
    request = GenerateReqInput(input_ids=[1], rid="owned", stream=True)

    class Manager:
        def generate_request(self, native_request, http_request):
            assert native_request is request
            assert http_request is None
            return responses

    return native_generate_stream(SimpleNamespace(tokenizer_manager=Manager()), request)


class _Responses:
    def __init__(self, responses=(), *, engine_error=None, cleanup_error=None):
        self.responses = iter(responses)
        self.engine_error = engine_error
        self.cleanup_error = cleanup_error
        self.close_calls = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.engine_error is not None:
            raise self.engine_error
        try:
            return next(self.responses)
        except StopIteration:
            raise StopAsyncIteration from None

    async def aclose(self):
        self.close_calls += 1
        if self.cleanup_error is not None:
            raise self.cleanup_error


async def test_outer_close_awaits_strongly_held_inner_generator_finally():
    state = {"closed": False}

    async def responses():
        try:
            yield {"output_ids": [7]}
            await asyncio.Event().wait()
        finally:
            state["closed"] = True

    async with aclosing(responses()) as inner:
        outer = _stream_for(inner)
        await anext(outer)
        assert not state["closed"]
        await outer.aclose()
        assert state["closed"]


async def test_normal_completion_closes_once_and_preserves_opaque_responses():
    messages = [{"output_ids": [7], "meta_info": {"id": "owned"}}, {}]
    inner = _Responses(messages)
    outer = _stream_for(inner)
    first = await anext(outer)
    assert inner.close_calls == 0
    second = await anext(outer)
    assert inner.close_calls == 0
    with pytest.raises(StopAsyncIteration):
        await anext(outer)
    assert inner.close_calls == 1
    assert first["token_ids"] == [7]
    assert second["token_ids"] == []
    assert first["engine_data"]["sglang_response"] is messages[0]
    assert second["engine_data"]["sglang_response"] is messages[1]


@pytest.mark.parametrize("error_type", [ValueError, asyncio.CancelledError])
async def test_engine_failure_stays_primary_when_inner_cleanup_also_fails(error_type):
    primary = error_type("engine failure")
    inner = _Responses(
        engine_error=primary, cleanup_error=RuntimeError("cleanup failure")
    )
    with pytest.raises(error_type) as caught:
        await anext(_stream_for(inner))
    assert caught.value is primary
    assert inner.close_calls == 1
    assert primary.__notes__ == [
        "SGLang native iterator cleanup also failed: RuntimeError"
    ]


@pytest.mark.parametrize("broken_sink", ["override", "malformed_notes"])
async def test_engine_failure_stays_primary_when_cleanup_note_cannot_be_added(
    broken_sink,
):
    class BrokenNoteError(ValueError):
        def add_note(self, note):
            raise RuntimeError("note sink failure")

    if broken_sink == "override":
        primary = BrokenNoteError("engine failure")
    else:
        primary = ValueError("engine failure")
        primary.__notes__ = 42
    inner = _Responses(
        engine_error=primary, cleanup_error=RuntimeError("cleanup failure")
    )
    with pytest.raises(ValueError) as caught:
        await anext(_stream_for(inner))
    assert caught.value is primary
    assert inner.close_calls == 1


async def test_task_cancellation_closes_inner_and_preserves_the_cancelled_error():
    started = asyncio.Event()

    class WaitingResponses(_Responses):
        async def __anext__(self):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as error:
                self.cancelled_error = error
                raise

    inner = WaitingResponses(cleanup_error=RuntimeError("cleanup failure"))

    async def pull():
        return await anext(_stream_for(inner))

    task = asyncio.create_task(pull())
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)
        assert task.cancelled()
        assert inner.close_calls == 1
        assert inner.cancelled_error.__notes__ == [
            "SGLang native iterator cleanup also failed: RuntimeError"
        ]
    finally:
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


async def test_cleanup_failure_after_normal_completion_is_not_hidden():
    primary = RuntimeError("cleanup failure")
    inner = _Responses(cleanup_error=primary)
    with pytest.raises(RuntimeError) as caught:
        await anext(_stream_for(inner))
    assert caught.value is primary
    assert inner.close_calls == 1


async def test_cleanup_failure_during_explicit_outer_close_is_not_generator_exit():
    primary = RuntimeError("cleanup failure")
    inner = _Responses([{"output_ids": [7]}], cleanup_error=primary)
    outer = _stream_for(inner)
    await anext(outer)
    with pytest.raises(RuntimeError) as caught:
        await outer.aclose()
    assert caught.value is primary
    assert inner.close_calls == 1
