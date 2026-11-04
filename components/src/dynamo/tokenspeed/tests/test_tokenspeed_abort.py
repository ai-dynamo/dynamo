# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request lifecycle of TokenspeedLLMEngine.generate()/abort().

The fake tokenizer manager reproduces TokenSpeed's AsyncLLM semantics:
``abort_request`` deletes the request state and sends AbortReq without waking
the coroutine waiting in ``generate_request``, and cancelling that coroutine
sends AbortReq from its cleanup.
"""

import asyncio
from types import SimpleNamespace

import pytest

from dynamo.tokenspeed import llm_engine
from dynamo.tokenspeed.llm_engine import TokenspeedLLMEngine

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]

TIMEOUT_S = 2.0


class FakeTokenizerManager:
    def __init__(self, outputs=None, fail_after=None):
        # outputs=None: stream tokens forever, one per 10 ms.
        self.outputs = outputs
        self.fail_after = fail_after
        self.rid_to_state = {}
        self.aborts_sent = []

    async def generate_request(self, obj):
        self.rid_to_state[obj.rid] = SimpleNamespace(finished=False)
        try:
            for i in range(10**9 if self.outputs is None else len(self.outputs)):
                if self.fail_after is not None and i == self.fail_after:
                    raise RuntimeError("native failure")
                if obj.rid not in self.rid_to_state:
                    # Aborted: AsyncLLM drops later outputs for this rid and
                    # nothing wakes the waiter again.
                    await asyncio.Event().wait()
                await asyncio.sleep(0.01)
                if self.outputs is None:
                    yield {"output_ids": [i], "meta_info": {"completion_tokens": i + 1}}
                else:
                    yield self.outputs[i]
            self.rid_to_state.pop(obj.rid, None)
        finally:
            if obj.rid in self.rid_to_state:
                self.abort_request(obj.rid)

    def abort_request(self, rid):
        if rid not in self.rid_to_state:
            return
        del self.rid_to_state[rid]
        self.aborts_sent.append(rid)


class PendingStepTokenizerManager(FakeTokenizerManager):
    """Yields one token, then waits for an output that never comes.

    ``entered_wait`` is set once the next native step is pending. Nothing
    ``abort_request`` does wakes that wait, as in AsyncLLM.
    """

    def __init__(self):
        super().__init__()
        self.entered_wait = asyncio.Event()

    async def generate_request(self, obj):
        self.rid_to_state[obj.rid] = SimpleNamespace(finished=False)
        try:
            yield {"output_ids": [0], "meta_info": {"completion_tokens": 1}}
            self.entered_wait.set()
            await asyncio.Event().wait()
        finally:
            if obj.rid in self.rid_to_state:
                self.abort_request(obj.rid)


class FakeContext:
    def __init__(self, request_id="req-1"):
        self._id = request_id

    def id(self):
        return self._id


class FakeGenerateReqInput:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.rid = None


@pytest.fixture(autouse=True)
def _fake_req_input(monkeypatch):
    monkeypatch.setattr(
        llm_engine, "_generate_req_input_cls", lambda: FakeGenerateReqInput
    )


def make_engine(tokenizer_manager):
    engine = TokenspeedLLMEngine(SimpleNamespace(disaggregation_mode=None))
    engine.engine = SimpleNamespace(tokenizer_manager=tokenizer_manager)
    return engine


REQUEST = {"token_ids": [1, 2, 3], "stop_conditions": {"max_tokens": 4}}


def finished(reason, prompt_tokens, completion_tokens):
    return {
        "output_ids": [],
        "meta_info": {
            "finish_reason": {"type": reason},
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }


@pytest.mark.asyncio
async def test_generate_completes_without_abort():
    tm = FakeTokenizerManager(
        outputs=[
            {"output_ids": [7], "meta_info": {"completion_tokens": 1}},
            {
                "output_ids": [8],
                "meta_info": {
                    "finish_reason": {"type": "stop"},
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                },
            },
        ]
    )
    engine = make_engine(tm)

    chunks = [c async for c in engine.generate(dict(REQUEST), FakeContext())]

    assert [c["token_ids"] for c in chunks] == [[7], [8]]
    assert chunks[-1]["finish_reason"] == "stop"
    assert chunks[-1]["completion_usage"]["total_tokens"] == 5
    assert tm.aborts_sent == []
    assert engine._native_steps == {}
    assert engine._active_rids_by_context == {}


@pytest.mark.asyncio
async def test_abort_between_chunks_ends_stream_with_cancelled_terminal():
    tm = FakeTokenizerManager()
    engine = make_engine(tm)
    context = FakeContext()
    stream = engine.generate(dict(REQUEST), context)

    first = await asyncio.wait_for(stream.__anext__(), TIMEOUT_S)
    assert first["token_ids"] == [0]
    await engine.abort(context)

    async def drain():
        return [c async for c in stream]

    rest = await asyncio.wait_for(drain(), TIMEOUT_S)

    assert rest[-1] == {"token_ids": [], "finish_reason": "cancelled"}
    assert all("finish_reason" not in c for c in rest[:-1])
    assert tm.aborts_sent == ["req-1"]
    assert tm.rid_to_state == {}
    assert engine._native_steps == {}
    assert engine._active_rids_by_context == {}
    assert engine._cancelled_contexts == set()


@pytest.mark.asyncio
async def test_abort_wakes_a_pending_native_step():
    tm = PendingStepTokenizerManager()
    engine = make_engine(tm)
    context = FakeContext()
    stream = engine.generate(dict(REQUEST), context)
    assert (await asyncio.wait_for(stream.__anext__(), TIMEOUT_S))["token_ids"] == [0]

    pending = asyncio.create_task(stream.__anext__())
    await asyncio.wait_for(tm.entered_wait.wait(), TIMEOUT_S)
    await engine.abort(context)

    assert await asyncio.wait_for(pending, TIMEOUT_S) == {
        "token_ids": [],
        "finish_reason": "cancelled",
    }
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(stream.__anext__(), TIMEOUT_S)
    assert tm.aborts_sent == ["req-1"]
    assert engine._native_steps == {}
    assert engine._active_rids_by_context == {}
    assert engine._cancelled_contexts == set()


@pytest.mark.asyncio
async def test_cancelling_the_consumer_during_a_native_step_propagates():
    tm = PendingStepTokenizerManager()
    engine = make_engine(tm)
    stream = engine.generate(dict(REQUEST), FakeContext())
    await asyncio.wait_for(stream.__anext__(), TIMEOUT_S)

    pending = asyncio.create_task(stream.__anext__())
    await asyncio.wait_for(tm.entered_wait.wait(), TIMEOUT_S)
    pending.cancel()

    done, _ = await asyncio.wait({pending}, timeout=TIMEOUT_S)
    # The caller's cancellation propagates instead of becoming a terminal chunk.
    assert pending in done and pending.cancelled()
    assert tm.aborts_sent == ["req-1"]
    assert tm.rid_to_state == {}
    assert engine._native_steps == {}
    assert engine._active_rids_by_context == {}


@pytest.mark.asyncio
async def test_abort_of_many_requests_leaves_no_state():
    tm = FakeTokenizerManager()
    engine = make_engine(tm)
    contexts = [FakeContext(f"req-{i}") for i in range(32)]
    streams = [engine.generate(dict(REQUEST), ctx) for ctx in contexts]
    await asyncio.gather(*(s.__anext__() for s in streams))

    await asyncio.gather(*(engine.abort(ctx) for ctx in contexts))

    async def drain(stream):
        return [c async for c in stream]

    results = await asyncio.wait_for(
        asyncio.gather(*(drain(s) for s in streams)), TIMEOUT_S
    )
    assert all(r[-1]["finish_reason"] == "cancelled" for r in results)
    assert sorted(tm.aborts_sent) == sorted(ctx.id() for ctx in contexts)
    assert engine._native_steps == {}


@pytest.mark.asyncio
async def test_dropped_stream_aborts_native_request():
    tm = FakeTokenizerManager()
    engine = make_engine(tm)
    stream = engine.generate(dict(REQUEST), FakeContext())
    await asyncio.wait_for(stream.__anext__(), TIMEOUT_S)

    # Stream dropped without engine.abort(), e.g. the response channel closed.
    await stream.aclose()
    await asyncio.sleep(0.05)

    assert tm.aborts_sent == ["req-1"]
    assert tm.rid_to_state == {}
    assert engine._native_steps == {}


@pytest.mark.asyncio
async def test_abort_before_submission_never_submits():
    tm = FakeTokenizerManager()
    engine = make_engine(tm)
    context = FakeContext()
    stream = engine.generate(dict(REQUEST), context)
    # Register the request, then abort before the native submission point.
    engine._active_rids_by_context[context.id()] = [context.id()]
    await engine.abort(context)

    chunk = await asyncio.wait_for(stream.__anext__(), TIMEOUT_S)

    assert chunk["finish_reason"] == "cancelled"
    assert tm.aborts_sent == []


@pytest.mark.asyncio
async def test_native_error_propagates():
    tm = FakeTokenizerManager(
        outputs=[{"output_ids": [7], "meta_info": {"completion_tokens": 1}}] * 3,
        fail_after=1,
    )
    engine = make_engine(tm)
    stream = engine.generate(dict(REQUEST), FakeContext())

    assert (await stream.__anext__())["token_ids"] == [7]
    with pytest.raises(RuntimeError, match="native failure"):
        await asyncio.wait_for(stream.__anext__(), TIMEOUT_S)
    assert engine._native_steps == {}


@pytest.mark.asyncio
async def test_native_abort_finish_reason_maps_to_cancelled():
    # Scheduler-side abort (e.g. pause_scheduler, request too long) arrives
    # as a normal finished output.
    tm = FakeTokenizerManager(outputs=[finished("abort", 3, 0)])
    engine = make_engine(tm)

    chunks = [c async for c in engine.generate(dict(REQUEST), FakeContext())]

    assert chunks[-1]["finish_reason"] == "cancelled"
