# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cancellation must stop preprocessing before a request can enter EngineCore."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from dynamo.llm.exceptions import EngineShutdown
from dynamo.vllm.constants import DisaggregationMode
from dynamo.vllm.handlers import DecodeWorkerHandler
from dynamo.vllm.multimodal_utils.request_processor import PreparedMultimodalInput

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.fault_tolerance,
    pytest.mark.asyncio,
    pytest.mark.timeout(10),
]


@pytest.fixture(params=[DisaggregationMode.AGGREGATED, DisaggregationMode.DECODE])
def admission(request):
    stopped = asyncio.Event()
    started = asyncio.Event()
    release = asyncio.Event()
    unwound = asyncio.Event()
    live_requests = set()
    events = []
    context = MagicMock()
    context.id.return_value = "cancel-admission"
    context.is_stopped.side_effect = stopped.is_set
    context.is_killed.return_value = False
    context.async_killed_or_stopped.side_effect = lambda: asyncio.create_task(
        stopped.wait()
    )
    context.trace_headers.return_value = None

    handler = DecodeWorkerHandler.__new__(DecodeWorkerHandler)
    handler.config = SimpleNamespace(disaggregation_mode=request.param, enable_rl=False)
    handler.use_vllm_tokenizer = False
    handler._first_token_source = None
    handler._custom_encoder = None
    handler.shutdown_event = None
    handler.model_max_len = 4096
    handler.default_sampling_params = {}
    handler._resolve_lora_request = lambda _: None
    handler._build_prompt_from_request = lambda *args, **kwargs: {
        "prompt_token_ids": [1, 2]
    }
    handler._to_local_dp_rank = lambda _: None

    async def abort(request_id):
        events.append("abort")
        live_requests.discard(request_id)

    handler.engine_client = SimpleNamespace(abort=AsyncMock(side_effect=abort))

    async def prepare(request, *args):
        return PreparedMultimodalInput(
            request=request, multi_modal_data=None, mm_processor_kwargs=None
        )

    handler._multimodal_request_processor = SimpleNamespace(
        prepare_input=prepare, validate_multimodal_request=lambda request: None
    )

    async def generate(*args, **kwargs):
        events.append("submit")
        live_requests.add(context.id())
        yield {"token_ids": [3], "index": 0}
        yield {"token_ids": [4], "index": 0, "finish_reason": "stop"}
        live_requests.discard(context.id())

    handler.generate_tokens = generate

    async def gate():
        started.set()
        try:
            await release.wait()
        finally:
            events.append("unwind")
            unwound.set()

    return SimpleNamespace(
        handler=handler,
        context=context,
        request={"token_ids": [1, 2], "stop_conditions": {"max_tokens": 2}},
        stopped=stopped,
        started=started,
        release=release,
        unwound=unwound,
        live_requests=live_requests,
        events=events,
        prepare=prepare,
        generate=generate,
        gate=gate,
    )


@pytest.mark.parametrize("stage", ["media", "engine", "registered"])
@pytest.mark.parametrize("cancel_consumer", [False, True])
async def test_cancel_during_admission(admission, stage, cancel_consumer):
    a = admission
    if stage == "media":

        async def prepare(*args):
            await a.gate()
            return await a.prepare(*args)

        a.handler._multimodal_request_processor.prepare_input = prepare
    else:

        async def generate(*args, **kwargs):
            if stage == "registered":
                a.live_requests.add(a.context.id())
            await a.gate()
            async for chunk in a.generate(*args, **kwargs):
                yield chunk

        a.handler.generate_tokens = generate

    async def consume():
        return [chunk async for chunk in a.handler.generate(a.request, a.context)]

    task = asyncio.create_task(consume())
    try:
        await a.started.wait()
        if cancel_consumer:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            a.stopped.set()
            assert await task == []
        assert a.unwound.is_set()
        assert "submit" not in a.events
        assert not a.live_requests
        # A final abort after unwinding reclaims a partially admitted request.
        assert a.events.index("unwind") < len(a.events) - 1
        assert a.events[-1] == "abort"
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_consumer_cancellation_during_context_cancellation_cleanup(admission):
    a = admission
    monitor_started = asyncio.Event()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()

    async def monitor(*args):
        monitor_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            # Model an abort monitor still draining its waiters when admission
            # is cancelled, before it has sent an engine abort.
            cleanup_started.set()
            try:
                await release_cleanup.wait()
            finally:
                a.events.append("monitor_unwound")

    async def generate(*args, **kwargs):
        a.live_requests.add(a.context.id())
        a.started.set()
        await asyncio.Event().wait()
        yield

    a.handler._monitor_abort = monitor
    a.handler.generate_tokens = generate

    async def consume():
        return [chunk async for chunk in a.handler.generate(a.request, a.context)]

    task = asyncio.create_task(consume())
    try:
        await a.started.wait()
        await monitor_started.wait()
        a.stopped.set()
        await cleanup_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not a.live_requests
        assert a.events == ["monitor_unwound", "abort"]
    finally:
        release_cleanup.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize(
    "prefill_result",
    [
        None,
        {"disaggregated_params": None},
        {"disaggregated_params": {"kv_transfer_params": {}}},
    ],
    ids=["no-prefill", "null-params", "empty-kv"],
)
async def test_already_cancelled_request_skips_preprocessing(admission, prefill_result):
    a = admission
    a.request["prefill_result"] = prefill_result
    a.handler._multimodal_request_processor.prepare_input = AsyncMock(
        side_effect=a.prepare
    )
    a.stopped.set()
    assert [chunk async for chunk in a.handler.generate(a.request, a.context)] == []
    a.handler._multimodal_request_processor.prepare_input.assert_not_awaited()
    a.handler.engine_client.abort.assert_not_awaited()


async def test_stream_can_be_polled_by_different_tasks(admission):
    """The Rust bridge starts a fresh Python future for each __anext__."""
    a = admission
    stream = a.handler.generate(a.request, a.context)
    try:
        first = await asyncio.create_task(anext(stream))
        second = await asyncio.create_task(anext(stream))
        assert first["token_ids"] == [3]
        assert second["token_ids"] == [4]
        with pytest.raises(StopAsyncIteration):
            await asyncio.create_task(anext(stream))
        a.handler.engine_client.abort.assert_not_awaited()
        a.context.notify_first_token.assert_called_once()
        assert not a.live_requests
    finally:
        await stream.aclose()


async def test_cancel_during_synchronous_media_preparation(admission):
    a = admission

    async def prepare(*args):
        # The Rust context can become stopped while synchronous media work
        # holds the GIL, before the asyncio cancellation waiter gets to run.
        a.stopped.set()
        return await a.prepare(*args)

    a.handler._multimodal_request_processor.prepare_input = prepare
    assert [chunk async for chunk in a.handler.generate(a.request, a.context)] == []
    assert "submit" not in a.events


@pytest.mark.parametrize("admission", [DisaggregationMode.DECODE], indirect=True)
async def test_cancelled_conditional_decode_skips_preprocessing(admission):
    a = admission
    a.request["annotations"] = ["x-bypass-remote-prefill"]
    a.stopped.set()
    assert [chunk async for chunk in a.handler.generate(a.request, a.context)] == []
    assert "submit" not in a.events


@pytest.mark.parametrize("admission", [DisaggregationMode.DECODE], indirect=True)
@pytest.mark.parametrize(
    ("kv_source", "cancel_at"),
    [
        ("router", "before"),
        ("router", "preparation"),
        ("router", "registered"),
        ("native", "before"),
        ("native", "registered"),
        ("native-sampling", "before"),
        ("native-sampling", "registered"),
    ],
)
async def test_cancelled_decode_admits_staged_kv_for_cleanup(
    admission, kv_source, cancel_at
):
    a = admission
    aborted = asyncio.Event()
    kv_params = {
        "do_remote_prefill": True,
        "do_remote_decode": False,
        "remote_engine_id": "prefill-worker",
        "remote_request_id": "prefill-request",
        "remote_block_ids": [[1, 2]],
        "remote_host": "127.0.0.1",
        "remote_port": 12345,
    }
    if kv_source == "router":
        a.request["prefill_result"] = {
            "disaggregated_params": {"kv_transfer_params": kv_params}
        }
    else:
        envelope = {"request_id": a.context.id(), "sampling_params": {"max_tokens": 2}}
        if kv_source == "native":
            envelope["kv_transfer_params"] = kv_params
        else:
            envelope["sampling_params"]["extra_args"] = {
                "kv_transfer_params": kv_params
            }
        a.request["extra_args"] = {"vllm_tito": envelope}
        a.handler._multimodal_request_processor.enable_multimodal = False
        a.handler.engine_client.vllm_config = SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_seqs=8),
            model_config=SimpleNamespace(max_model_len=4096),
        )

    async def prepare(*args):
        if cancel_at == "preparation":
            a.stopped.set()
        return await a.prepare(*args)

    async def abort(request_id):
        a.events.append("abort")
        a.live_requests.discard(request_id)
        aborted.set()

    async def generate(prompt, sampling_params, *args, **kwargs):
        assert sampling_params.extra_args["kv_transfer_params"] == kv_params
        a.events.append("submit")
        a.live_requests.add(a.context.id())
        if cancel_at == "registered":
            a.stopped.set()
        # The connector must see admission followed by abort, even if no
        # output arrives from the remote KV read.
        await aborted.wait()
        return
        yield

    a.handler._multimodal_request_processor.prepare_input = prepare
    a.handler.engine_client.abort.side_effect = abort
    a.handler.generate_tokens = generate
    if cancel_at == "before":
        a.stopped.set()
    assert [chunk async for chunk in a.handler.generate(a.request, a.context)] == []
    assert a.events == ["submit", "abort"]
    assert not a.live_requests
    a.context.notify_first_token.assert_not_called()


@pytest.mark.parametrize("fail", [False, True], ids=["empty-stream", "error"])
async def test_admission_preserves_empty_stream_and_errors(admission, fail):
    a = admission

    async def generate(*args, **kwargs):
        if fail:
            raise ValueError("invalid input")
        return
        yield  # Make this an async generator without producing output.

    a.handler.generate_tokens = generate
    if fail:
        with pytest.raises(ValueError, match="invalid input"):
            _ = [chunk async for chunk in a.handler.generate(a.request, a.context)]
    else:
        assert [chunk async for chunk in a.handler.generate(a.request, a.context)] == []
    a.handler.engine_client.abort.assert_not_awaited()


async def test_cancel_after_first_output_uses_existing_abort_monitor(admission):
    a = admission
    aborted = asyncio.Event()

    async def abort(request_id):
        a.live_requests.discard(request_id)
        aborted.set()

    async def generate(*args, **kwargs):
        a.live_requests.add(a.context.id())
        yield {"token_ids": [3], "index": 0}
        await aborted.wait()

    a.handler.engine_client.abort.side_effect = abort
    a.handler.generate_tokens = generate
    stream = a.handler.generate(a.request, a.context)
    try:
        assert (await asyncio.create_task(anext(stream)))["token_ids"] == [3]
        a.stopped.set()
        assert await asyncio.create_task(anext(stream, None)) is None
        a.handler.engine_client.abort.assert_awaited_once_with(a.context.id())
        assert not a.live_requests
    finally:
        await stream.aclose()


async def test_engine_shutdown_before_first_output_is_preserved(admission):
    a = admission
    a.handler.shutdown_event = asyncio.Event()
    aborted = asyncio.Event()

    async def abort(request_id):
        aborted.set()

    async def generate(*args, **kwargs):
        a.started.set()
        await aborted.wait()
        return
        yield

    a.handler.engine_client.abort.side_effect = abort
    a.handler.generate_tokens = generate

    async def consume():
        return [chunk async for chunk in a.handler.generate(a.request, a.context)]

    task = asyncio.create_task(consume())
    try:
        await a.started.wait()
        a.handler.shutdown_event.set()
        with pytest.raises(EngineShutdown):
            await task
        a.handler.engine_client.abort.assert_awaited_once_with(a.context.id())
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
