# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import signal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from dynamo.sglang.request_handlers.llm.prefill_handler import PrefillWorkerHandler
from dynamo.sglang.request_handlers.multimodal.worker_handler import (
    MultimodalPrefillWorkerHandler,
)
from dynamo.sglang.request_handlers.prefill_drain import PrefillResultDrain

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]


@pytest.mark.asyncio
async def test_prefill_result_drain_waits_for_pending_transfer():
    result_drain = PrefillResultDrain()
    transfer_terminal = asyncio.Event()

    async def consume_until_terminal():
        await transfer_terminal.wait()

    result_drain.create_task(consume_until_terminal())
    drain_task = asyncio.create_task(result_drain.drain())
    await asyncio.sleep(0)

    assert result_drain.pending_count == 1
    assert not drain_task.done()

    transfer_terminal.set()
    await drain_task
    assert result_drain.pending_count == 0


@pytest.mark.asyncio
async def test_prefill_result_drain_timeout_does_not_cancel_transfer():
    result_drain = PrefillResultDrain()
    transfer_terminal = asyncio.Event()

    async def consume_until_terminal():
        await transfer_terminal.wait()

    transfer_task = result_drain.create_task(consume_until_terminal())

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(result_drain.drain(), timeout=0.01)

    assert not transfer_task.done()
    assert result_drain.pending_count == 1

    result_drain.cancel()
    with pytest.raises(asyncio.CancelledError):
        await transfer_task


@pytest.mark.asyncio
async def test_regular_prefill_registers_transfer_before_bootstrap_yield(monkeypatch):
    transfer_terminal = asyncio.Event()
    result_drain = PrefillResultDrain()
    handler = PrefillWorkerHandler.__new__(PrefillWorkerHandler)
    handler.bootstrap_host = "127.0.0.1"
    handler.bootstrap_port = 1234
    handler._result_drain = result_drain
    handler._generate_bootstrap_room = lambda: 42
    handler._get_input_param = lambda request: {"input_ids": request["token_ids"]}
    handler._resolve_lora = lambda request: None
    handler._priority_kwargs = lambda priority: {}
    handler.enable_trace = False
    handler.engine = SimpleNamespace(async_generate=AsyncMock(return_value=object()))

    async def consume_results(results, context):
        await transfer_terminal.wait()

    handler._consume_results = consume_results
    monkeypatch.setattr(
        "dynamo.sglang.request_handlers.llm.prefill_handler.require_reasoning_kwargs",
        lambda engine, request: {},
    )

    context = SimpleNamespace(
        id=lambda: "request-id",
        trace_id="trace-id",
        trace_headers=lambda: {},
    )
    stream = handler.generate({"request": {"token_ids": [1, 2, 3]}}, context)

    bootstrap = await anext(stream)
    assert bootstrap["disaggregated_params"]["bootstrap_room"] == 42
    assert result_drain.pending_count == 1

    drain_task = asyncio.create_task(handler.drain())
    await asyncio.sleep(0)
    assert not drain_task.done()

    transfer_terminal.set()
    await drain_task
    with pytest.raises(StopAsyncIteration):
        await anext(stream)


@pytest.mark.asyncio
async def test_multimodal_prefill_registers_work_before_bootstrap_yield(monkeypatch):
    from dynamo.sglang.request_handlers.multimodal import worker_handler

    transfer_terminal = asyncio.Event()
    result_drain = PrefillResultDrain()
    handler = MultimodalPrefillWorkerHandler.__new__(MultimodalPrefillWorkerHandler)
    handler.bootstrap_host = "127.0.0.1"
    handler.bootstrap_port = 1234
    handler._result_drain = result_drain
    handler._generate_bootstrap_room = lambda: 42
    handler._validate_and_parse_disagg_request = lambda request: request

    async def process_prefill(disagg_request, bootstrap_room, context=None):
        await transfer_terminal.wait()

    handler._process_prefill_generation = process_prefill
    monkeypatch.setattr(worker_handler._nvtx, "start_range", lambda *args, **kwargs: 1)
    monkeypatch.setattr(worker_handler._nvtx, "end_range", lambda *args, **kwargs: None)

    stream = handler.generate(object(), SimpleNamespace())
    bootstrap = json.loads(await anext(stream))

    assert bootstrap["bootstrap_room"] == 42
    assert result_drain.pending_count == 1

    drain_task = asyncio.create_task(handler.drain())
    await asyncio.sleep(0)
    assert not drain_task.done()

    transfer_terminal.set()
    await drain_task
    with pytest.raises(StopAsyncIteration):
        await anext(stream)


@pytest.mark.asyncio
async def test_multimodal_prefill_work_reaches_engine_stream_terminal(monkeypatch):
    from dynamo.sglang.request_handlers.multimodal import worker_handler

    transfer_terminal = asyncio.Event()

    async def result_stream():
        await transfer_terminal.wait()
        if False:
            yield None

    handler = MultimodalPrefillWorkerHandler.__new__(MultimodalPrefillWorkerHandler)
    handler.bootstrap_host = "127.0.0.1"
    handler.bootstrap_port = 1234
    handler.enable_trace = False
    handler.embeddings_processor = object()
    handler.engine = SimpleNamespace(
        async_generate=AsyncMock(return_value=result_stream())
    )
    monkeypatch.setattr(
        worker_handler,
        "_build_mm_items",
        AsyncMock(return_value=(None, None, None, None)),
    )

    request = SimpleNamespace(
        request=SimpleNamespace(request=SimpleNamespace(token_ids=[1, 2, 3])),
        sampling_params={},
    )
    work_task = asyncio.create_task(
        handler._process_prefill_generation(request, 42, context=None)
    )
    await asyncio.sleep(0)

    assert not work_task.done()
    transfer_terminal.set()
    await work_task


@pytest.mark.asyncio
async def test_multimodal_prefill_cancellation_releases_loaded_embeddings(monkeypatch):
    from dynamo.sglang.request_handlers.multimodal import worker_handler

    engine_started = asyncio.Event()

    async def start_generation(**kwargs):
        engine_started.set()
        await asyncio.Event().wait()

    embeddings_processor = SimpleNamespace(release_embeddings=MagicMock())
    handler = MultimodalPrefillWorkerHandler.__new__(MultimodalPrefillWorkerHandler)
    handler.bootstrap_host = "127.0.0.1"
    handler.bootstrap_port = 1234
    handler.enable_trace = False
    handler.embeddings_processor = embeddings_processor
    handler.engine = SimpleNamespace(async_generate=start_generation)
    monkeypatch.setattr(
        worker_handler,
        "_build_mm_items",
        AsyncMock(return_value=(None, None, None, 7)),
    )

    request = SimpleNamespace(
        request=SimpleNamespace(request=SimpleNamespace(token_ids=[1, 2, 3])),
        sampling_params={},
    )
    work_task = asyncio.create_task(
        handler._process_prefill_generation(request, 42, context=None)
    )
    await asyncio.wait_for(engine_started.wait(), timeout=1)

    work_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await work_task

    embeddings_processor.release_embeddings.assert_called_once_with(7)


@pytest.mark.asyncio
async def test_shutdown_uses_callback_registered_after_signal_install(monkeypatch):
    from dynamo.sglang import shutdown as shutdown_module

    captured_handlers = {}
    shutdown_complete = asyncio.Event()
    call_order = []
    drain_callbacks = []

    loop = SimpleNamespace(
        add_signal_handler=MagicMock(),
        call_soon_threadsafe=lambda callback: callback(),
    )
    monkeypatch.setattr(
        shutdown_module.signal,
        "signal",
        lambda sig, callback: captured_handlers.__setitem__(sig, callback),
    )

    async def graceful_shutdown(runtime, endpoints, **kwargs):
        call_order.append("shutdown")
        await kwargs["drain_callback"]()
        shutdown_complete.set()

    monkeypatch.setattr(
        shutdown_module, "graceful_shutdown_with_discovery", graceful_shutdown
    )

    shutdown_module.install_graceful_shutdown(
        loop,
        object(),
        [],
        asyncio.Event(),
        drain_callbacks=drain_callbacks,
        signals=(signal.SIGTERM,),
    )

    async def drain():
        call_order.append("drain")

    drain_callbacks.append(drain)
    captured_handlers[signal.SIGTERM](signal.SIGTERM, None)
    await asyncio.wait_for(shutdown_complete.wait(), timeout=1)

    assert call_order == ["shutdown", "drain"]
