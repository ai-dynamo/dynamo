# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E402 - optional vLLM dependency is checked before Dynamo imports

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("vllm.lora.request")
pytest.importorskip("zstandard")

from dynamo.artifacts.vllm import capture as capture_mod
from dynamo.common.lora.runtime import ResolvedLoRA
from dynamo.llm import HttpError
from dynamo.vllm import handlers as handlers_mod
from dynamo.vllm.multimodal_utils.request_processor import PreparedMultimodalInput
from dynamo.vllm.tests.test_vllm_delta_streaming import _output, _request_output
from dynamo.vllm.tests.test_vllm_runtime_lora import _coordinator, _request, _snapshot

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class _Context:
    def __init__(self):
        self.killed = asyncio.Event()

    def trace_headers(self):
        return None

    def async_killed_or_stopped(self):
        return asyncio.create_task(self.killed.wait())


def _token_handler(state_handler, coordinator, request):
    handler = handlers_mod.DecodeWorkerHandler.__new__(handlers_mod.DecodeWorkerHandler)
    handler.__dict__.update(vars(state_handler))
    handler._runtime_lora_coordinator = coordinator
    handler._custom_encoder = None
    handler._deferred_aborts = {}
    handler.shutdown_event = asyncio.Event()
    handler.default_sampling_params = {}
    handler.model_max_len = 128
    handler.dp_range = (0, 1)
    handler.artifact_router_profile = ()
    handler.config.enable_rl = False
    handler.runtime = SimpleNamespace(shutdown=lambda: None)
    handler._multimodal_request_processor = SimpleNamespace(
        prepare_input=AsyncMock(
            return_value=PreparedMultimodalInput(
                request=request, multi_modal_data=None, mm_processor_kwargs=None
            )
        )
    )
    handler._build_prompt_from_request = MagicMock(
        return_value={"prompt_token_ids": request["token_ids"]}
    )
    handler._log_with_lora_context = MagicMock()
    handler.engine_client.abort = AsyncMock()
    return handler


def _token_request():
    return {
        **_request("wandb-artifact:///entity/project/adapter:v1"),
        "token_ids": [10, 11],
        "sampling_options": {},
        "stop_conditions": {"max_tokens": 2},
        "output_options": {},
    }


@pytest.mark.parametrize("signal", ["shutdown", "kill"])
@pytest.mark.asyncio
async def test_token_resolution_observes_shutdown_and_client_cancellation(
    monkeypatch, tmp_path, signal
):
    started = asyncio.Event()

    async def resolve(**_kwargs):
        started.set()
        await asyncio.Event().wait()

    state_handler, coordinator = _coordinator(monkeypatch, tmp_path, resolve)
    request = _token_request()
    handler = _token_handler(state_handler, coordinator, request)
    context = _Context()
    stream = handler._generate_token_mode(request, context, "request")
    waiter = asyncio.create_task(anext(stream))
    await asyncio.wait_for(started.wait(), timeout=2)
    try:
        if signal == "shutdown":
            handler.shutdown_event.set()
            expected = handlers_mod.EngineShutdown
        else:
            context.killed.set()
            expected = asyncio.CancelledError
        with pytest.raises(expected):
            await asyncio.wait_for(waiter, timeout=1)
        assert not state_handler._lora_state.runtime_pending_admissions
        assert not state_handler._lora_state.runtime_pending_leases
        assert len(state_handler._lora_state.runtime_load_tasks) == 1
    finally:
        if not waiter.done():
            waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
        await stream.aclose()
        await handler.aclose()
    assert not state_handler._lora_state.runtime_load_tasks


@pytest.mark.asyncio
async def test_token_path_holds_active_runtime_lease_during_stream(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("DYN_LORA_MAX_RESIDENT_RUNTIME_LORAS", "1")
    snapshot = _snapshot(tmp_path)
    state_handler, coordinator = _coordinator(
        monkeypatch, tmp_path, ResolvedLoRA(snapshot, "digest:immutable")
    )
    request = _token_request()
    handler = _token_handler(state_handler, coordinator, request)
    finish = asyncio.Event()
    received = []

    def generate(*_args, **kwargs):
        received.append(kwargs["lora_request"])

        async def responses():
            yield _request_output([_output([1])], prompt_token_ids=request["token_ids"])
            await finish.wait()
            yield _request_output(
                [_output([2], finish_reason="length")],
                prompt_token_ids=request["token_ids"],
            )

        return responses()

    handler.engine_client.generate = generate
    stream = handler._generate_token_mode(request, _Context(), "request")
    first = await anext(stream)
    try:
        key = request["routing"]["lora_name"]
        assert first["token_ids"] == [1]
        assert received[0].lora_name == key
        assert (
            received[0].lora_int_id == state_handler._lora_state.runtime_loras[key].id
        )
        assert state_handler._lora_state.runtime_active_leases == {key: 1}
        assert not state_handler._lora_state.runtime_pending_admissions
        with pytest.raises(HttpError, match="lora_capacity_exceeded"):
            await coordinator.ensure_from_request(
                _request("wandb-artifact:///other/adapter:v1"), "competing"
            )
        state_handler.engine_client.remove_lora.assert_not_awaited()
    finally:
        finish.set()
        remaining = [chunk async for chunk in stream]
        await handler.aclose()
    assert remaining[-1]["finish_reason"] == "length"
    assert not state_handler._lora_state.runtime_pending_leases
    assert not state_handler._lora_state.runtime_active_leases


@pytest.mark.asyncio
async def test_artifact_budget_is_not_reserved_during_adapter_resolution(
    monkeypatch, tmp_path
):
    snapshot = _snapshot(tmp_path)
    started = asyncio.Event()
    finish = asyncio.Event()

    async def resolve(**_kwargs):
        started.set()
        await finish.wait()
        return ResolvedLoRA(snapshot, "digest:immutable")

    state_handler, coordinator = _coordinator(monkeypatch, tmp_path, resolve)
    request = {
        **_token_request(),
        "nvext": {
            "metadata_upload": {
                "kind": "generation_artifact",
                "contents": [],
                "mode": "inline_base64",
            }
        },
    }
    handler = _token_handler(state_handler, coordinator, request)
    budget = capture_mod.CaptureBudget(1 << 20)
    monkeypatch.setattr(capture_mod, "process_capture_budget", lambda: budget)

    async def responses(*_args, **_kwargs):
        yield _request_output(
            [_output([1], finish_reason="length")],
            prompt_token_ids=request["token_ids"],
        )

    handler.engine_client.generate = responses

    async def collect():
        return [
            chunk
            async for chunk in handler._generate_token_mode(
                request, _Context(), "request"
            )
        ]

    waiter = asyncio.create_task(collect())
    await asyncio.wait_for(started.wait(), timeout=2)
    try:
        assert budget.used == 0
    finally:
        finish.set()
        chunks = await waiter
        await handler.aclose()
    assert chunks[-1]["engine_data"]["metadata_upload"]["mode"] == "inline_base64"
    assert budget.used == 0


@pytest.mark.asyncio
async def test_owned_pending_admission_survives_waiting_on_engine_pause_lock(
    monkeypatch, tmp_path
):
    snapshot = _snapshot(tmp_path)
    state_handler, coordinator = _coordinator(
        monkeypatch, tmp_path, ResolvedLoRA(snapshot, "digest:immutable")
    )
    request = _token_request()
    handler = _token_handler(state_handler, coordinator, request)
    await coordinator.ensure_from_request(request, "prime")
    await coordinator.release_pending_admission("prime")
    coordinator._settings = replace(
        coordinator.settings, pending_admission_timeout_seconds=0
    )
    recorded = asyncio.Event()
    original_record = coordinator._record_pending_admission_locked

    def record(*args, **kwargs):
        original_record(*args, **kwargs)
        recorded.set()

    monkeypatch.setattr(coordinator, "_record_pending_admission_locked", record)

    async def responses(*_args, **_kwargs):
        yield _request_output(
            [_output([1], finish_reason="length")],
            prompt_token_ids=request["token_ids"],
        )

    handler.engine_client.generate = responses
    await handler._pause_lock.acquire()
    pause_owned = True
    stream = handler._generate_token_mode(request, _Context(), "request")
    waiter = asyncio.create_task(anext(stream))
    try:
        await asyncio.wait_for(recorded.wait(), timeout=2)
        for _ in range(5):
            await asyncio.sleep(0)
        key = request["routing"]["lora_name"]
        assert state_handler._lora_state.runtime_pending_leases == {key: 1}
        assert not waiter.done()
        handler._pause_lock.release()
        pause_owned = False
        first = await asyncio.wait_for(waiter, timeout=1)
        assert first["token_ids"] == [1]
        assert state_handler._lora_state.runtime_active_leases == {key: 1}
        assert not state_handler._lora_state.runtime_pending_admissions
        assert [chunk async for chunk in stream] == []
        assert not state_handler._lora_state.runtime_active_leases
    finally:
        if pause_owned:
            handler._pause_lock.release()
        if not waiter.done():
            waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
        await stream.aclose()
        await handler.aclose()
