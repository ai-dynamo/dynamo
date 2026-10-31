# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dynamo.llm import HttpError
from dynamo.llm.exceptions import InvalidArgument
from dynamo.vllm import transcription_handler
from dynamo.vllm.transcription_handler import TranscriptionWorkerHandler

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.multimodal,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def make_handler() -> TranscriptionWorkerHandler:
    handler = TranscriptionWorkerHandler.__new__(TranscriptionWorkerHandler)
    handler.config = SimpleNamespace(
        model="openai/whisper-tiny",
        served_model_name="whisper",
        served_model_aliases=[],
    )
    return handler


def test_handler_registers_served_model_aliases(monkeypatch):
    registered = {}

    class FakeModels:
        def __init__(self, _engine, base_model_paths):
            registered["base_model_paths"] = base_model_paths

    monkeypatch.setattr(transcription_handler, "OpenAIServingModels", FakeModels)
    monkeypatch.setattr(
        transcription_handler,
        "OpenAIServingTranscription",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        transcription_handler,
        "VllmEngineMonitor",
        lambda *_args, **_kwargs: object(),
    )

    config = SimpleNamespace(
        model="openai/whisper-tiny",
        served_model_name="whisper",
        served_model_aliases=["whisper-alias", "stt"],
    )
    TranscriptionWorkerHandler(runtime=object(), engine=object(), config=config)

    paths = registered["base_model_paths"]
    assert [path.name for path in paths] == ["whisper", "whisper-alias", "stt"]
    assert {path.model_path for path in paths} == {"openai/whisper-tiny"}


def test_decode_audio_rejects_invalid_base64():
    with pytest.raises(ValueError, match="invalid base64"):
        make_handler()._decode_audio({"audio_b64": "not base64"})


def test_native_request_forwards_supported_openai_fields():
    request = make_handler()._native_request(
        {
            "filename": "sample.wav",
            "model": "whisper",
            "language": "en",
            "prompt": "NVIDIA Dynamo",
            "response_format": "verbose_json",
            "temperature": 0.2,
            "timestamp_granularities": ["word"],
        }
    )

    assert request.file.filename == "sample.wav"
    assert request.model == "whisper"
    assert request.language == "en"
    assert request.prompt == "NVIDIA Dynamo"
    assert request.response_format == "verbose_json"
    assert request.timestamp_granularities == ["word"]


def test_generate_delegates_to_native_vllm_serving():
    asyncio.run(_assert_generate_delegates_to_native_vllm_serving())


async def _assert_generate_delegates_to_native_vllm_serving():
    class NativeResponse:
        def model_dump(self, **_kwargs):
            return {
                "text": "hello",
                "usage": {"type": "duration", "seconds": 1},
            }

    handler = make_handler()
    handler._run_with_cancellation = AsyncMock(return_value=NativeResponse())
    context = object()
    chunks = [
        chunk
        async for chunk in handler.generate(
            {
                "audio_b64": base64.b64encode(b"audio").decode("ascii"),
                "filename": "sample.wav",
                "response_format": "json",
            },
            context,
        )
    ]

    assert chunks == [{"text": "hello", "usage": {"type": "duration", "seconds": 1}}]
    handler._run_with_cancellation.assert_awaited_once()


def test_generate_preserves_native_vllm_error_status():
    asyncio.run(_assert_generate_preserves_native_vllm_error_status())


async def _assert_generate_preserves_native_vllm_error_status():
    # Imported here, as in test_realtime_handler, so collection does not need
    # vLLM's serving modules; skip where they are not installed.
    protocol = pytest.importorskip("vllm.entrypoints.serve.engine.protocol")
    ErrorInfo, ErrorResponse = protocol.ErrorInfo, protocol.ErrorResponse

    handler = make_handler()
    handler._run_with_cancellation = AsyncMock(
        return_value=ErrorResponse(
            error=ErrorInfo(
                message="The model does not exist",
                type="NotFoundError",
                code=404,
            )
        )
    )

    with pytest.raises(HttpError, match="The model does not exist") as error:
        await anext(
            handler.generate(
                {
                    "audio_b64": base64.b64encode(b"audio").decode("ascii"),
                    "filename": "sample.wav",
                    "response_format": "json",
                },
                object(),
            )
        )

    assert error.value.code == 404


@pytest.mark.parametrize("error_name", ["VLLMValidationError", "ValueError"])
def test_generate_reports_vllm_request_errors_as_invalid_argument(error_name):
    asyncio.run(_assert_generate_reports_vllm_request_error(error_name))


async def _assert_generate_reports_vllm_request_error(error_name):
    # VLLMValidationError (a prompt longer than the model context, for one) is not a
    # ValueError; both must reach the client as a 400 with their message.
    exceptions = pytest.importorskip("vllm.exceptions")
    error_cls = (
        ValueError if error_name == "ValueError" else getattr(exceptions, error_name)
    )
    handler = make_handler()
    handler._run_with_cancellation = AsyncMock(
        side_effect=error_cls("bad request detail")
    )

    with pytest.raises(InvalidArgument, match="bad request detail"):
        await anext(
            handler.generate(
                {
                    "audio_b64": base64.b64encode(b"audio").decode("ascii"),
                    "filename": "sample.wav",
                    "response_format": "json",
                },
                object(),
            )
        )


@pytest.mark.timeout(5)
def test_outer_cancellation_stops_native_vllm_inference():
    asyncio.run(_assert_outer_cancellation_stops_native_vllm_inference())


async def _assert_outer_cancellation_stops_native_vllm_inference():
    inference_started = asyncio.Event()
    inference_cancelled = asyncio.Event()

    async def create_transcription(*_args, **_kwargs):
        inference_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            inference_cancelled.set()

    async def wait_for_context_cancellation():
        await asyncio.Event().wait()

    handler = make_handler()
    handler.serving = SimpleNamespace(create_transcription=create_transcription)
    context = SimpleNamespace(async_killed_or_stopped=wait_for_context_cancellation)

    task = asyncio.create_task(
        handler._run_with_cancellation(b"audio", object(), context)
    )
    await inference_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.wait_for(inference_cancelled.wait(), timeout=1)


@pytest.mark.timeout(5)
def test_context_cancellation_stops_native_vllm_inference():
    asyncio.run(_assert_context_cancellation_stops_native_vllm_inference())


async def _assert_context_cancellation_stops_native_vllm_inference():
    inference_started = asyncio.Event()
    inference_cancelled = asyncio.Event()
    context_cancelled = asyncio.Event()

    async def create_transcription(*_args, **_kwargs):
        inference_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            inference_cancelled.set()

    async def wait_for_context_cancellation():
        await context_cancelled.wait()

    handler = make_handler()
    handler.serving = SimpleNamespace(create_transcription=create_transcription)
    context = SimpleNamespace(async_killed_or_stopped=wait_for_context_cancellation)

    task = asyncio.create_task(
        handler._run_with_cancellation(b"audio", object(), context)
    )
    await inference_started.wait()
    context_cancelled.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.wait_for(inference_cancelled.wait(), timeout=1)
