# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("fastapi")

from dynamo.vllm.transcription_handler import TranscriptionWorkerHandler  # noqa: E402

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
]


def make_handler() -> TranscriptionWorkerHandler:
    handler = TranscriptionWorkerHandler.__new__(TranscriptionWorkerHandler)
    handler.config = SimpleNamespace(
        model="openai/whisper-tiny",
        served_model_name="whisper",
    )
    return handler


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
