# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo adapter for vLLM's native OpenAI transcription serving layer."""

import asyncio
import base64
import binascii
import logging
from io import BytesIO
from typing import Any, AsyncIterator

from fastapi import UploadFile
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.speech_to_text.transcription.protocol import TranscriptionRequest
from vllm.entrypoints.speech_to_text.transcription.serving import (
    OpenAIServingTranscription,
)

from dynamo._core import Context

from .args import Config
from .engine_monitor import VllmEngineMonitor

logger = logging.getLogger(__name__)

_TRANSCRIPTION_FIELDS = (
    "language",
    "prompt",
    "response_format",
    "temperature",
    "timestamp_granularities",
)


class TranscriptionWorkerHandler:
    """Serve batch STT requests using vLLM's model-native implementation."""

    def __init__(
        self,
        runtime: Any,
        engine: Any,
        config: Config,
        shutdown_event: asyncio.Event | None = None,
    ) -> None:
        self.runtime = runtime
        self.engine_client = engine
        self.config = config
        self.shutdown_event = shutdown_event
        self.engine_monitor = VllmEngineMonitor(runtime, engine, shutdown_event)

        model_name = config.served_model_name or config.model
        models = OpenAIServingModels(
            engine,
            [BaseModelPath(name=model_name, model_path=config.model)],
        )
        self.serving = OpenAIServingTranscription(
            engine,
            models,
            request_logger=None,
        )
        logger.info("Transcription worker handler initialized for %s", model_name)

    @staticmethod
    def _decode_audio(request: dict[str, Any]) -> bytes:
        encoded = request.get("audio_b64")
        if not isinstance(encoded, str) or not encoded:
            raise ValueError("Transcription request missing required 'audio_b64' field")
        try:
            audio = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError(
                "Transcription request contains invalid base64 audio"
            ) from error
        if not audio:
            raise ValueError("Transcription audio must not be empty")
        return audio

    def _native_request(self, request: dict[str, Any]) -> TranscriptionRequest:
        filename = request.get("filename") or "audio"
        upload = UploadFile(file=BytesIO(b""), filename=filename)
        values = {
            field: request[field]
            for field in _TRANSCRIPTION_FIELDS
            if request.get(field) is not None
        }
        granularities = values.pop("timestamp_granularities", None)
        if granularities is not None:
            values["timestamp_granularities[]"] = granularities
        values["file"] = upload
        values["model"] = (
            request.get("model") or self.config.served_model_name or self.config.model
        )
        return TranscriptionRequest(**values)

    async def _run_with_cancellation(
        self,
        audio: bytes,
        request: TranscriptionRequest,
        context: Context,
    ) -> Any:
        inference = asyncio.create_task(
            self.serving.create_transcription(audio, request, raw_request=None)
        )
        cancelled = asyncio.ensure_future(context.async_killed_or_stopped())
        try:
            done, _ = await asyncio.wait(
                (inference, cancelled), return_when=asyncio.FIRST_COMPLETED
            )
            if cancelled in done and not inference.done():
                raise asyncio.CancelledError
            return await inference
        finally:
            if not inference.done():
                inference.cancel()
            await asyncio.gather(inference, return_exceptions=True)
            if not cancelled.done():
                cancelled.cancel()
            await asyncio.gather(cancelled, return_exceptions=True)

    async def generate(
        self, request: dict[str, Any], context: Context
    ) -> AsyncIterator[dict[str, Any]]:
        audio = self._decode_audio(request)
        native_request = self._native_request(request)
        if native_request.response_format not in ("json", "verbose_json"):
            raise ValueError(
                "Transcription response_format must be 'json' or 'verbose_json'"
            )

        result = await self._run_with_cancellation(audio, native_request, context)
        if isinstance(result, ErrorResponse):
            raise ValueError(result.message)
        if not hasattr(result, "model_dump"):
            raise TypeError(
                f"Unexpected vLLM transcription response type: {type(result).__name__}"
            )
        yield result.model_dump(mode="json", exclude_none=True)
