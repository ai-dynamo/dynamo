# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

"""ASR protocol models and Pipecat adapter for Dynamo ASR endpoints."""

from __future__ import annotations

import asyncio
import base64
import os
import sys
from typing import Any, Literal

PIPECAT_SRC = os.environ.get("PIPECAT_SRC", "/home/nealv/dynamo/pipecat/src")
if os.path.isdir(PIPECAT_SRC):
    sys.path.insert(0, PIPECAT_SRC)

from pipecat.frames.frames import (
    ErrorFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
)
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.utils.time import time_now_iso8601
from pydantic import BaseModel

from dynamo.runtime import DistributedRuntime


class AsrRequest(BaseModel):
    audio_b64: str
    encoding: Literal["pcm_s16le", "wav"] = "pcm_s16le"
    sample_rate_hz: int
    channels: int
    language: str = "en-US"
    model: str | None = None
    request_id: str | None = None


class AsrTranscript(BaseModel):
    text: str
    is_final: bool = True
    start_ms: int | None = None
    end_ms: int | None = None
    confidence: float | None = None


def runtime_from_env() -> DistributedRuntime:
    return DistributedRuntime(
        asyncio.get_running_loop(),
        os.environ.get("DYN_DISCOVERY_BACKEND", "etcd"),
        os.environ.get("DYN_REQUEST_PLANE", "tcp"),
    )


def parse_asr_transcript(chunk: Any) -> AsrTranscript:
    raw = chunk.data() if hasattr(chunk, "data") else chunk
    if isinstance(raw, AsrTranscript):
        return raw
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        return AsrTranscript.model_validate_json(raw)
    return AsrTranscript.model_validate(raw)


class DynamoASRService(SegmentedSTTService):
    """Segmented Pipecat STT service backed by a Dynamo ASR endpoint."""

    def __init__(
        self,
        *,
        endpoint_path: str,
        sample_rate: int,
        channels: int,
        language: str,
        model: str | None,
    ):
        super().__init__(
            sample_rate=sample_rate,
            audio_passthrough=False,
            settings=STTSettings(model=model, language=language),
            ttfs_p99_latency=1.0,
        )
        self._endpoint_path = endpoint_path
        self._channels = channels
        self._language = language
        self._model = model
        self._runtime: DistributedRuntime | None = None
        self._client: Any | None = None

    async def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client

        self._runtime = runtime_from_env()
        endpoint = self._runtime.endpoint(self._endpoint_path)
        self._client = await endpoint.client()
        await self._client.wait_for_instances()
        return self._client

    async def start(self, frame):
        await super().start(frame)
        await self._ensure_client()

    async def run_stt(self, audio: bytes):
        try:
            client = await self._ensure_client()
            request = AsrRequest(
                audio_b64=base64.b64encode(audio).decode("ascii"),
                encoding="wav",
                sample_rate_hz=self._sample_rate,
                channels=self._channels,
                language=self._language,
                model=self._model,
            )
            stream = await client.generate(request.model_dump_json())
            async for chunk in stream:
                transcript = parse_asr_transcript(chunk)
                if not transcript.text:
                    continue

                result = transcript.model_dump()
                if transcript.is_final:
                    yield TranscriptionFrame(
                        transcript.text,
                        self._user_id,
                        time_now_iso8601(),
                        None,
                        result,
                    )
                else:
                    yield InterimTranscriptionFrame(
                        transcript.text,
                        self._user_id,
                        time_now_iso8601(),
                        None,
                        result,
                    )
        except Exception as exc:
            yield ErrorFrame(
                error=f"Dynamo ASR endpoint {self._endpoint_path} failed: {exc}"
            )
