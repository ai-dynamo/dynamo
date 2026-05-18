# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mock Dynamo ASR worker for the voice-agent realtime example."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import wave

import uvloop
from asr_protocol import AsrRequest, AsrTranscript

from dynamo.runtime import DistributedRuntime, dynamo_endpoint, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

ASR_ENDPOINT = "voice_agent.asr.transcribe"

logger = logging.getLogger(__name__)
configure_dynamo_logging(service_name="voice-agent-mock-asr")


def audio_duration_seconds(request: AsrRequest, audio: bytes) -> float:
    if request.encoding == "wav":
        with wave.open(io.BytesIO(audio), "rb") as wav:
            return wav.getnframes() / wav.getframerate()

    bytes_per_sample = 2
    return len(audio) / (request.sample_rate_hz * request.channels * bytes_per_sample)


@dynamo_endpoint(AsrRequest, AsrTranscript)
async def transcribe(request: AsrRequest):
    audio = base64.b64decode(request.audio_b64, validate=False)
    seconds = audio_duration_seconds(request, audio)
    logger.info(
        "mock ASR request: %.2fs, %s Hz, %s channel(s), encoding=%s",
        seconds,
        request.sample_rate_hz,
        request.channels,
        request.encoding,
    )

    segments = [
        "mock dynamo asr transcript",
        f"{seconds:.2f} seconds of audio",
        f"{request.sample_rate_hz} Hz, {request.channels} channel",
    ]
    for index, text in enumerate(segments):
        await asyncio.sleep(0.05)
        yield AsrTranscript(
            text=text,
            is_final=True,
            start_ms=None if index == 0 else 0,
            end_ms=int(seconds * 1000) if index == len(segments) - 1 else None,
            confidence=1.0,
        ).model_dump()


@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    endpoint = runtime.endpoint(ASR_ENDPOINT)
    logger.info("Serving mock ASR endpoint %s", ASR_ENDPOINT)
    await endpoint.serve_endpoint(transcribe)


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
