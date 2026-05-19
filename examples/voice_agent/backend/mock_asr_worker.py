# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo ASR worker backed by a Parakeet/Riva gRPC ASR service."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import wave
from dataclasses import dataclass
from typing import AsyncIterator, Iterator

import riva.client
import uvloop
from asr_protocol import AsrRequest, AsrTranscript
from riva.client.asr import streaming_request_generator
from riva.client.proto.riva_asr_pb2 import RivaSpeechRecognitionConfigRequest

from dynamo.runtime import DistributedRuntime, dynamo_endpoint, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

ASR_ENDPOINT = "voice_agent.asr.transcribe"
DEFAULT_RIVA_URI = "127.0.0.1:50051"
DEFAULT_CHUNK_MS = 100
DEFAULT_RIVA_TIMEOUT_SECONDS = 300.0
DEFAULT_RIVA_CONFIG_TIMEOUT_SECONDS = 10.0

logger = logging.getLogger(__name__)
configure_dynamo_logging(service_name="voice-agent-parakeet-asr")


@dataclass(frozen=True)
class AudioInput:
    pcm: bytes
    sample_rate_hz: int
    channels: int
    seconds: float


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    return float(value)


def decode_audio(request: AsrRequest, audio: bytes) -> AudioInput:
    if request.encoding == "wav":
        with wave.open(io.BytesIO(audio), "rb") as wav:
            sample_width = wav.getsampwidth()
            if sample_width != 2:
                raise ValueError(
                    f"Riva streaming ASR expects 16-bit PCM WAV, got {sample_width} bytes/sample"
                )
            sample_rate_hz = wav.getframerate()
            channels = wav.getnchannels()
            pcm = wav.readframes(wav.getnframes())
            seconds = wav.getnframes() / sample_rate_hz
            return AudioInput(
                pcm=pcm,
                sample_rate_hz=sample_rate_hz,
                channels=channels,
                seconds=seconds,
            )

    bytes_per_sample = 2
    seconds = len(audio) / (
        request.sample_rate_hz * request.channels * bytes_per_sample
    )
    return AudioInput(
        pcm=audio,
        sample_rate_hz=request.sample_rate_hz,
        channels=request.channels,
        seconds=seconds,
    )


def audio_chunks(audio: AudioInput, chunk_ms: int) -> Iterator[bytes]:
    frame_bytes = audio.channels * 2
    chunk_frames = max(1, int(audio.sample_rate_hz * chunk_ms / 1000))
    chunk_bytes = max(frame_bytes, chunk_frames * frame_bytes)
    chunk_bytes -= chunk_bytes % frame_bytes
    for offset in range(0, len(audio.pcm), chunk_bytes):
        yield audio.pcm[offset : offset + chunk_bytes]


def riva_streaming_config(
    request: AsrRequest,
    audio: AudioInput,
    model: str,
):
    return riva.client.StreamingRecognitionConfig(
        config=riva.client.RecognitionConfig(
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
            sample_rate_hertz=audio.sample_rate_hz,
            language_code=request.language,
            audio_channel_count=audio.channels,
            max_alternatives=1,
            enable_automatic_punctuation=env_bool("RIVA_ASR_PUNCTUATION", True),
            model=model,
        ),
        interim_results=env_bool("RIVA_ASR_INTERIM_RESULTS", True),
    )


def model_parameters(model_config) -> dict[str, str]:
    return {key: value for key, value in model_config.parameters.items()}


def model_supports_streaming(model_config) -> bool:
    parameters = model_parameters(model_config)
    is_streaming = parameters.get("streaming", "").lower() == "true"
    model_type = parameters.get("type", "").lower()
    return is_streaming and model_type != "offline"


def discover_streaming_model(
    service: riva.client.ASRService,
    auth: riva.client.Auth,
) -> str:
    timeout = env_float(
        "RIVA_ASR_CONFIG_TIMEOUT_SECONDS",
        DEFAULT_RIVA_CONFIG_TIMEOUT_SECONDS,
    )
    response = service.stub.GetRivaSpeechRecognitionConfig(
        RivaSpeechRecognitionConfigRequest(),
        metadata=auth.get_auth_metadata(),
        timeout=timeout,
    )
    models = list(response.model_config)
    streaming_models = [
        model.model_name for model in models if model_supports_streaming(model)
    ]
    if streaming_models:
        return streaming_models[0]

    advertised = ", ".join(
        f"{model.model_name} ({model_parameters(model)})" for model in models
    )
    raise RuntimeError(
        "Riva ASR service does not advertise a streaming model for "
        f"StreamingRecognize. Available models: {advertised or '<none>'}"
    )


def resolve_model(
    request: AsrRequest,
    service: riva.client.ASRService,
    auth: riva.client.Auth,
) -> str:
    explicit_model = os.environ.get("RIVA_ASR_MODEL") or request.model
    if explicit_model:
        return explicit_model
    return discover_streaming_model(service, auth)


def riva_transcripts(request: AsrRequest, audio: AudioInput) -> Iterator[AsrTranscript]:
    uri = os.environ.get("RIVA_ASR_URI", DEFAULT_RIVA_URI)
    chunk_ms = env_int("RIVA_ASR_CHUNK_MS", DEFAULT_CHUNK_MS)
    timeout = env_float("RIVA_ASR_TIMEOUT_SECONDS", DEFAULT_RIVA_TIMEOUT_SECONDS)
    auth = riva.client.Auth(uri=uri, use_ssl=env_bool("RIVA_ASR_USE_SSL", False))
    service = riva.client.ASRService(auth)
    model = resolve_model(request, service, auth)
    streaming_config = riva_streaming_config(request, audio, model)

    logger.info(
        "streaming ASR request to Riva: %.2fs, %s Hz, %s channel(s), chunk_ms=%s, uri=%s, model=%s, timeout=%.1fs",
        audio.seconds,
        audio.sample_rate_hz,
        audio.channels,
        chunk_ms,
        uri,
        model,
        timeout,
    )

    request_stream = streaming_request_generator(
        audio_chunks(audio, chunk_ms),
        streaming_config,
    )
    response_stream = service.stub.StreamingRecognize(
        request_stream,
        metadata=auth.get_auth_metadata(),
        timeout=timeout,
    )

    for response in response_stream:
        for result in response.results:
            if not result.alternatives:
                continue
            alternative = result.alternatives[0]
            transcript = alternative.transcript.strip()
            if not transcript:
                continue
            yield AsrTranscript(
                text=transcript,
                is_final=result.is_final,
                start_ms=None,
                end_ms=(
                    int(result.audio_processed * 1000)
                    if result.audio_processed
                    else None
                ),
                confidence=alternative.confidence or None,
            )


async def stream_riva_transcripts(
    request: AsrRequest,
    audio: AudioInput,
) -> AsyncIterator[AsrTranscript]:
    queue: asyncio.Queue[AsrTranscript | Exception | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def produce() -> None:
        try:
            saw_transcript = False
            for transcript in riva_transcripts(request, audio):
                saw_transcript = True
                loop.call_soon_threadsafe(queue.put_nowait, transcript)
            if not saw_transcript:
                raise RuntimeError("Riva ASR stream completed without a transcript")
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    producer_task = asyncio.create_task(asyncio.to_thread(produce))
    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        await producer_task


@dynamo_endpoint(AsrRequest, AsrTranscript)
async def transcribe(request: AsrRequest):
    audio_bytes = base64.b64decode(request.audio_b64, validate=False)
    audio = decode_audio(request, audio_bytes)
    logger.info(
        "Dynamo ASR request: %.2fs, %s Hz, %s channel(s), encoding=%s",
        audio.seconds,
        audio.sample_rate_hz,
        audio.channels,
        request.encoding,
    )

    async for transcript in stream_riva_transcripts(request, audio):
        yield transcript.model_dump()


@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    endpoint = runtime.endpoint(ASR_ENDPOINT)
    logger.info(
        "Serving ASR endpoint %s using Riva gRPC service %s",
        ASR_ENDPOINT,
        os.environ.get("RIVA_ASR_URI", DEFAULT_RIVA_URI),
    )
    await endpoint.serve_endpoint(transcribe)


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
