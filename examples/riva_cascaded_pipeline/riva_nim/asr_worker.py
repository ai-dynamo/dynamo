# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RIVA ASR worker: transcribe speech to text via a RIVA NIM.

Internal Dynamo worker (called by the orchestrator, not the frontend). Takes a
base64-encoded audio buffer and returns the recognized transcript.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import logging

import uvloop
from pydantic import BaseModel, Field
from riva.client import AudioEncoding, RecognitionConfig

from dynamo.runtime import DistributedRuntime, dynamo_endpoint, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

from .config import (
    RivaConnectionConfig,
    add_riva_connection_args,
    riva_connection_config_from_namespace,
)
from .riva_client import build_asr_service

logger = logging.getLogger(__name__)
configure_dynamo_logging(service_name="riva-asr")

# RIVA ASR default: 16 kHz LINEAR_PCM input. Empty model lets the server pick.
DEFAULT_SAMPLE_RATE_HZ = 16000
DEFAULT_LANGUAGE_CODE = "en-US"
DEFAULT_MODEL = ""


class AsrRequest(BaseModel):
    audio_base64: str = Field(
        description="Base64-encoded LINEAR_PCM audio to transcribe."
    )


class AsrResponse(BaseModel):
    transcript: str = Field(description="Recognized transcript text.")


class AsrBackend:
    """Wraps a RIVA ``ASRService`` as an audio-to-text endpoint.

    Args:
        connection: RIVA gRPC connection settings.
        sample_rate_hz: Sample rate of the incoming LINEAR_PCM audio.
        language_code: BCP-47 language code passed to RIVA.
        model: RIVA ASR model name; empty lets the server choose.
    """

    def __init__(
        self,
        connection: RivaConnectionConfig,
        sample_rate_hz: int,
        language_code: str,
        model: str,
    ) -> None:
        self.sample_rate_hz = sample_rate_hz
        self.language_code = language_code
        self.model = model
        self.asr = build_asr_service(connection)

    async def generate(self, request: AsrRequest) -> AsrResponse:
        """Transcribe ``request.audio_base64`` and return the recognized text.

        Args:
            request: The base64-encoded LINEAR_PCM audio buffer.

        Returns:
            The recognized transcript (concatenated across result segments).
        """
        audio = base64.b64decode(request.audio_base64)
        recognition_config = RecognitionConfig(
            encoding=AudioEncoding.LINEAR_PCM,
            sample_rate_hertz=self.sample_rate_hz,
            language_code=self.language_code,
            model=self.model,
            max_alternatives=1,
            enable_automatic_punctuation=True,
        )
        # offline_recognize() is a blocking gRPC call; keep it off the event loop.
        response = await asyncio.to_thread(
            self.asr.offline_recognize, audio, recognition_config
        )
        transcript = " ".join(
            result.alternatives[0].transcript
            for result in response.results
            if result.alternatives
        )
        return AsrResponse(transcript=transcript)

    @dynamo_endpoint(AsrRequest, AsrResponse)
    async def recognize_endpoint(self, request: AsrRequest):
        yield (await self.generate(request)).model_dump()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RIVA ASR worker for Dynamo")
    add_riva_connection_args(parser)
    parser.add_argument(
        "--sample-rate-hz",
        type=int,
        default=DEFAULT_SAMPLE_RATE_HZ,
        help="Sample rate of the incoming audio in Hz.",
    )
    parser.add_argument(
        "--language-code",
        default=DEFAULT_LANGUAGE_CODE,
        help="BCP-47 language code.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="RIVA ASR model name (empty lets the server choose).",
    )
    parser.add_argument(
        "--endpoint",
        default="dynamo.riva-asr.generate",
        help="Dynamo endpoint to serve (namespace.component.endpoint).",
    )
    return parser.parse_args()


@dynamo_worker()
async def worker(runtime: DistributedRuntime, args: argparse.Namespace) -> None:
    backend = AsrBackend(
        riva_connection_config_from_namespace(args),
        sample_rate_hz=args.sample_rate_hz,
        language_code=args.language_code,
        model=args.model,
    )
    endpoint = runtime.endpoint(args.endpoint)
    logger.info("Serving RIVA ASR endpoint %s", args.endpoint)
    await endpoint.serve_endpoint(backend.recognize_endpoint)


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker(_parse_args()))
