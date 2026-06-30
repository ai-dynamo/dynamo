# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RIVA TTS worker: synthesize speech from text via a RIVA NIM.

Internal Dynamo worker (called by the orchestrator, not the frontend). Takes a
text request and returns the synthesized audio base64-encoded.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import logging

import uvloop
from pydantic import BaseModel, Field

from dynamo.runtime import DistributedRuntime, dynamo_endpoint, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

from .config import (
    RivaConnectionConfig,
    add_riva_connection_args,
    riva_connection_config_from_namespace,
)
from .riva_client import build_tts_service

logger = logging.getLogger(__name__)
configure_dynamo_logging(service_name="riva-tts")

# RIVA TTS defaults: Magpie multilingual voice at 22.05 kHz LINEAR_PCM.
DEFAULT_VOICE = "Magpie-Multilingual.EN-US.Aria"
DEFAULT_LANGUAGE_CODE = "en-US"
DEFAULT_SAMPLE_RATE_HZ = 22050


class TtsRequest(BaseModel):
    text: str = Field(description="Text to synthesize into speech.")


class TtsResponse(BaseModel):
    audio_base64: str = Field(description="Base64-encoded synthesized audio.")
    sample_rate_hz: int = Field(description="Sample rate of the audio, in Hz.")
    encoding: str = Field(default="LINEAR_PCM", description="Audio sample encoding.")


class TtsBackend:
    """Wraps a RIVA ``SpeechSynthesisService`` as a text-to-audio endpoint.

    Args:
        connection: RIVA gRPC connection settings.
        voice: RIVA voice name to synthesize with.
        language_code: BCP-47 language code passed to RIVA.
        sample_rate_hz: Output sample rate requested from RIVA.
    """

    def __init__(
        self,
        connection: RivaConnectionConfig,
        voice: str,
        language_code: str,
        sample_rate_hz: int,
    ) -> None:
        self.voice = voice
        self.language_code = language_code
        self.sample_rate_hz = sample_rate_hz
        self.tts = build_tts_service(connection)

    async def generate(self, request: TtsRequest) -> TtsResponse:
        """Synthesize ``request.text`` and return the audio base64-encoded.

        Args:
            request: The text to synthesize.

        Returns:
            The synthesized audio as base64-encoded ``LINEAR_PCM`` at the
            worker's configured sample rate.
        """
        # synthesize() is a blocking gRPC call; keep it off the event loop.
        response = await asyncio.to_thread(
            self.tts.synthesize,
            request.text,
            voice_name=self.voice,
            language_code=self.language_code,
            sample_rate_hz=self.sample_rate_hz,
        )
        return TtsResponse(
            audio_base64=base64.b64encode(response.audio).decode(),
            sample_rate_hz=self.sample_rate_hz,
        )

    @dynamo_endpoint(TtsRequest, TtsResponse)
    async def synthesize_endpoint(self, request: TtsRequest):
        yield (await self.generate(request)).model_dump()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RIVA TTS worker for Dynamo")
    add_riva_connection_args(parser)
    parser.add_argument("--voice", default=DEFAULT_VOICE, help="RIVA voice name.")
    parser.add_argument(
        "--language-code",
        default=DEFAULT_LANGUAGE_CODE,
        help="BCP-47 language code.",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=int,
        default=DEFAULT_SAMPLE_RATE_HZ,
        help="Output sample rate in Hz.",
    )
    parser.add_argument(
        "--endpoint",
        default="dynamo.riva-tts.generate",
        help="Dynamo endpoint to serve (namespace.component.endpoint).",
    )
    return parser.parse_args()


@dynamo_worker()
async def worker(runtime: DistributedRuntime, args: argparse.Namespace) -> None:
    backend = TtsBackend(
        riva_connection_config_from_namespace(args),
        voice=args.voice,
        language_code=args.language_code,
        sample_rate_hz=args.sample_rate_hz,
    )
    endpoint = runtime.endpoint(args.endpoint)
    logger.info("Serving RIVA TTS endpoint %s", args.endpoint)
    await endpoint.serve_endpoint(backend.synthesize_endpoint)


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker(_parse_args()))
