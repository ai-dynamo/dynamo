# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Expose a Magpie TTS NIM through Dynamo's OpenAI audio/speech API."""

from __future__ import annotations

import argparse
import asyncio
import logging

import uvloop

from dynamo.llm import ModelInput, ModelType, WorkerType, register_model
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

from .audio_speech import SpeechNimAudioSpeechBackend
from .config import (
    add_nim_connection_args,
    nim_connection_config_from_namespace,
    resolve_dynamo_endpoint,
)
from .riva_client import build_tts_service, wait_for_service_ready

logger = logging.getLogger(__name__)
configure_dynamo_logging(service_name="speech-tts")

DEFAULT_VOICE = "Magpie-Multilingual.EN-US.Aria"
DEFAULT_LANGUAGE_CODE = "en-US"
DEFAULT_SAMPLE_RATE_HZ = 24_000
DEFAULT_MODEL_NAME = "nvidia/magpie-tts-multilingual"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_nim_connection_args(parser)
    parser.add_argument("--voice", default=DEFAULT_VOICE, help="Default Magpie voice.")
    parser.add_argument(
        "--language-code",
        default=DEFAULT_LANGUAGE_CODE,
        help="Default BCP-47 language code.",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=int,
        default=DEFAULT_SAMPLE_RATE_HZ,
        help="PCM output sample rate.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Model name exposed through /v1/audio/speech.",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Dynamo endpoint backing the public audio model.",
    )
    return parser.parse_args()


@dynamo_worker()
async def worker(runtime: DistributedRuntime, args: argparse.Namespace) -> None:
    endpoint_name = resolve_dynamo_endpoint(args.endpoint, "speech-tts-audio")
    endpoint = runtime.endpoint(endpoint_name)
    tts_service = build_tts_service(nim_connection_config_from_namespace(args))
    await wait_for_service_ready(tts_service, args.nim_startup_timeout_s)

    backend = SpeechNimAudioSpeechBackend(
        tts_service=tts_service,
        model_name=args.model_name,
        voice=args.voice,
        language_code=args.language_code,
        sample_rate_hz=args.sample_rate_hz,
    )
    await register_model(
        ModelInput.Text,
        ModelType.Audios,
        endpoint,
        args.model_name,
        model_name=args.model_name,
        worker_type=WorkerType.Aggregated,
    )
    logger.info(
        "Serving Magpie TTS model=%s endpoint=%s", args.model_name, endpoint_name
    )
    await endpoint.serve_endpoint(backend.speech_endpoint)


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker(_parse_args()))
