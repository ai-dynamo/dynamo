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

"""Expose a Riva streaming ASR NIM through Dynamo's OpenAI Realtime API."""

from __future__ import annotations

import argparse
import asyncio
import logging

import uvloop

from dynamo.llm import ModelInput, ModelType, WorkerType, register_model
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.vllm.realtime import RealtimeHandler

from .config import (
    add_riva_connection_args,
    resolve_dynamo_endpoint,
    riva_connection_config_from_namespace,
)
from .realtime_asr import RivaRealtimeTranscriptionHandler
from .riva_client import build_asr_service, wait_for_service_ready

logger = logging.getLogger(__name__)
configure_dynamo_logging(service_name="riva-asr")

DEFAULT_LANGUAGE_CODE = "en-US"
DEFAULT_RIVA_MODEL = ""
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MODEL_NAME = "nemotron-asr-streaming"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_riva_connection_args(parser)
    parser.add_argument(
        "--language-code",
        default=DEFAULT_LANGUAGE_CODE,
        help="BCP-47 language code accepted by the worker.",
    )
    parser.add_argument(
        "--riva-model",
        default=DEFAULT_RIVA_MODEL,
        help="Riva model name (empty lets the NIM choose).",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help="Deadline for a committed transcription turn.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Model name exposed through /v1/realtime.",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Dynamo endpoint backing the public realtime model.",
    )
    return parser.parse_args()


@dynamo_worker()
async def worker(runtime: DistributedRuntime, args: argparse.Namespace) -> None:
    endpoint_name = resolve_dynamo_endpoint(args.endpoint, "riva-asr-realtime")
    endpoint = runtime.endpoint(endpoint_name)
    asr_service = build_asr_service(riva_connection_config_from_namespace(args))
    await wait_for_service_ready(asr_service, args.riva_startup_timeout_s)

    handler = RealtimeHandler(
        {
            "transcription": RivaRealtimeTranscriptionHandler(
                asr_service=asr_service,
                model_name=args.model_name,
                riva_model=args.riva_model,
                language_code=args.language_code,
                timeout_s=args.timeout_s,
            )
        }
    )
    await register_model(
        ModelInput.Text,
        ModelType.Realtime,
        endpoint,
        args.model_name,
        model_name=args.model_name,
        worker_type=WorkerType.Aggregated,
    )
    logger.info("Serving Riva ASR model=%s endpoint=%s", args.model_name, endpoint_name)
    await endpoint.serve_bidirectional_endpoint(handler.generate)


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker(_parse_args()))
