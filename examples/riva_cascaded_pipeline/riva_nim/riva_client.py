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

"""Riva gRPC clients shared by the ASR and TTS workers."""

from __future__ import annotations

import asyncio
from typing import Any

import grpc
from riva.client import ASRService, Auth, SpeechSynthesisService

from .config import RivaConnectionConfig


def build_auth(config: RivaConnectionConfig) -> Auth:
    """Create local or NVCF Riva authentication from connection settings."""
    metadata = []
    if config.function_id:
        metadata.append(["function-id", config.function_id])
    if config.api_key:
        metadata.append(["authorization", f"Bearer {config.api_key}"])
    return Auth(
        ssl_root_cert=config.ssl_root_cert,
        use_ssl=config.use_ssl,
        uri=config.server,
        metadata_args=metadata or None,
    )


async def wait_for_service_ready(service: Any, timeout_s: float) -> None:
    """Wait until a Riva service's gRPC channel is ready."""
    try:
        await asyncio.to_thread(
            grpc.channel_ready_future(service.auth.channel).result,
            timeout=timeout_s,
        )
    except grpc.FutureTimeoutError as exc:
        raise TimeoutError(
            f"Riva gRPC server {service.auth.uri} was not ready after "
            f"{timeout_s:g} seconds"
        ) from exc


def build_tts_service(config: RivaConnectionConfig) -> SpeechSynthesisService:
    """Create a Riva speech synthesis client."""
    return SpeechSynthesisService(build_auth(config))


def build_asr_service(config: RivaConnectionConfig) -> ASRService:
    """Create a Riva ASR client."""
    return ASRService(build_auth(config))
