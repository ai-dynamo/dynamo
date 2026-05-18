# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JSON protocol for the voice-agent mock ASR Dynamo endpoint."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


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
