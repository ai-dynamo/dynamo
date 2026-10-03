# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Generated from lib/llm/src/protocols/openai/media_schemas/audio.json by
# scripts/generate_media_protocols.py. Do not edit.

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AudioData(BaseModel):
    """
    Audio data in response
    """

    output_format: str
    """
    Actual codec used for this audio: "wav", "mp3", "pcm", "flac", "aac", "opus"
    """
    url: str | None = None
    """
    URL of the generated audio (if data_source is "url")
    """
    b64_json: str | None = None
    """
    Base64-encoded audio data (if data_source is "b64_json")
    """


class AudioNvExt(BaseModel):
    """
    NVIDIA extensions to the Audio Speech API
    """

    annotations: list[str] | None = None
    """
    Annotations for SSE stream events
    """
    frontend_accepts_audio_chunks: bool | None = None
    """
    Internal frontend-to-worker compatibility signal.

    New frontends set this before forwarding `/v1/audio/speech`. When absent
    or false, workers must return one aggregated response so older frontends
    do not decode only the first chunk during rolling upgrades.

    TODO(v1.7): Remove after v1.4 leaves the N-2 compatibility window.
    """
    cfg_scale: float | None = None
    """
    Classifier-free guidance scale (Audex only, hence an extension rather
    than a top-level OpenAI field). Unset or 1.0 decodes unguided; higher
    values follow the prompt more closely. Declared here because serde drops
    unknown `nvext` keys, so without the field the client's value never
    reaches the worker and guidance is silently never applied.
    """


class NvAudioSpeechResponse(BaseModel):
    """
    Response structure for audio speech generation
    """

    id: str
    """
    Unique identifier for the response
    """
    object: str = 'audio.speech'
    """
    Object type (always "audio.speech")
    """
    model: str
    """
    Model used for generation
    """
    status: str = 'completed'
    """
    Status of the generation ("completed", "failed", etc.)
    """
    progress: int = 100
    """
    Progress percentage (0-100)
    """
    created: int
    """
    Unix timestamp of creation
    """
    data: list[AudioData] = Field([], validate_default=True)
    """
    Generated audio data
    """
    error: str | None = None
    """
    Error message if generation failed
    """
    inference_time_s: float | None = None
    """
    Inference time in seconds
    """


class NvCreateAudioSpeechRequest(BaseModel):
    """
    Request for audio speech generation (/v1/audio/speech endpoint).

    Follows vLLM-Omni's OpenAICreateSpeechRequest format with TTS-specific
    parameters as top-level fields.
    """

    input: str
    """
    The text to synthesize into speech (required)
    """
    model: str | None = None
    """
    The TTS model to use
    """
    voice: str | None = None
    """
    Voice/speaker name (e.g., "vivian", "ryan", "aiden")
    """
    data_source: str | None = None
    response_format: str | None = None
    """
    Output codec: "wav", "mp3", "pcm", "flac", "aac", "opus" (default: "wav")
    """
    speed: float | None = None
    """
    Speed factor. The frontend rejects a value outside 0.25 to 4.0.
    Absent means 1.0.
    """
    task_type: str | None = None
    """
    TTS task type: "CustomVoice", "VoiceDesign", or "Base"
    """
    language: str | None = None
    """
    Language: "Auto", "Chinese", "English", "Japanese", etc.
    """
    instructions: str | None = None
    """
    Voice style/emotion instructions (for VoiceDesign)
    """
    ref_audio: str | None = None
    """
    Reference audio URL or base64 (for voice cloning with Base task)
    """
    ref_text: str | None = None
    """
    Reference transcript (for voice cloning with Base task)
    """
    max_new_tokens: int | None = None
    """
    Maximum tokens to generate (default: 2048)
    """
    user: str | None = None
    """
    Optional user identifier
    """
    nvext: AudioNvExt | None = None
    extra_args: dict[str, Any] | None = None
    """
    Worker-boundary passthrough. The frontend nests unknown top-level request fields (an OpenAI client's extra_body) under the "media_passthrough" key.
    """
