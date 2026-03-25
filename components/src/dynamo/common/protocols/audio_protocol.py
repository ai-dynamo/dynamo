# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protocol types for audio speech generation.

These types match the Rust protocol types in lib/llm/src/protocols/openai/audios.rs
to ensure compatibility with the Dynamo HTTP frontend.
"""
# TODO: Replace these Pydantic models with Python bindings to the Rust protocol types once PyO3 bindings are available.

from typing import Optional

from pydantic import BaseModel


class AudioNvExt(BaseModel):
    """NVIDIA extensions for audio speech generation requests.

    Matches Rust NvExt in lib/llm/src/protocols/openai/audios/nvext.rs.
    """

    annotations: Optional[list[str]] = None
    """Annotations for SSE stream events."""

    task_type: Optional[str] = None
    """Task type (e.g. 'tts', 'voice_clone')."""

    language: Optional[str] = None
    """Language code (e.g. 'en', 'zh')."""

    instructions: Optional[str] = None
    """Additional instructions for speech generation."""

    ref_audio: Optional[str] = None
    """Base64-encoded reference audio for voice cloning."""

    ref_text: Optional[str] = None
    """Reference text corresponding to ref_audio."""

    max_new_tokens: Optional[int] = None
    """Maximum number of tokens to generate."""

    seed: Optional[int] = None
    """Random seed for reproducibility."""


class NvCreateAudioSpeechRequest(BaseModel):
    """Request for audio speech generation (/v1/audio/speech endpoint).

    Matches Rust NvCreateAudioSpeechRequest in lib/llm/src/protocols/openai/audios.rs.
    """

    # Required fields
    input: str
    """The text to generate audio for."""

    model: str
    """The model to use for audio generation."""

    voice: str
    """The voice to use for generation."""

    # Optional fields
    response_format: Optional[str] = None
    """Audio format: mp3, wav, opus, aac, flac, pcm."""

    speed: Optional[float] = None
    """Playback speed (0.25 to 4.0, default 1.0)."""

    nvext: Optional[AudioNvExt] = None
    """NVIDIA extensions."""


class NvAudiosResponse(BaseModel):
    """Response structure for audio speech generation.

    Matches Rust NvAudiosResponse in lib/llm/src/protocols/openai/audios.rs.
    Internal transport uses base64-encoded audio. The HTTP handler decodes
    this to return raw binary audio to clients.
    """

    audio_b64: str
    """Base64-encoded audio bytes."""

    content_type: str
    """MIME content type (e.g. 'audio/mpeg', 'audio/wav')."""

    model: str
    """Model used for generation."""

    created: int
    """Unix timestamp of creation."""
