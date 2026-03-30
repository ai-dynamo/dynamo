# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Optional

from pydantic import BaseModel

# For omni models, we need to support raw_request parsing and json output format. We need to have these protocols defined here for serialization and deserialization.
# TODO: Replace these Pydantic models with Python bindings to the Rust protocol types once PyO3 bindings are available.


class AudioNvExt(BaseModel):
    """NVIDIA extensions for audio speech generation requests.

    Matches Rust NvExt in lib/llm/src/protocols/openai/audios/nvext.rs.
    """

    annotations: Optional[list[str]] = None
    """Annotations for SSE stream events."""

    task_type: Optional[str] = None
    """The type of the audio generation task e.g. 'CustomVoice', 'Base', 'VoiceDesign', etc."""

    language: Optional[str] = None
    """Language code (e.g. 'en', 'zh')."""

    max_new_tokens: Optional[int] = None
    """Maximum number of tokens to generate."""

    ref_audio: Optional[str] = None
    """Base64-encoded reference audio for voice cloning."""

    ref_text: Optional[str] = None
    """Reference text corresponding to ref_audio."""

    x_vector_only_mode: Optional[bool] = None
    """If True, the model uses only the speaker's acoustic features (embedding) without considering the ref_text."""

    speaker_embedding: Optional[list[float]] = None
    """A pre-computed vector representing a voice."""

    initial_codec_chunk_frames: Optional[int] = None
    """Controls the initial buffering size for the audio codec."""


class NvCreateAudioRequest(BaseModel):
    """Request for Audio generation (/v1/audio/speech endpoint).

    Matches the flattened Rust NvCreateAudioRequest in lib/llm/src/protocols/openai/audios.rs
    """

    input: str
    """The text to generate audio for."""

    model: str
    """The TTS model to use for Audio generation."""

    voice: Optional[str] = None
    """The voice to use when generating the audio."""

    instructions: Optional[str] = None
    """Control the voice of your generated audio with additional instructions."""

    response_format: Optional[str] = None
    """The format to audio in. Supported formats are mp3, opus, aac, flac, wav, and pcm."""

    speed: Optional[float] = None
    """Playback speed (0.25 to 4.0, default 1.0)."""

    stream_format: Optional[str] = None
    """The format to stream the audio in."""

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
