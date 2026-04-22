# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protocol types for audio generation (TTS).

These types follow the vLLM-Omni OpenAICreateSpeechRequest format,
with TTS-specific parameters as top-level fields (not nested in nvext).

Note: These Pydantic models mirror the Rust protocol types in
lib/llm/src/protocols/openai/audios.rs. Ideally these should be
code-generated from the Rust definitions; for now they are maintained
manually and must be kept in sync.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class NvCreateAudioSpeechRequest(BaseModel):
    """Request for audio speech generation (/v1/audio/speech endpoint).

    Follows vLLM-Omni's OpenAICreateSpeechRequest format.
    """

    # Standard OpenAI params
    input: str
    """The text to synthesize into speech."""

    model: Optional[str] = None
    """The TTS model to use."""

    voice: Optional[str] = None
    """Voice/speaker name (e.g., 'vivian', 'ryan', 'aiden')."""

    data_source: Optional[str] = None
    """How the generated data should be returned: 'url' or 'b64_json' (default: 'b64_json').
    Note that in image and video generation, the 'response_format' is the equivalent of
    this field. However, in audio generation, OpenAI specifies the 'response_format'
    to be used for output format."""

    response_format: Optional[
        Literal["wav", "pcm", "flac", "mp3", "aac", "opus"]
    ] = "wav"
    """Output format."""

    speed: Optional[float] = Field(default=1.0, ge=0.25, le=4.0)
    """Speed factor."""

    # Qwen3-TTS specific params (top-level, matching vLLM-Omni)
    task_type: Optional[Literal["CustomVoice", "VoiceDesign", "Base"]] = None
    """TTS task type."""

    language: Optional[str] = None
    """Language: Auto, Chinese, English, Japanese, Korean, etc."""

    instructions: Optional[str] = None
    """Voice style/emotion instructions (for VoiceDesign)."""

    ref_audio: Optional[str] = None
    """Reference audio URL or base64 (for voice cloning with Base task)."""

    ref_text: Optional[str] = None
    """Reference transcript (for voice cloning with Base task)."""

    max_new_tokens: Optional[int] = None
    """Maximum tokens to generate (default: 2048)."""


class AudioData(BaseModel):
    """Audio data in response."""

    output_format: str
    """Actual codec used for this audio: 'wav', 'mp3', 'pcm', 'flac', 'aac', 'opus'."""

    url: Optional[str] = None
    """URL of the generated audio (if data_source is 'url')."""

    b64_json: Optional[str] = None
    """Base64-encoded audio data (if data_source is 'b64_json')."""


class NvAudioSpeechResponse(BaseModel):
    """Response structure for audio speech generation."""

    id: str
    """Unique identifier for the response."""

    object: str = "audio.speech"
    """Object type."""

    model: str
    """Model used for generation."""

    status: str = "completed"
    """Generation status."""

    progress: int = 100
    """Progress percentage (0-100)."""

    created: int
    """Unix timestamp of creation."""

    data: list[AudioData] = []
    """List of generated audio data."""

    error: Optional[str] = None
    """Error message if generation failed."""

    inference_time_s: Optional[float] = None
    """Inference time in seconds."""
