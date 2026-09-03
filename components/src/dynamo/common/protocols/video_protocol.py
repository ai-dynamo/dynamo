# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protocol types for video generation.

These types match the Rust protocol types in lib/llm/src/protocols/openai/videos.rs
to ensure compatibility with the Dynamo HTTP frontend.
"""
# TODO: Replace these Pydantic models with Python bindings to the Rust protocol types once PyO3 bindings are available.

from typing import Literal, Optional

from pydantic import BaseModel, model_validator


class VideoInputReference(BaseModel):
    """Typed conditioning input for video generation."""

    type: Literal["image", "video", "audio"]
    """Reference media type."""

    source: str
    """HTTP(S), data, file URL, or an allowed local path."""


class VideoNvExt(BaseModel):
    """NVIDIA extensions for video generation requests.

    Matches Rust NvExt in lib/llm/src/protocols/openai/videos/nvext.rs.
    """

    annotations: Optional[list[str]] = None
    """Annotations for SSE stream events."""

    fps: Optional[int] = None
    """Frames per second (default: 24)."""

    num_frames: Optional[int] = None
    """Number of frames to generate (overrides fps * seconds if set)."""

    negative_prompt: Optional[str] = None
    """Optional negative prompt."""

    num_inference_steps: Optional[int] = None
    """Number of denoising steps (default: 50)."""

    guidance_scale: Optional[float] = None
    """CFG guidance scale (default: 5.0)."""

    seed: Optional[int] = None
    """Random seed for reproducibility."""

    boundary_ratio: Optional[float] = None
    """MoE expert switching boundary as a fraction of the denoising schedule (vLLM-Omni I2V)."""

    guidance_scale_2: Optional[float] = None
    """CFG scale for the low-noise expert (vLLM-Omni I2V dual-guidance)."""


class NvCreateVideoRequest(BaseModel):
    """Request for video generation (/v1/videos endpoint).

    Matches Rust NvCreateVideoRequest in lib/llm/src/protocols/openai/videos.rs.
    """

    # Required fields
    prompt: str
    """The text prompt for video generation."""

    model: str
    """The model to use for video generation."""

    # Optional fields
    input_reference: Optional[str] = None
    """Optional image reference that guides generation (for I2V)."""

    input_references: Optional[list[VideoInputReference]] = None
    """Typed references; order is preserved within each media type."""

    seconds: Optional[int] = None
    """Clip duration in seconds."""

    size: Optional[str] = None
    """Video size in WxH format (default: '832x480')."""

    user: Optional[str] = None
    """Optional user identifier."""

    response_format: Optional[Literal["url", "b64_json"]] = None
    """How the generated data should be returned: 'url' or 'b64_json'.
    If unset, handlers default to 'url'."""

    output_format: Optional[str] = None
    """Requested container format (e.g. 'mp4', 'mjpeg').
    This is a hint; check output_format in the response data for the actual format."""

    stream: Optional[bool] = None
    """Whether to stream the video generation (default: false)."""

    nvext: Optional[VideoNvExt] = None
    """NVIDIA extensions."""

    @model_validator(mode="after")
    def validate_input_references(self) -> "NvCreateVideoRequest":
        if self.input_reference is not None and self.input_references is not None:
            raise ValueError(
                "input_reference and input_references are mutually exclusive"
            )
        if self.input_references is not None and not self.input_references:
            raise ValueError("input_references must not be empty")
        return self


class VideoData(BaseModel):
    """Video data in response.

    Matches Rust VideoData in lib/llm/src/protocols/openai/videos.rs.
    """

    output_format: str
    """Actual container format of this video."""

    url: Optional[str] = None
    """URL of the generated video (if response_format is 'url')."""

    b64_json: Optional[str] = None
    """Base64-encoded video (if response_format is 'b64_json')."""


class NvVideosResponse(BaseModel):
    """Response structure for video generation.

    Matches Rust NvVideosResponse in lib/llm/src/protocols/openai/videos.rs.
    """

    id: str
    """Unique identifier for the response."""

    object: str = "video"
    """Object type (always 'video')."""

    model: str
    """Model used for generation."""

    status: str = "completed"
    """Generation status."""

    progress: int = 100
    """Progress percentage (0-100)."""

    created: int
    """Unix timestamp of creation."""

    data: list[VideoData] = []
    """List of generated videos."""

    error: Optional[str] = None
    """Error message if generation failed."""

    inference_time_s: Optional[float] = None
    """Inference time in seconds."""
