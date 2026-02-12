# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protocol types for video generation.

These types match the Rust protocol types in lib/llm/src/protocols/openai/videos.rs
to ensure compatibility with the Dynamo HTTP frontend.
"""

from typing import Optional

from pydantic import BaseModel


class NvCreateVideoRequest(BaseModel):
    """Request for video generation (/v1/videos/generations endpoint).

    Matches Rust NvCreateVideoRequest in lib/llm/src/protocols/openai/videos.rs.
    """

    # Required fields
    prompt: str
    """The text prompt for video generation."""

    model: str
    """The model to use for video generation."""

    # Optional fields
    input_reference: Optional[str] = None
    """Optional input reference for I2V (image path/url)."""

    seconds: Optional[int] = None
    """Duration in seconds (default: 4)."""

    fps: Optional[int] = None
    """Frames per second (default: 24)."""

    num_frames: Optional[int] = None
    """Number of frames to generate (overrides fps * seconds if set)."""

    size: Optional[str] = None
    """Video size in WxH format (default: '832x480')."""

    num_inference_steps: Optional[int] = None
    """Number of denoising steps (default: 50)."""

    guidance_scale: Optional[float] = None
    """CFG guidance scale (default: 5.0)."""

    negative_prompt: Optional[str] = None
    """Optional negative prompt."""

    seed: Optional[int] = None
    """Random seed for reproducibility."""

    user: Optional[str] = None
    """Optional user identifier."""

    response_format: Optional[str] = None
    """Response format: 'url' or 'b64_json' (default: 'url')."""


class VideoData(BaseModel):
    """Video data in response.

    Matches Rust VideoData in lib/llm/src/protocols/openai/videos.rs.
    """

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
