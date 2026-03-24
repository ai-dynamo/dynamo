# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Optional

from pydantic import BaseModel

# For omni models, we need to support raw_request parsing and json output format. We need to have these protocols defined here for serialization and deserialization.
# TODO: Replace these Pydantic models with Python bindings to the Rust protocol types once PyO3 bindings are available.


class ImageNvExt(BaseModel):
    """NVIDIA extensions for image generation requests.

    Matches Rust NvExt in lib/llm/src/protocols/openai/images/nvext.rs.
    """

    annotations: Optional[list[str]] = None
    """Annotations for SSE stream events."""

    negative_prompt: Optional[str] = None
    """Optional negative prompt."""

    num_inference_steps: Optional[int] = None
    """Number of denoising steps."""

    guidance_scale: Optional[float] = None
    """CFG guidance scale."""

    seed: Optional[int] = None
    """Random seed for reproducibility."""


class NvCreateImageRequest(BaseModel):
    """Request for image generation (/v1/images/generations endpoint).

    Matches the flattened Rust NvCreateImageRequest in lib/llm/src/protocols/openai/images.rs
    """

    prompt: str
    """The text prompt for image generation."""

    model: Optional[str] = None
    """The model to use for image generation."""

    n: Optional[int] = None
    """Number of images to generate (1-10)."""

    quality: Optional[str] = None
    """Image quality: standard, hd, high, medium, low, auto."""

    response_format: Optional[str] = None
    """Response format: url or b64_json."""

    size: Optional[str] = None
    """Image size in WxH format (e.g. 1024x1024)."""

    style: Optional[str] = None
    """Image style: vivid or natural."""

    user: Optional[str] = None
    """Optional user identifier."""

    moderation: Optional[str] = None
    """Content moderation level: auto or low."""

    nvext: Optional[ImageNvExt] = None
    """NVIDIA extensions."""


class ImageData(BaseModel):
    """Individual image data in a response.

    Matches the flattened Rust Image enum in lib/async-openai/src/types/image.rs.
    """

    url: Optional[str] = None
    """URL of the generated image (if response_format is url)."""

    b64_json: Optional[str] = None
    """Base64-encoded image (if response_format is b64_json)."""

    revised_prompt: Optional[str] = None
    """Revised prompt, when the model rewrites the original prompt."""


class NvImagesResponse(BaseModel):
    """Response structure for image generation.

    Matches the flattened Rust NvImagesResponse in lib/llm/src/protocols/openai/images.rs
    """

    created: int
    """Unix timestamp of creation."""

    data: list[ImageData] = []
    """List of generated images."""
