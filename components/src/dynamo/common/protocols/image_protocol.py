# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Any, Dict, Optional

from pydantic import BaseModel, field_serializer, field_validator

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
    """Request for image generation or editing endpoints.

    Matches the flattened Rust NvCreateImageRequest in lib/llm/src/protocols/openai/images.rs
    """

    extra_args: Optional[Dict[str, Any]] = None
    """Worker-boundary passthrough. The frontend nests unknown top-level
    request fields (an OpenAI client's extra_body) under the
    "media_passthrough" key."""

    prompt: str
    """The text prompt for image generation."""

    model: Optional[str] = None
    """The model to use for image generation."""

    n: Optional[int] = None
    """Number of images to generate (1-10)."""

    quality: Optional[str] = None
    """Image quality: standard, hd, high, medium, low, auto."""

    background: Optional[str] = None
    """Image background: transparent, opaque, or auto."""

    output_format: Optional[str] = None
    """Output image encoding: png, jpeg, or webp."""

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

    input_reference: Optional[list[str]] = None
    """Ordered edit references; singletons retain the legacy string wire form."""

    @field_validator("input_reference", mode="before")
    @classmethod
    def normalize_input_reference(cls, value):
        """Normalize supplied strings and ordered arrays to a nonempty list.

        Keep reference contents out of the validation messages raised here.
        """
        if value is None:
            return None
        references = [value] if isinstance(value, str) else value
        if (
            not isinstance(references, list)
            or not references
            or any(not isinstance(item, str) or not item.strip() for item in references)
        ):
            raise ValueError("input_reference must contain non-empty string references")
        return references

    @field_serializer("input_reference")
    def serialize_input_reference(self, value: Optional[list[str]]):
        """Keep singleton strings for N-2 readers; arrays remain explicit."""
        if value is not None and len(value) == 1:
            return value[0]
        return value

    nvext: Optional[ImageNvExt] = None
    """NVIDIA extensions."""


class ImageData(BaseModel):
    """Individual image data in a response.

    Matches the flattened Rust Image enum in lib/protocols/src/types/mod.rs.
    """

    url: Optional[str] = None
    """URL of the generated image (if response_format is url)."""

    b64_json: Optional[str] = None
    """Base64-encoded image (if response_format is b64_json)."""

    revised_prompt: Optional[str] = None
    """Revised prompt, when the model rewrites the original prompt."""


class ImageAccounting(BaseModel):
    """Per-request accounting in image units.

    Matches Rust ImageAccounting in lib/llm/src/protocols/openai/images/nvext.rs,
    which states the rules.
    """

    output_images: int
    """Images generated and returned in ``data``, not the requested ``n``."""

    input_images: int = 0
    """Reference images the request consumed: 0 for text-to-image, 1 for an
    ``input_reference`` or ``/v1/images/edits`` request."""

    num_inference_steps: Optional[int] = None
    """The step count the backend ran. ``None`` when the backend does not know it."""


class ImagesResponseNvExt(BaseModel):
    """NVIDIA extensions to the image response.

    Matches Rust ImagesResponseNvExt in lib/llm/src/protocols/openai/images/nvext.rs.
    """

    usage: Optional[ImageAccounting] = None
    """Image-based accounting for models without token usage."""


class NvImagesResponse(BaseModel):
    """Response structure for image generation.

    Matches the flattened Rust NvImagesResponse in lib/llm/src/protocols/openai/images.rs
    """

    created: int
    """Unix timestamp of creation."""

    data: list[ImageData] = []
    """List of generated images."""

    background: Optional[str] = None
    """Background of the generation: transparent or opaque."""

    output_format: Optional[str] = None
    """Output format of the generated images: png, webp, or jpeg."""

    size: Optional[str] = None
    """Size of the generated images in WxH format."""

    quality: Optional[str] = None
    """Quality of the generated images: low, medium, or high."""

    usage: Optional[Dict[str, Any]] = None
    """Token usage of the generation, when the model reports it."""

    nvext: Optional[ImagesResponseNvExt] = None
    """NVIDIA response extensions. The frontend omits the key when unset."""
