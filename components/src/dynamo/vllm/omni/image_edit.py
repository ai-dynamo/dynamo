# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model-aware admission and ordered loading for image edit references."""

import os
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING

from PIL import Image

from dynamo.common.http import HttpStatusError
from dynamo.common.http.url_validator import MediaSizeError

if TYPE_CHECKING:
    from dynamo.common.multimodal import ImageLoader

try:
    from vllm_omni.diffusion.model_metadata import get_diffusion_model_metadata
except ModuleNotFoundError as error:
    # Older/optional Omni installations may not publish capability metadata.
    # Missing transitive dependencies must still fail visibly.
    if error.name not in (
        "vllm_omni",
        "vllm_omni.diffusion",
        "vllm_omni.diffusion.model_metadata",
    ):
        raise
    get_diffusion_model_metadata = None


@dataclass(frozen=True)
class ImageEditLimits:
    images: int = 4
    pixels: int = 16_000_000
    total_pixels: int = 32_000_000

    @classmethod
    def from_env(cls) -> "ImageEditLimits":
        """Read deployment limits, rejecting empty, non-ASCII, or nonpositive values."""
        values = {}
        for field, name in (
            ("images", "DYN_IMAGE_EDIT_MAX_IMAGES"),
            ("pixels", "DYN_IMAGE_EDIT_MAX_PIXELS"),
            ("total_pixels", "DYN_IMAGE_EDIT_MAX_TOTAL_PIXELS"),
        ):
            raw = os.environ.get(name)
            if raw is None:
                continue
            if not raw.isascii() or not raw.isdecimal() or int(raw) < 1:
                raise ValueError(f"{name} must be a positive integer")
            values[field] = int(raw)
        return cls(**values)


def image_edit_capability(
    model_class_name: str | None, limits: ImageEditLimits
) -> dict:
    """Build version-1 discovery metadata from the loaded pipeline and limits."""
    return {
        "schema_version": 1,
        "max_input_images": image_edit_input_limit(
            model_class_name, operator_limit=limits.images
        ),
    }


def image_edit_input_limit(
    model_class_name: str | None, *, operator_limit: int = 4
) -> int:
    """Resolve an edit limit from the loaded pipeline, never the served alias.

    Unknown metadata preserves single-image edits but cannot authorize multiple
    references. ``supports_multimodal_inputs=False`` does not mean that single
    image editing is unsupported: Omni also uses it for single-image pipelines.
    """
    if (
        isinstance(operator_limit, bool)
        or not isinstance(operator_limit, Integral)
        or operator_limit < 1
    ):
        raise ValueError("Image edit operator limit must be a positive integer")
    if get_diffusion_model_metadata is None:
        return 1

    metadata = get_diffusion_model_metadata(model_class_name)
    # This optional metadata field is absent in older Omni releases. A video
    # pipeline's reference-image limit does not establish multi-image editing
    # support, nor does its default output prove single-image edits unsupported.
    output_type = getattr(metadata, "final_output_type", None)
    if output_type not in (None, "image"):
        return 1
    if getattr(metadata, "supports_multimodal_inputs", None) is not True:
        return 1
    model_limit = getattr(metadata, "max_multimodal_image_inputs", None)
    if (
        isinstance(model_limit, bool)
        or not isinstance(model_limit, Integral)
        or model_limit < 1
    ):
        return 1
    return min(int(model_limit), int(operator_limit))


async def load_image_edit_references(
    references: list[str],
    *,
    image_loader: "ImageLoader",
    model_class_name: str | None,
    operator_limit: int = 4,
    max_total_pixels: int = 32_000_000,
) -> list[Image.Image]:
    """Check count before I/O, then decode every reference in request order.

    ImageLoader can share cached images across requests, so never close its
    results on failure. Release the partial list instead. The loader owns URL,
    byte-size, format and decompression policies; do not bypass it for uploads.
    """
    if not references:
        raise ValueError("At least one image edit reference is required")
    limit = image_edit_input_limit(model_class_name, operator_limit=operator_limit)
    if len(references) > limit:
        raise ValueError(
            f"Image edits support at most {limit} input image(s) for this deployment; "
            f"received {len(references)}"
        )

    for index, reference in enumerate(references):
        if not isinstance(reference, str) or not reference.strip():
            field = (
                "input_reference"
                if len(references) == 1
                else f"input_reference[{index}]"
            )
            raise ValueError(f"Failed to load {field}")

    images: list[Image.Image] = []
    total_pixels = 0
    try:
        for index, reference in enumerate(references):
            field = (
                "input_reference"
                if len(references) == 1
                else f"input_reference[{index}]"
            )
            try:
                image = await image_loader.load_image(
                    reference, max_pixels=max_total_pixels - total_pixels
                )
                total_pixels += image.width * image.height
                if total_pixels > max_total_pixels:
                    del image
                    raise HttpStatusError(
                        413, "Image edit exceeds aggregate decoded pixels", "image edit"
                    )
                images.append(image)
                del image
            except MediaSizeError as error:
                raise HttpStatusError(
                    413,
                    f"{field} exceeds the configured media byte limit",
                    "image edit",
                ) from error
            except Image.DecompressionBombError as error:
                raise HttpStatusError(
                    413, f"{field} exceeds the decoded pixel limit", "image edit"
                ) from error
            except HttpStatusError as error:
                # Preserve status without forwarding origin messages or signed URLs.
                detail = (
                    "exceeds a configured image resource limit"
                    if error.status == 413
                    else "could not be loaded"
                )
                raise HttpStatusError(
                    error.status, f"{field} {detail}", "image edit"
                ) from error
            except ValueError as error:
                # Preserve the legacy single-reference error and keep payloads
                # and signed URLs out of multi-reference errors as well.
                field = (
                    "input_reference"
                    if len(references) == 1
                    else f"input_reference[{index}]"
                )
                raise ValueError(f"Failed to load {field}") from error
    except BaseException:
        # The partial list must not retain decoded data through the traceback.
        # Cache entries may still be in use by another request.
        images.clear()
        raise
    return images
