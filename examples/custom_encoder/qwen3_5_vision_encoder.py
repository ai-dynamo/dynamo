# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small Qwen3.5 implementation of Dynamo's custom vision-encoder contract.

This is a teaching example, not a performance reference. It favors a short,
readable implementation over fast checkpoint loading, caching, CUDA graphs, or
production media handling. In particular, ``build`` temporarily loads the full
checkpoint before retaining only the vision tower. ``preprocess`` accepts only
bounded inline image data so the example never performs server-side URL fetches.
Production backends should add the policies and optimizations their workload
requires.
"""

from __future__ import annotations

import base64
import binascii
import gc
import io
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from dynamo.vllm.multimodal_utils.custom_encoder import (
    Preprocessed,
    Qwen3VLImageEncoding,
    VisionEncoderBackend,
)

_MAX_IMAGE_BYTES = 20 * 1024 * 1024
_MAX_BASE64_CHARS = 4 * ((_MAX_IMAGE_BYTES + 2) // 3)
_MAX_IMAGE_PIXELS = 16_777_216


def _decode_image_data_url(raw: str) -> bytes:
    """Decode one bounded base64 image data URL without network access."""
    header, separator, encoded = raw.partition(",")
    media_type, *parameters = header[5:].split(";")
    if (
        not separator
        or header[:5].lower() != "data:"
        or not media_type.lower().startswith("image/")
        or "base64" not in {parameter.lower() for parameter in parameters}
    ):
        raise ValueError("Expected an inline base64 image data URL")
    if len(encoded) > _MAX_BASE64_CHARS:
        raise ValueError(f"Inline image exceeds {_MAX_IMAGE_BYTES} bytes")

    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Inline image contains invalid base64 data") from exc
    if len(image_bytes) > _MAX_IMAGE_BYTES:
        raise ValueError(f"Inline image exceeds {_MAX_IMAGE_BYTES} bytes")
    return image_bytes


def _open_bounded_image(image_bytes: bytes) -> Image.Image:
    """Decode one RGB image without exceeding the processor's pixel ceiling."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            width, height = source.size
            if width < 1 or height < 1 or width * height > _MAX_IMAGE_PIXELS:
                raise ValueError(
                    f"Inline image exceeds {_MAX_IMAGE_PIXELS} decoded pixels"
                )
            return source.convert("RGB")
    except (Image.DecompressionBombError, OSError) as exc:
        raise ValueError("Inline image could not be decoded") from exc


@dataclass(frozen=True)
class Qwen35ImageInputs:
    """CPU processor output for one image."""

    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor


class Qwen35VisionEncoder(
    VisionEncoderBackend[str, Qwen35ImageInputs, Qwen3VLImageEncoding]
):
    """Run the Qwen3.5 vision tower as a minimal custom-encoder example.

    The class demonstrates the four lifecycle hooks and native ``TokensPrompt``
    artifact shape. This implementation reuses the checkpoint's vision tower,
    but users can bring their own vision encoder and projector by replacing the
    loading and forward paths. The replacement must return projected rows and
    grid metadata compatible with the running Qwen3.5 decoder.
    """

    preprocess_concurrency = 4

    def __init__(self) -> None:
        self._device = torch.device("cpu")
        self._processor: Any | None = None
        self._visual: Any | None = None

    def build(self, model_id: str) -> None:
        """Load the processor and retain only the checkpoint's vision tower."""
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = AutoProcessor.from_pretrained(model_id)
        # This teaching backend retains Qwen3.5's tower. A custom backend can
        # load its own encoder and projector here instead.
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        self._visual = model.model.visual.eval().to(self._device)
        model.model.visual = None
        del model
        gc.collect()

    def preprocess(self, raw: str) -> Preprocessed[Qwen35ImageInputs]:
        """Decode and patchify one bounded inline image on a preprocessing thread."""
        if self._processor is None:
            raise RuntimeError("Qwen35VisionEncoder is not loaded")

        image = _open_bounded_image(_decode_image_data_url(raw))
        try:
            inputs = self._processor.image_processor(
                images=[image], return_tensors="pt"
            )
        finally:
            image.close()
        return Preprocessed(
            Qwen35ImageInputs(
                pixel_values=inputs["pixel_values"].contiguous(),
                image_grid_thw=inputs["image_grid_thw"].to(dtype=torch.long),
            )
        )

    def forward_batch(
        self,
        items: list[Qwen35ImageInputs],
        target_bucket: int | None = None,
    ) -> list[Qwen3VLImageEncoding]:
        """Encode a Dynamo-formed batch and restore one artifact per image."""
        del target_bucket
        if self._visual is None:
            raise RuntimeError("Qwen35VisionEncoder is not loaded")

        pixel_values = torch.cat([item.pixel_values for item in items], dim=0).to(
            device=self._device,
            dtype=self._visual.dtype,
        )
        image_grid_thw_cpu = torch.cat([item.image_grid_thw for item in items], dim=0)
        image_grid_thw = image_grid_thw_cpu.to(self._device)

        with torch.inference_mode():
            vision_output = self._visual(
                pixel_values,
                grid_thw=image_grid_thw,
                return_dict=True,
            )

        embeddings = vision_output.pooler_output.to(
            device="cpu", dtype=torch.bfloat16
        ).contiguous()
        split_sizes = (
            image_grid_thw_cpu.prod(dim=-1) // self._visual.spatial_merge_size**2
        ).tolist()

        return [
            Qwen3VLImageEncoding(
                embeddings=rows,
                grid_thw=(int(grid[0]), int(grid[1]), int(grid[2])),
            )
            for rows, grid in zip(
                torch.split(embeddings, split_sizes),
                image_grid_thw_cpu.tolist(),
                strict=True,
            )
        ]

    def close(self) -> None:
        """Release resources created by ``build``."""
        self._processor = None
        self._visual = None
