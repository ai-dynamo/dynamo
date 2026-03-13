# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, Sequence

import blake3
import numpy as np

logger = logging.getLogger(__name__)


def image_to_bytes(img: Any) -> bytes:
    """Convert a supported image object to PNG bytes for hashing."""
    from PIL import Image

    if isinstance(img, bytes):
        return img

    if isinstance(img, Image.Image | np.ndarray):
        return img.tobytes()

    raise TypeError(f"Unsupported image type for hashing: {type(img)}")


def unwrap_pil_image(img: Any) -> Any:
    """Unwrap vLLM MediaWithBytes to extract the inner PIL Image.

    vLLM's preprocessing wraps downloaded images in MediaWithBytes(media,
    original_bytes).  image_to_bytes checks isinstance(img, Image.Image)
    but MediaWithBytes is a plain dataclass — not a PIL subclass — so it
    would raise TypeError instead of hashing the pixel data.

    The backend's _compute_mm_uuids() in handlers.py receives actual PIL
    images from image_loader and hashes img.tobytes() (pixel data).  For
    the routing mm_hash to match the KV-event mm_hash, callers must unwrap
    before passing to compute_mm_uuids_from_images.
    """
    if hasattr(img, "media"):
        return img.media
    return img


def compute_mm_uuids_from_images(images: Sequence[Any]) -> list[str]:
    """Compute blake3 hex UUIDs for image inputs."""
    uuids: list[str] = []
    for img in images:
        raw_bytes = image_to_bytes(img)
        uuids.append(blake3.blake3(raw_bytes).hexdigest())
    return uuids


def find_image_token_ranges(
    tokens: list[int], image_token_id: int
) -> list[tuple[int, int]]:
    """Find contiguous runs of image_token_id in a token sequence."""
    ranges = []
    start = None
    for i, t in enumerate(tokens):
        if t == image_token_id:
            if start is None:
                start = i
        elif start is not None:
            ranges.append((start, i))
            start = None
    if start is not None:
        ranges.append((start, len(tokens)))
    return ranges


def build_block_mm_infos(
    num_tokens: int,
    block_size: int,
    mm_hashes: list[int],
    image_ranges: list[tuple[int, int]],
) -> list[dict | None]:
    """Build per-block mm_info list for MM-aware KV routing."""
    num_blocks = (num_tokens + block_size - 1) // block_size
    result = []
    for block_idx in range(num_blocks):
        block_start = block_idx * block_size
        block_end = block_start + block_size
        mm_objects = [
            {"mm_hash": mm_hash, "offsets": []}
            for mm_hash, (img_start, img_end) in zip(mm_hashes, image_ranges)
            if block_end > img_start and block_start <= img_end
        ]
        result.append({"mm_objects": mm_objects} if mm_objects else None)
    return result
