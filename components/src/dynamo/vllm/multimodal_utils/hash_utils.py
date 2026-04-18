# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, Sequence

import blake3
import numpy as np

logger = logging.getLogger(__name__)


def image_to_bytes(img: Any) -> bytes:
    """Serialize an image to bytes for hashing.

    The output embeds shape and mode/dtype in a header ahead of the raw pixel
    buffer. Without the header, two images sharing the same pixel product but
    different dimensions (e.g. 30x150 vs 150x30 RGB uint8) emit identical
    ``tobytes()`` output and collide under Blake3, allowing one request's
    cached embedding to be served for another — a cache-poisoning vector.
    """
    from PIL import Image

    if isinstance(img, bytes):
        return img

    if isinstance(img, Image.Image):
        header = f"PIL:{img.mode}:{img.size[0]}x{img.size[1]}:".encode()
        return header + img.tobytes()

    if isinstance(img, np.ndarray):
        header = f"NDA:{img.dtype}:{img.shape}:".encode()
        return header + img.tobytes()

    raise TypeError(f"Unsupported image type for hashing: {type(img)}")


def compute_mm_uuids_from_images(images: Sequence[Any]) -> list[str]:
    """
    Compute blake3 hex UUIDs for image inputs.
    """
    uuids: list[str] = []
    for img in images:
        raw_bytes = image_to_bytes(img)
        uuids.append(blake3.blake3(raw_bytes).hexdigest())
    return uuids
