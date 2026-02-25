# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
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


def compute_mm_uuids_from_images(images: Sequence[Any]) -> list[str]:
    """
    Compute blake3 hex UUIDs for image inputs.
    """
    uuids: list[str] = []
    for img in images:
        raw_bytes = image_to_bytes(img)
        uuids.append(blake3.blake3(raw_bytes).hexdigest())
    return uuids
