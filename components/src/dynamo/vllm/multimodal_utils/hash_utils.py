# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import logging
from typing import Any, Sequence

logger = logging.getLogger(__name__)


def image_to_bytes(img: Any) -> bytes:
    """Convert a supported image object to PNG bytes for hashing."""
    from PIL import Image

    if isinstance(img, bytes):
        return img

    if isinstance(img, Image.Image):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    # Frontend-decoding can provide image tensors as numpy arrays.
    try:
        import numpy as np

        if isinstance(img, np.ndarray):
            pil_img = Image.fromarray(img)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            return buf.getvalue()
    except ImportError:
        pass

    raise TypeError(f"Unsupported image type for hashing: {type(img)}")


def compute_mm_uuids_from_images(images: Sequence[Any]) -> list[str]:
    """
    Compute SHA256 hex UUIDs for image inputs.
    """
    uuids: list[str] = []
    for img in images:
        raw_bytes = image_to_bytes(img)
        uuids.append(hashlib.sha256(raw_bytes).hexdigest())
    return uuids
