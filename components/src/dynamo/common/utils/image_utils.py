# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image utilities for image diffusion.

Provides helpers for encoding numpy images to PNG format.
"""

import io
import logging

import numpy as np

logger = logging.getLogger(__name__)


def encode_to_png_bytes(
    image: np.ndarray,
) -> bytes:
    """Encode numpy image to PNG bytes (in-memory).

    Args:
        image: Numpy image array of shape (H, W, 3) with uint8 values 0-255.

    Returns:
        PNG-encoded bytes.

    Raises:
        ImportError: If Pillow is not available.
        RuntimeError: If encoding fails.
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError(
            "Pillow is required for PNG encoding. " "Install with: pip install Pillow"
        ) from None

    logger.info(f"Encoding image of shape {image.shape} to PNG")

    try:
        img = Image.fromarray(image)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()
        logger.info(f"Encoded PNG image to {len(image_bytes)} bytes")
        return image_bytes

    except Exception as e:
        logger.error(f"Failed to encode image to PNG bytes: {e}")
        raise RuntimeError(f"PNG encoding failed: {e}") from e
