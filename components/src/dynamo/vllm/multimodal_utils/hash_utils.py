# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import struct
from typing import Any, Sequence

import blake3
import numpy as np

logger = logging.getLogger(__name__)

# Preimage layout (12 bytes, fixed-length) || pixel bytes.
#   offset  bytes  field
#     0      1     version_byte   currently 0x01; bump to rotate preimage format
#     1      1     mode_tag       0x00 = RGB
#     2      1     dtype_tag      0x00 = uint8
#     3      1     channels       currently 3
#     4      4     height         u32 little-endian
#     8      4     width          u32 little-endian
#
# The fixed-length header prevents delimiter-ambiguity collisions: no two
# distinct (header, pixel) pairs can produce the same preimage.
_HEADER_STRUCT = struct.Struct("<BBBBII")
_VERSION_BYTE = 0x01
_MODE_RGB = 0x00
_DTYPE_UINT8 = 0x00
_CHANNELS_RGB = 3


def _header(height: int, width: int) -> bytes:
    return _HEADER_STRUCT.pack(
        _VERSION_BYTE, _MODE_RGB, _DTYPE_UINT8, _CHANNELS_RGB, height, width
    )


def _image_preimage_parts(img: Any) -> tuple[bytes, bytes]:
    """Return `(header_bytes, pixel_bytes)` for a canonicalized image.

    The returned pair is safe to feed into an incremental blake3 hasher. Two
    images that differ only in (H, W) produce different `header_bytes` and
    therefore different digests.

    Raises:
        ValueError: input shape, dtype, or mode violates the RGB uint8 contract.
        TypeError: input is neither a PIL.Image.Image nor an np.ndarray.
    """
    from PIL import Image

    if isinstance(img, Image.Image):
        if img.mode != "RGB":
            raise ValueError(
                f"compute_mm_uuids_from_images expected RGB mode, got {img.mode!r}"
            )
        width, height = img.size
        return _header(height, width), img.tobytes()

    if isinstance(img, np.ndarray):
        if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != _CHANNELS_RGB:
            raise ValueError(
                "compute_mm_uuids_from_images expected dtype=uint8 and shape "
                f"(H, W, 3), got dtype={img.dtype} shape={img.shape}"
            )
        contiguous = np.ascontiguousarray(img)
        height, width, _ = contiguous.shape
        return _header(height, width), contiguous.tobytes()

    raise TypeError(
        "compute_mm_uuids_from_images expected PIL.Image.Image or np.ndarray, "
        f"got {type(img).__name__}"
    )


def compute_mm_uuids_from_images(images: Sequence[Any]) -> list[str]:
    """Compute blake3 hex UUIDs for image inputs.

    Each preimage is a fixed-length header (version, mode, dtype, channels,
    height, width) followed by the raw RGB uint8 pixel bytes. Including
    geometry in the preimage prevents two different-shape images with equal
    pixel count from colliding on the same cache key.
    """
    uuids: list[str] = []
    for img in images:
        header, pixels = _image_preimage_parts(img)
        h = blake3.blake3()
        h.update(header)
        h.update(pixels)
        uuids.append(h.hexdigest())
    return uuids
