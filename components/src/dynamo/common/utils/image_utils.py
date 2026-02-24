# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image loading utilities for diffusion pipelines.

Resolves an input_reference string (base64 data URI, HTTP URL, or local path)
into a PIL Image suitable for passing to vLLM-Omni's multi_modal_data.
"""

import base64
import io
import logging
from urllib.parse import urlparse

import PIL.Image

logger = logging.getLogger(__name__)


async def load_image_from_reference(input_reference: str) -> PIL.Image.Image:
    """Load an image from a reference string.

    Supports:
    - Base64 data URI: ``data:image/png;base64,iVBOR...``
    - HTTP/HTTPS URL: ``https://example.com/image.png``
    - Local file path: ``/tmp/image.png``

    Args:
        input_reference: Image source string.

    Returns:
        PIL Image in RGB mode.

    Raises:
        ValueError: If the reference format is unrecognized or image cannot be loaded.
    """
    if input_reference.startswith("data:"):
        return _load_from_data_uri(input_reference)

    parsed = urlparse(input_reference)
    if parsed.scheme in ("http", "https"):
        return await _load_from_url(input_reference)

    return _load_from_path(input_reference)


def _load_from_data_uri(data_uri: str) -> PIL.Image.Image:
    """Decode a base64 data URI into a PIL Image."""
    try:
        # Format: data:image/png;base64,<data>
        header, encoded = data_uri.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        return PIL.Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to decode base64 data URI: {e}") from e


async def _load_from_url(url: str) -> PIL.Image.Image:
    """Download an image from a URL."""
    try:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    raise ValueError(
                        f"Failed to download image from {url}: HTTP {resp.status}"
                    )
                image_bytes = await resp.read()
        return PIL.Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except ImportError:
        # Fallback to synchronous requests if aiohttp not available
        import asyncio

        import requests

        resp = await asyncio.to_thread(requests.get, url, timeout=30)
        resp.raise_for_status()
        return PIL.Image.open(io.BytesIO(resp.content)).convert("RGB")


def _load_from_path(path: str) -> PIL.Image.Image:
    """Load an image from a local file path."""
    try:
        return PIL.Image.open(path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to load image from path '{path}': {e}") from e
