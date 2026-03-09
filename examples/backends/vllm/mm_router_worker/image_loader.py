# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Lightweight image loader for MM Router routing decisions.

Only extracts raw bytes and dimensions (width, height) — does NOT do full
PIL decode or convert("RGB"), since the Router never needs pixel data.
Supports async batch loading for parallel HTTP downloads.
"""

import asyncio
import base64
import logging
import os
from collections import deque
from io import BytesIO
from urllib.parse import urlparse

from PIL import Image

from dynamo.common.multimodal.http_client import get_http_client

logger = logging.getLogger(__name__)

# Result type: (raw_bytes, width, height)
ImageMeta = tuple[bytes, int, int]


class RoutingImageLoader:
    """Async image loader optimized for routing: raw bytes + dimensions only."""

    def __init__(
        self,
        cache_size: int = int(os.environ.get("DYN_MM_IMAGE_LOADER_CACHE_SIZE", "64")),
        http_timeout: float = 30.0,
    ):
        self._http_timeout = http_timeout
        self._cache: dict[str, ImageMeta] = {}
        self._cache_order: deque[str] = deque()
        self._cache_size = cache_size

    # ---- Public API ----

    async def load(self, url: str) -> ImageMeta:
        """Load a single image, returning (raw_bytes, width, height)."""
        # Cache lookup (HTTP URLs only)
        cached = self._cache.get(url)
        if cached is not None:
            return cached

        raw_bytes = await self._fetch_raw_bytes(url)
        w, h = _parse_dimensions(raw_bytes)
        result = (raw_bytes, w, h)

        # Cache HTTP URLs
        parsed = urlparse(url)
        if parsed.scheme in ("http", "https"):
            self._put_cache(url, result)

        return result

    async def load_batch(self, urls: list[str]) -> list[ImageMeta]:
        """Load multiple images concurrently."""
        if len(urls) <= 1:
            return [await self.load(urls[0])] if urls else []
        return list(await asyncio.gather(*[self.load(url) for url in urls]))

    # ---- Internal ----

    async def _fetch_raw_bytes(self, url: str) -> bytes:
        parsed = urlparse(url)

        if parsed.scheme == "data":
            # data:image/png;base64,<data>
            _, data = parsed.path.split(",", 1)
            return base64.b64decode(data)

        if parsed.scheme in ("http", "https"):
            client = get_http_client(self._http_timeout)
            response = await client.get(url)
            response.raise_for_status()
            if not response.content:
                raise ValueError("Empty response from image URL")
            return response.content

        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")

    def _put_cache(self, key: str, value: ImageMeta) -> None:
        if key in self._cache:
            return
        if len(self._cache) >= self._cache_size:
            oldest = self._cache_order.popleft()
            self._cache.pop(oldest, None)
        self._cache[key] = value
        self._cache_order.append(key)


def _parse_dimensions(raw_bytes: bytes) -> tuple[int, int]:
    """Get (width, height) from image bytes using PIL lazy open (header only)."""
    img = Image.open(BytesIO(raw_bytes))  # Image.open() is lazy and this is fast
    return img.size
