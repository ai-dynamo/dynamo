# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import binascii
import logging
from io import BytesIO
from typing import Optional
from urllib.parse import urlparse

import httpx
from PIL import Image

logger = logging.getLogger(__name__)

# Global HTTP client instance
_global_http_client: Optional[httpx.AsyncClient] = None


def get_http_client(timeout: float = 60.0) -> httpx.AsyncClient:
    """
    Get or create a shared HTTP client instance.

    Args:
        timeout: Timeout for HTTP requests

    Returns:
        Shared HTTP client instance
    """
    global _global_http_client

    if _global_http_client is None or _global_http_client.is_closed:
        _global_http_client = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
        )
        logger.info(f"Shared HTTP client initialized with timeout={timeout}s")

    return _global_http_client


class ImageLoader:
    CACHE_SIZE_MAXIMUM = 8

    def __init__(
        self, cache_size: int = CACHE_SIZE_MAXIMUM, http_timeout: float = 30.0
    ):
        self._http_timeout = http_timeout
        self._image_cache: dict[str, Image.Image] = {}
        self._cache_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=cache_size)

    async def load_image(self, image_url: str) -> Image.Image:
        parsed_url = urlparse(image_url)

        # For HTTP(S) URLs, check cache first
        if parsed_url.scheme in ("http", "https"):
            image_url_lower = image_url.lower()
            if image_url_lower in self._image_cache:
                logger.debug(f"Image found in cache for URL: {image_url}")
                return self._image_cache[image_url_lower]

        try:
            if parsed_url.scheme == "data":
                # Parse data URL format: data:[<media type>][;base64],<data>
                if not parsed_url.path.startswith("image/"):
                    raise ValueError("Data URL must be an image type")

                # Split the path into media type and data
                media_type, data = parsed_url.path.split(",", 1)
                if ";base64" not in media_type:
                    raise ValueError("Data URL must be base64 encoded")

                try:
                    image_bytes = base64.b64decode(data)
                    image_data = BytesIO(image_bytes)
                except binascii.Error as e:
                    raise ValueError(f"Invalid base64 encoding: {e}")
            elif parsed_url.scheme in ("http", "https"):
                http_client = get_http_client(self._http_timeout)

                response = await http_client.get(image_url)
                response.raise_for_status()

                if not response.content:
                    raise ValueError("Empty response content from image URL")

                image_data = BytesIO(response.content)
            else:
                raise ValueError(f"Invalid image source scheme: {parsed_url.scheme}")

            # PIL is sync, so offload to a thread to avoid blocking the event loop
            image = await asyncio.to_thread(Image.open, image_data)

            # Validate image format and convert to RGB
            if image.format not in ("JPEG", "PNG", "WEBP"):
                raise ValueError(f"Unsupported image format: {image.format}")

            image_converted = image.convert("RGB")

            # Cache HTTP(S) URLs
            if parsed_url.scheme in ("http", "https"):
                image_url_lower = image_url.lower()
                # Cache the image for future use, and evict the oldest image if the cache is full
                if self._cache_queue.full():
                    oldest_image_url = await self._cache_queue.get()
                    del self._image_cache[oldest_image_url]

                self._image_cache[image_url_lower] = image_converted
                await self._cache_queue.put(image_url_lower)

            return image_converted

        except httpx.HTTPError as e:
            logger.error(f"HTTP error loading image: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            raise ValueError(f"Failed to load image: {e}")
