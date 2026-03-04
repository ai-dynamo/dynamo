# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import base64
import binascii
import logging
import os
from io import BytesIO
from typing import Any, Dict, Final, List, Optional
from urllib.parse import urlparse

import httpx
from PIL import Image

import dynamo.nixl_connect as nixl_connect
from dynamo.common.utils.media_nixl import read_decoded_media_via_nixl

from .http_client import get_http_client

logger = logging.getLogger(__name__)

# Constants for multimodal data variants
URL_VARIANT_KEY: Final = "Url"
DECODED_VARIANT_KEY: Final = "Decoded"


class ImageLoader:
    CACHE_SIZE_MAXIMUM = int(os.environ.get("DYN_MM_IMAGE_LOADER_CACHE_SIZE", 8))

    def __init__(
        self, cache_size: int = CACHE_SIZE_MAXIMUM, http_timeout: float = 30.0
    ):
        self._http_timeout = http_timeout
        self._image_cache: dict[str, tuple[Image.Image, bytes]] = {}
        self._cache_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=cache_size)
        self._light_cache: dict[str, tuple[bytes, int, int]] = {}

    async def load_image(self, image_url: str) -> Image.Image:
        """Load image from URL, returning PIL Image only (backward compatible)."""
        image, _ = await self._fetch_image(image_url)
        return image

    async def load_image_with_raw_bytes(
        self, image_url: str
    ) -> tuple[Image.Image, bytes]:
        """Load image from URL, returning both PIL Image and original raw bytes."""
        return await self._fetch_image(image_url)

    async def load_image_light(
        self, image_url: str
    ) -> tuple[bytes, int, int]:
        """Load image returning (raw_bytes, width, height) without convert("RGB").

        Uses PIL lazy open (header-only) to get dimensions. Much faster than
        full image decode — suitable when only raw bytes and dimensions are needed
        (e.g., MM Router hashing and token expansion).

        HTTP URLs are cached (raw bytes + dimensions) for repeated access.
        """
        parsed_url = urlparse(image_url)

        # Check light cache for HTTP URLs
        if parsed_url.scheme in ("http", "https"):
            image_url_lower = image_url.lower()
            if image_url_lower in self._light_cache:
                logger.debug(f"Light cache hit for URL: {image_url[:80]}")
                return self._light_cache[image_url_lower]

        raw_bytes = await self._fetch_raw_bytes(image_url)
        image_data = BytesIO(raw_bytes)
        image = await asyncio.to_thread(
            Image.open, image_data, formats=["JPEG", "PNG", "WEBP"]
        )
        result = (raw_bytes, image.width, image.height)

        # Cache HTTP URLs
        if parsed_url.scheme in ("http", "https"):
            image_url_lower = image_url.lower()
            if len(self._light_cache) >= self.CACHE_SIZE_MAXIMUM:
                # Evict oldest entry
                oldest = next(iter(self._light_cache))
                del self._light_cache[oldest]
            self._light_cache[image_url_lower] = result

        return result

    async def _fetch_raw_bytes(self, image_url: str) -> bytes:
        """Internal: fetch raw image bytes from URL or data URI."""
        parsed_url = urlparse(image_url)

        try:
            if parsed_url.scheme == "data":
                if not parsed_url.path.startswith("image/"):
                    raise ValueError("Data URL must be an image type")
                media_type, data = parsed_url.path.split(",", 1)
                if ";base64" not in media_type:
                    raise ValueError("Data URL must be base64 encoded")
                try:
                    return base64.b64decode(data)
                except binascii.Error as e:
                    raise ValueError(f"Invalid base64 encoding: {e}")
            elif parsed_url.scheme in ("http", "https"):
                http_client = get_http_client(self._http_timeout)
                response = await http_client.get(image_url)
                response.raise_for_status()
                if not response.content:
                    raise ValueError("Empty response content from image URL")
                return response.content
            else:
                raise ValueError(f"Invalid image source scheme: {parsed_url.scheme}")
        except httpx.HTTPError as e:
            logger.error(f"HTTP error loading image: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            raise ValueError(f"Failed to load image: {e}")

    async def _fetch_image(self, image_url: str) -> tuple[Image.Image, bytes]:
        """Internal: fetch image and return (PIL Image, raw bytes)."""
        parsed_url = urlparse(image_url)

        # For HTTP(S) URLs, check cache first
        if parsed_url.scheme in ("http", "https"):
            image_url_lower = image_url.lower()
            if image_url_lower in self._image_cache:
                logger.debug(f"Image found in cache for URL: {image_url}")
                return self._image_cache[image_url_lower]

        raw_bytes = await self._fetch_raw_bytes(image_url)

        image_data = BytesIO(raw_bytes)

        # PIL is sync, so offload to a thread to avoid blocking the event loop
        # Restrict to supported formats to prevent PSD parsing (GHSA-cfh3-3jmp-rvhc)
        image = await asyncio.to_thread(
            Image.open, image_data, formats=["JPEG", "PNG", "WEBP"]
        )

        # Validate image format and convert to RGB
        if image.format not in ("JPEG", "PNG", "WEBP"):
            raise ValueError(f"Unsupported image format: {image.format}")

        image_converted = image.convert("RGB")

        # Cache HTTP(S) URLs
        if parsed_url.scheme in ("http", "https"):
            image_url_lower = image_url.lower()
            if self._cache_queue.full():
                oldest_image_url = await self._cache_queue.get()
                del self._image_cache[oldest_image_url]

            self._image_cache[image_url_lower] = (image_converted, raw_bytes)
            await self._cache_queue.put(image_url_lower)

        return (image_converted, raw_bytes)

    async def load_image_batch(
        self,
        image_mm_items: List[Dict[str, Any]],
        enable_frontend_decoding: bool = False,
        nixl_connector: Optional["nixl_connect.Connector"] = None,
    ) -> List[Any]:
        """
        Load a batch of images from multimodal data items.

        Supports two paths:
        1. Url variant: Download and decode image from URL (default)
        2. Decoded variant: Read pre-decoded image via NIXL RDMA (requires enable_frontend_decoding=True)

        Args:
            image_mm_items: List of multimodal data items for images
            enable_frontend_decoding: If True, enables NIXL RDMA for decoded images
            nixl_connector: NIXL connector for frontend decoding (required if enable_frontend_decoding=True)

        Returns:
            List of loaded image data

        Raises:
            Exception: If any image fails to load
            ValueError: If enable_frontend_decoding=True but nixl_connector is None
        """
        image_futures = []

        for item in image_mm_items:
            if isinstance(item, dict) and URL_VARIANT_KEY in item:
                # URL path: download and decode in Python backend
                url = item[URL_VARIANT_KEY]
                image_futures.append(self.load_image(url))
                logger.debug(f"Preparing to load image from URL: {url[:80]}...")
            elif isinstance(item, dict) and DECODED_VARIANT_KEY in item:
                if enable_frontend_decoding:
                    if nixl_connector is None:
                        logger.error(
                            "Frontend decoding enabled but nixl_connector not provided. "
                            "Caller must pass an initialized NIXL connector."
                        )
                        raise ValueError(
                            "nixl_connector required when enable_frontend_decoding=True"
                        )

                    metadata = item[DECODED_VARIANT_KEY]
                    image_futures.append(
                        read_decoded_media_via_nixl(nixl_connector, metadata)
                    )
                else:
                    logger.error(
                        "Received Decoded multimodal data but enable_frontend_decoding=False. "
                        "Set enable_frontend_decoding=True to enable NIXL RDMA image transfer."
                    )
                    raise ValueError("Could not load decoded media from frontend")

        # Process images in parallel
        results = await asyncio.gather(*image_futures, return_exceptions=True)
        loaded_images = []
        collective_exceptions = ""
        for media_item, result in zip(image_mm_items, results):
            if isinstance(result, Exception):
                source = media_item.get(URL_VARIANT_KEY, "decoded")
                logger.error(f"Failed to load image from {source[:80]}...: {result}")
                collective_exceptions += (
                    f"Failed to load image from {source[:80]}...: {result}\n"
                )
                continue
            loaded_images.append(result)

        if collective_exceptions:
            raise Exception(collective_exceptions)

        return loaded_images

    async def load_image_batch_with_raw_bytes(
        self,
        image_mm_items: List[Dict[str, Any]],
    ) -> tuple[List[Any], List[bytes]]:
        """Load a batch of images, returning both PIL Images and raw bytes.

        Returns:
            Tuple of (images, raw_bytes_list)
        """
        image_futures = []

        for item in image_mm_items:
            if isinstance(item, dict) and URL_VARIANT_KEY in item:
                url = item[URL_VARIANT_KEY]
                image_futures.append(self.load_image_with_raw_bytes(url))

        results = await asyncio.gather(*image_futures, return_exceptions=True)
        images = []
        raw_bytes_list = []
        collective_exceptions = ""
        for media_item, result in zip(image_mm_items, results):
            if isinstance(result, Exception):
                source = media_item.get(URL_VARIANT_KEY, "unknown")
                logger.error(f"Failed to load image from {source[:80]}...: {result}")
                collective_exceptions += (
                    f"Failed to load image from {source[:80]}...: {result}\n"
                )
                continue
            img, raw = result
            images.append(img)
            raw_bytes_list.append(raw)

        if collective_exceptions:
            raise Exception(collective_exceptions)

        return images, raw_bytes_list
