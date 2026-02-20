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
