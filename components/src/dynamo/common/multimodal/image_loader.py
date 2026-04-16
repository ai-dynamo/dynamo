# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import binascii
import logging
import os
from collections import OrderedDict
from io import BytesIO
from typing import Any, Dict, Final, List
from urllib.parse import urlparse

import httpx
from PIL import Image

import dynamo.nixl_connect as nixl_connect
from dynamo.common.utils import nvtx_utils as _nvtx
from dynamo.common.utils.media_nixl import read_decoded_media_via_nixl
from dynamo.common.utils.runtime import run_async

from .http_client import get_http_client
from .url_validator import (
    UrlValidationPolicy,
    fetch_with_revalidation,
    validate_media_url,
)

logger = logging.getLogger(__name__)

# Constants for multimodal data variants
URL_VARIANT_KEY: Final = "Url"
DECODED_VARIANT_KEY: Final = "Decoded"


class ImageLoader:
    CACHE_SIZE_MAXIMUM = int(os.environ.get("DYN_MM_IMAGE_CACHE_SIZE", "8"))

    def __init__(
        self,
        cache_size: int = CACHE_SIZE_MAXIMUM,
        http_timeout: float = 30.0,
        enable_frontend_decoding: bool = False,
        url_policy: UrlValidationPolicy | None = None,
    ):
        """Initialize the ImageLoader with caching, HTTP settings, and optional frontend NIXL RDMA decoding.

        ``url_policy`` defaults to ``UrlValidationPolicy.from_env()``.
        """
        self._http_timeout = http_timeout
        self._cache_size = cache_size
        self._image_cache: OrderedDict[str, Image.Image] = OrderedDict()
        self._inflight: dict[str, asyncio.Task[Image.Image]] = {}
        self._enable_frontend_decoding = enable_frontend_decoding
        self._url_policy = url_policy or UrlValidationPolicy.from_env()
        # Lazy-init NIXL connector only when frontend decoding is enabled
        self._nixl_connector = None
        if self._enable_frontend_decoding:
            self._nixl_connector = nixl_connect.Connector()
            run_async(
                self._nixl_connector.initialize
            )  # Synchronously wait for async init

    @staticmethod
    def _open_image_sync(image_data: BytesIO) -> Image.Image:
        """Open, validate, and decode an image from raw bytes. Runs in a thread."""
        image = Image.open(image_data, formats=["JPEG", "PNG", "WEBP"])
        if image.format not in ("JPEG", "PNG", "WEBP"):
            raise ValueError(f"Unsupported image format: {image.format}")
        # Image.open() is lazy — convert() forces the actual pixel decode
        return image.convert("RGB")

    @staticmethod
    async def _open_image(image_data: BytesIO) -> Image.Image:
        """Open and validate an image from raw bytes, converting to RGB."""
        with _nvtx.annotate("mm:img:pil_open_convert", color="lime"):
            return await asyncio.to_thread(ImageLoader._open_image_sync, image_data)

    def _cache_put(self, key: str, image: Image.Image) -> None:
        """Insert into cache if not already present. Sync — no awaits."""
        if key not in self._image_cache:
            if len(self._image_cache) >= self._cache_size:
                self._image_cache.popitem(last=False)
            self._image_cache[key] = image

    async def _fetch_and_process(self, image_url: str) -> Image.Image:
        """Fetch image via HTTP(S), decode with PIL, return RGB Image.

        All exception normalization happens here so shared callers
        see identical error types.
        """
        try:
            with _nvtx.annotate("mm:img:http_fetch", color="lime"):
                http_client = get_http_client(self._http_timeout)
                response = await fetch_with_revalidation(
                    http_client, image_url, self._url_policy
                )
                response.raise_for_status()
                if not response.content:
                    raise ValueError("Empty response content from image URL")
                image_data = BytesIO(response.content)

            return await self._open_image(image_data)

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP {e.response.status_code} loading image: '{image_url}'")
            raise
        except httpx.TimeoutException as e:
            logger.error(
                f"{type(e).__name__} loading image: '{image_url}' "
                f"(timeout={self._http_timeout}s)"
            )
            raise ValueError(f"Timeout loading image: '{image_url}'") from e
        except httpx.HTTPError as e:
            logger.error(f"{type(e).__name__} loading image: '{image_url}': {e}")
            raise
        except Exception as e:
            logger.error(f"{type(e).__name__} loading image: '{image_url}': {e}")
            raise ValueError(f"Failed to load image: '{image_url}': {e}") from e

    async def _fetch_and_cache(self, key: str, image_url: str) -> Image.Image:
        """Shared task: fetch, cache, then remove from _inflight."""
        try:
            image = await self._fetch_and_process(image_url)
            self._cache_put(key, image)
            return image
        finally:
            self._inflight.pop(key, None)

    @_nvtx.annotate("mm:img:load_image", color="lime")
    async def load_image(self, image_url: str) -> Image.Image:
        parsed_url = urlparse(image_url)
        if parsed_url.scheme in ("", "file"):
            raise ValueError(
                "Invalid image source scheme: local file access is not allowed"
            )
        normalized_url = validate_media_url(image_url, self._url_policy)
        parsed_url = urlparse(normalized_url)

        if parsed_url.scheme in ("http", "https"):
            key = normalized_url.lower()

            if key in self._image_cache:
                logger.debug(f"Image found in cache for URL: {image_url}")
                self._image_cache.move_to_end(key)
                return self._image_cache[key]

            if key not in self._inflight:
                task = asyncio.create_task(self._fetch_and_cache(key, normalized_url))
                # Suppress "exception was never retrieved" if all waiters cancel
                task.add_done_callback(
                    lambda t: t.exception() if not t.cancelled() else None
                )
                self._inflight[key] = task

            return await asyncio.shield(self._inflight[key])

        if parsed_url.scheme == "data":
            try:
                with _nvtx.annotate("mm:img:base64_decode", color="lime"):
                    if not parsed_url.path.startswith("image/"):
                        raise ValueError("Data URL must be an image type")

                    media_type, data = parsed_url.path.split(",", 1)
                    if ";base64" not in media_type:
                        raise ValueError("Data URL must be base64 encoded")

                    try:
                        image_bytes = base64.b64decode(data, validate=True)
                    except binascii.Error as e:
                        raise ValueError(f"Invalid base64 encoding: {e}") from e
                    image_data = BytesIO(image_bytes)
                return await self._open_image(image_data)
            except Exception as e:
                logger.error(f"{type(e).__name__} decoding image: '{image_url}': {e}")
                raise ValueError(f"Failed to decoding image: '{image_url}': {e}") from e

        # It's not file:, http:, https:, or data:
        raise ValueError(f"Invalid image source scheme: {parsed_url.scheme}")

    async def load_image_batch(
        self,
        image_mm_items: List[Dict[str, Any]],
    ) -> List[Any]:
        """
        Load a batch of images from multimodal data items.

        Supports two paths:
        1. Url variant: Download and decode image from URL (default)
        2. Decoded variant: Read pre-decoded image via NIXL RDMA (requires enable_frontend_decoding=True)

        Args:
            image_mm_items: List of multimodal data items for images

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
                if self._enable_frontend_decoding:
                    metadata = item[DECODED_VARIANT_KEY]
                    if self._nixl_connector is None:
                        raise RuntimeError("NIXL connector is not initialized")
                    image_futures.append(
                        read_decoded_media_via_nixl(self._nixl_connector, metadata)
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
