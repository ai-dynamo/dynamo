# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Custom vLLM MediaConnector that wraps dynamo's ImageLoader.

Register with vLLM by setting VLLM_MEDIA_CONNECTOR=dynamo before starting
the frontend. The renderer will use dynamo's ImageLoader (with LRU cache +
in-flight dedup) for all image fetching, eliminating data URI encoding and
double downloads.

Usage:
    # Import this module to register the connector, then set the env var:
    import dynamo.common.multimodal.media_connector  # noqa: F401
    os.environ["VLLM_MEDIA_CONNECTOR"] = "dynamo"
"""

import logging

from PIL import Image

from dynamo.common.multimodal.image_loader import ImageLoader

logger = logging.getLogger(__name__)

# Shared ImageLoader singleton — survives across connector instances.
# The MEDIA_CONNECTOR_REGISTRY creates a new connector per request,
# so the cache must live outside the connector.
_shared_image_loader: ImageLoader | None = None


def _get_shared_image_loader() -> ImageLoader:
    global _shared_image_loader
    if _shared_image_loader is None:
        _shared_image_loader = ImageLoader()
        logger.info(
            "DynamoMediaConnector: shared ImageLoader initialized (cache_size=%d)",
            _shared_image_loader._cache_size,
        )
    return _shared_image_loader


# Lazy registration — only register if vLLM is available.
try:
    from vllm.multimodal.media import MEDIA_CONNECTOR_REGISTRY, MediaConnector

    @MEDIA_CONNECTOR_REGISTRY.register("dynamo")
    class DynamoMediaConnector(MediaConnector):
        """MediaConnector that uses dynamo's ImageLoader for image fetching.

        Benefits over the default HTTP connector:
        - In-memory LRU cache (configurable via DYN_MM_IMAGE_CACHE_SIZE)
        - In-flight request deduplication (concurrent requests for the same
          URL share a single fetch task)
        - Async-native with proper cancellation handling
        """

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._image_loader = _get_shared_image_loader()

        def fetch_image(
            self,
            image_url: str,
            *,
            image_mode: str = "RGB",
        ) -> Image.Image:
            """Synchronous image fetch with cache lookup.

            ImageLoader is async-only for HTTP fetches, so we can't call it
            here. But we can check its LRU cache (sync-safe) and return a
            hit without re-downloading. On miss, delegate to the parent's
            sync implementation.
            """
            cached = self._image_loader.get_cached_sync(image_url.lower(), image_mode)
            if cached is not None:
                return cached
            return super().fetch_image(image_url, image_mode=image_mode)

        async def fetch_image_async(
            self,
            image_url: str,
            *,
            image_mode: str = "RGB",
        ) -> Image.Image:
            """Async image fetch via dynamo's ImageLoader with LRU cache."""
            try:
                img = await self._image_loader.load_image(image_url)
            except (ValueError, FileNotFoundError, OSError) as exc:
                # Fall back to parent for unsupported URL schemes or local
                # file paths that ImageLoader doesn't handle.
                logger.debug(
                    "DynamoMediaConnector: falling back to parent for %s (%s)",
                    image_url[:80],
                    exc,
                )
                return await super().fetch_image_async(image_url, image_mode=image_mode)
            if image_mode != "RGB" and img.mode != image_mode:
                img = img.convert(image_mode)
            return img

    logger.debug("Registered 'dynamo' MediaConnector with vLLM")

except ImportError:
    logger.debug("vLLM not available, skipping DynamoMediaConnector registration")
