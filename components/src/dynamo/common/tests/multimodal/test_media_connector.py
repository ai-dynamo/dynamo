# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DynamoMediaConnector and its ImageLoader integration."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

import dynamo.common.multimodal.media_connector as media_connector
from dynamo.common.http import HttpStatusError
from dynamo.common.http.url_validator import UrlValidationError
from dynamo.common.multimodal.image_loader import ImageLoader

_HAS_CONNECTOR = hasattr(media_connector, "DynamoMediaConnector")


def _make_connector():
    if not _HAS_CONNECTOR:
        pytest.skip("DynamoMediaConnector requires vllm")
    return media_connector.DynamoMediaConnector.__new__(
        media_connector.DynamoMediaConnector
    )

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _make_pil_image() -> Image.Image:
    return Image.new("RGB", (4, 4), color="blue")


class TestImageLoaderCache:
    """Test the ImageLoader LRU cache used by DynamoMediaConnector."""

    def test_cache_put_and_get(self):
        """ImageLoader caches images by URL key."""
        loader = ImageLoader()
        img = _make_pil_image()
        url = "http://example.com/test.jpg"

        loader._cache_put(url.lower(), img)
        assert url.lower() in loader._image_cache
        assert loader._image_cache[url.lower()] is img

    def test_cache_eviction(self):
        """Oldest entry is evicted when cache is full."""
        loader = ImageLoader(cache_size=2)

        img1 = _make_pil_image()
        img2 = _make_pil_image()
        img3 = _make_pil_image()

        loader._cache_put("url1", img1)
        loader._cache_put("url2", img2)
        assert len(loader._image_cache) == 2

        loader._cache_put("url3", img3)
        assert len(loader._image_cache) == 2
        assert "url1" not in loader._image_cache  # evicted
        assert "url3" in loader._image_cache

    def test_cache_no_duplicate(self):
        """Putting the same key twice doesn't create duplicates."""
        loader = ImageLoader(cache_size=2)
        img = _make_pil_image()

        loader._cache_put("url1", img)
        loader._cache_put("url1", img)
        assert len(loader._image_cache) == 1


class TestClientErrorPassthrough:
    """The connector must not downgrade a client error into a server error.

    ImageLoader already decides that a URL is unfetchable and raises a typed
    error carrying that verdict. Falling back to vLLM's fetcher re-fetches the
    same dead URL and re-raises as an aiohttp OSError, which the PyO3 bridge
    classifies as Unknown -> 500. The typed error must survive instead.
    """

    @pytest.mark.asyncio
    async def test_url_validation_error_is_not_swallowed(self):
        connector = _make_connector()
        connector._image_loader = SimpleNamespace(
            load_image=AsyncMock(
                side_effect=UrlValidationError(
                    "Could not resolve host 'nonexistent.invalid'"
                )
            )
        )

        with pytest.raises(UrlValidationError):
            await connector.fetch_image_async("https://nonexistent.invalid/x.jpg")

    @pytest.mark.asyncio
    async def test_http_status_error_is_not_swallowed(self):
        connector = _make_connector()
        connector._image_loader = SimpleNamespace(
            load_image=AsyncMock(
                side_effect=HttpStatusError(415, "Unsupported Media Type", "https://x/y")
            )
        )

        with pytest.raises(HttpStatusError) as exc_info:
            await connector.fetch_image_async("https://x/y")
        assert exc_info.value.status == 415

    @pytest.mark.asyncio
    async def test_local_path_still_falls_back_to_parent(self):
        """A scheme ImageLoader does not handle must still reach vLLM.

        This is what keeps --allowed-local-media-path working, so the fix above
        must not turn every ValueError into a hard failure.
        """
        connector = _make_connector()
        connector._image_loader = SimpleNamespace(
            load_image=AsyncMock(
                side_effect=ValueError(
                    "Invalid image source scheme: local file access is not allowed"
                )
            )
        )
        sentinel = _make_pil_image()
        with patch.object(
            media_connector.MediaConnector,
            "fetch_image_async",
            new=AsyncMock(return_value=sentinel),
        ) as parent:
            assert await connector.fetch_image_async("/tmp/x.jpg") is sentinel
        parent.assert_awaited_once()
