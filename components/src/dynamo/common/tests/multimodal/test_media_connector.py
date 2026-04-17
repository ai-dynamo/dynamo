# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DynamoMediaConnector and its ImageLoader integration."""

import pytest
from PIL import Image

from dynamo.common.multimodal.image_loader import ImageLoader

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
