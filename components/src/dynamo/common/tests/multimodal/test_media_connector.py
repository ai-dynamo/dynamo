# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DynamoMediaConnector and its ImageLoader integration."""

import base64
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

import dynamo.common.multimodal.media_connector as media_connector_module
from dynamo.common.http import HttpStatusError
from dynamo.common.http.url_validator import UrlValidationError
from dynamo.common.multimodal.image_loader import ImageLoader

try:
    from vllm.multimodal.media.image import ImageMediaIO
except ImportError:  # vLLM is optional here; the vllm-marked tests skip without it
    ImageMediaIO = None  # type: ignore[assignment,misc]

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _make_pil_image() -> Image.Image:
    return Image.new("RGB", (4, 4), color="blue")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client_error",
    [
        UrlValidationError("blocked host"),
        HttpStatusError(415, "Unsupported Media Type", "https://example.com/x.jpg"),
    ],
)
async def test_connector_does_not_fallback_on_client_error(client_error, monkeypatch):
    connector_class = getattr(media_connector_module, "DynamoMediaConnector", None)
    if connector_class is None:
        pytest.skip("vLLM is not installed")

    parent_fetch = AsyncMock()
    monkeypatch.setattr(connector_class.__mro__[1], "fetch_image_async", parent_fetch)
    connector = object.__new__(connector_class)
    connector.media_io_kwargs = {}
    connector._image_loader = AsyncMock()
    connector._image_loader.load_image.side_effect = client_error

    with pytest.raises(type(client_error)) as exc_info:
        await connector.fetch_image_async("https://example.com/x.jpg")

    assert exc_info.value is client_error
    parent_fetch.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("fetch", ["fetch_audio_async", "fetch_video_async"])
async def test_connector_rejects_remote_urls_when_disabled(fetch, monkeypatch):
    connector_class = getattr(media_connector_module, "DynamoMediaConnector", None)
    if connector_class is None:
        pytest.skip("vLLM is not installed")

    monkeypatch.setenv("DYN_MM_ALLOW_REMOTE_URLS", "0")
    parent_load = AsyncMock()
    monkeypatch.setattr(connector_class.__mro__[1], "load_from_url_async", parent_load)
    connector = object.__new__(connector_class)
    connector.media_io_kwargs = {}

    with pytest.raises(UrlValidationError, match="DYN_MM_ALLOW_REMOTE_URLS"):
        await getattr(connector, fetch)("https://example.com/media")

    parent_load.assert_not_awaited()


def test_connector_sync_load_rejects_remote_urls_when_disabled(monkeypatch):
    connector_class = getattr(media_connector_module, "DynamoMediaConnector", None)
    if connector_class is None:
        pytest.skip("vLLM is not installed")

    monkeypatch.setenv("DYN_MM_ALLOW_REMOTE_URLS", "0")
    parent_load = MagicMock()
    monkeypatch.setattr(connector_class.__mro__[1], "load_from_url", parent_load)
    connector = object.__new__(connector_class)

    with pytest.raises(UrlValidationError, match="DYN_MM_ALLOW_REMOTE_URLS"):
        connector.load_from_url("https://example.com/a.wav", MagicMock())

    parent_load.assert_not_called()


@pytest.mark.vllm
@pytest.mark.asyncio
@pytest.mark.parametrize("exif_orientation", [None, 6])
async def test_connector_image_matches_vllm_connector(exif_orientation, monkeypatch):
    connector_class = getattr(media_connector_module, "DynamoMediaConnector", None)
    if connector_class is None:
        pytest.skip("vLLM is not installed")

    # The loader is a process-wide singleton; build a fresh one for this test.
    monkeypatch.setattr(media_connector_module, "_shared_image_loader", None)
    image = Image.new("RGBA", (4, 2))
    image.putpixel((0, 0), (0, 0, 0, 255))
    exif = Image.Exif()
    if exif_orientation is not None:
        exif[0x0112] = exif_orientation
    buf = BytesIO()
    image.save(buf, format="PNG", exif=exif)
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    loaded = await connector_class().fetch_image_async(url)

    vllm_connector_class = connector_class.__mro__[1]
    expected = await vllm_connector_class().fetch_image_async(url)
    assert (loaded.size, loaded.mode, loaded.tobytes()) == (
        expected.size,
        expected.mode,
        expected.tobytes(),
    )


@pytest.mark.vllm
@pytest.mark.multimodal
@pytest.mark.asyncio
async def test_connector_applies_rgba_background_color_like_vllm_connector(
    monkeypatch,
):
    connector_class = getattr(media_connector_module, "DynamoMediaConnector", None)
    if connector_class is None:
        pytest.skip("vLLM is not installed")

    monkeypatch.setattr(media_connector_module, "_shared_image_loader", None)
    image = Image.new("RGBA", (4, 2))
    image.putpixel((0, 0), (0, 0, 0, 255))
    buf = BytesIO()
    image.save(buf, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    media_io_kwargs = {"image": {"rgba_background_color": [255, 0, 0]}}

    loaded = await connector_class(media_io_kwargs=media_io_kwargs).fetch_image_async(
        url
    )

    vllm_connector_class = connector_class.__mro__[1]
    expected = await vllm_connector_class(
        media_io_kwargs=media_io_kwargs
    ).fetch_image_async(url)
    assert (loaded.size, loaded.mode, loaded.tobytes()) == (
        expected.size,
        expected.mode,
        expected.tobytes(),
    )


@pytest.mark.vllm
@pytest.mark.multimodal
@pytest.mark.asyncio
async def test_connector_keeps_rgb_when_options_set_image_mode(monkeypatch):
    """vLLM's Kimi K3 renderer sets image_mode None on every request; the image stays RGB."""
    connector_class = getattr(media_connector_module, "DynamoMediaConnector", None)
    if connector_class is None:
        pytest.skip("vLLM is not installed")
    monkeypatch.setattr(media_connector_module, "_shared_image_loader", None)
    image = Image.new("RGBA", (4, 2))
    image.putpixel((0, 0), (0, 0, 0, 255))
    buf = BytesIO()
    image.save(buf, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    connector = connector_class(media_io_kwargs={"image": {"image_mode": None}})

    loaded = await connector.fetch_image_async(url)

    expected = ImageMediaIO(image_mode="RGB").load_bytes(buf.getvalue()).media
    assert (loaded.mode, loaded.tobytes()) == (expected.mode, expected.tobytes())


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

    def test_cache_size_zero_disables_cache(self):
        """A capacity of zero stores nothing rather than evicting from an empty cache."""
        loader = ImageLoader(cache_size=0)

        loader._cache_put("url1", _make_pil_image())
        loader._cache_put("url2", _make_pil_image())

        assert len(loader._image_cache) == 0
