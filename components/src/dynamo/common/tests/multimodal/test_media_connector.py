# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DynamoMediaConnector and its ImageLoader integration."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

import dynamo.common.multimodal.media_connector as media_connector_module
from dynamo.common.http import HttpStatusError
from dynamo.common.http.url_validator import MediaSizeError, UrlValidationError
from dynamo.common.multimodal.image_loader import ImageLoader

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _make_pil_image() -> Image.Image:
    return Image.new("RGB", (4, 4), color="blue")


@pytest.mark.vllm
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
    connector.allowed_media_domains = None
    connector._image_loader = AsyncMock()
    connector._image_loader.load_image.side_effect = client_error

    with pytest.raises(type(client_error)) as exc_info:
        await connector.fetch_image_async("https://example.com/x.jpg")

    assert exc_info.value is client_error
    parent_fetch.assert_not_awaited()


@pytest.mark.vllm
@pytest.mark.asyncio
async def test_connector_applies_allowed_media_domains_to_images():
    """vLLM's --allowed-media-domains applies to images, which ImageLoader
    fetches, as it does to audio and video, which vLLM fetches."""
    connector_class = getattr(media_connector_module, "DynamoMediaConnector", None)
    if connector_class is None:
        pytest.skip("vLLM is not installed")

    connector = object.__new__(connector_class)
    connector.allowed_media_domains = ["example.com"]
    connector._image_loader = AsyncMock()
    connector._image_loader.load_image.return_value = _make_pil_image()

    with pytest.raises(ValueError, match="allowed domains"):
        await connector.fetch_image_async("http://169.254.169.254/latest/meta-data/")
    connector._image_loader.load_image.assert_not_awaited()

    image = await connector.fetch_image_async("https://example.com/x.jpg")
    assert image.size == (4, 4)
    # data: URLs have no domain, so the allowlist does not apply.
    await connector.fetch_image_async("data:image/png;base64,AAAA")


@pytest.mark.vllm
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
    connector.decoded_budget = None

    with pytest.raises(UrlValidationError, match="DYN_MM_ALLOW_REMOTE_URLS"):
        await getattr(connector, fetch)("https://example.com/media")

    parent_load.assert_not_awaited()


@pytest.mark.vllm
@pytest.mark.asyncio
@pytest.mark.parametrize("fetch", ["fetch_audio_async", "fetch_video_async"])
@pytest.mark.parametrize(
    "url, error",
    [
        ("https://127.0.0.1/latest/meta-data/", "blocked range"),
        ("https://169.254.169.254/latest/meta-data/", "blocked range"),
        ("http://example.com/media", "http:// URLs are not allowed"),
    ],
)
async def test_connector_applies_url_policy_to_audio_and_video(
    fetch, url, error, monkeypatch
):
    """Audio and video follow the same DYN_MM_* URL rules as images, so the
    default policy refuses private addresses before vLLM downloads them."""
    connector_class = getattr(media_connector_module, "DynamoMediaConnector", None)
    if connector_class is None:
        pytest.skip("vLLM is not installed")

    for var in ("DYN_MM_ALLOW_INTERNAL", "DYN_MM_ALLOW_REMOTE_URLS"):
        monkeypatch.delenv(var, raising=False)
    parent_load = AsyncMock()
    monkeypatch.setattr(connector_class.__mro__[1], "load_from_url_async", parent_load)
    connector = object.__new__(connector_class)
    connector.media_io_kwargs = {}
    connector.decoded_budget = None

    with pytest.raises(UrlValidationError, match=error):
        await getattr(connector, fetch)(url)
    parent_load.assert_not_awaited()

    # data: URLs carry the media inline, so the URL rules don't apply.
    await getattr(connector, fetch)("data:audio/wav;base64,AAAA")
    parent_load.assert_awaited_once()


@pytest.mark.vllm
@pytest.mark.asyncio
@pytest.mark.parametrize("fetch", ["fetch_audio_async", "fetch_video_async"])
async def test_connector_applies_data_url_size_limit(fetch, monkeypatch):
    """Audio and video data: URLs get the DYN_MM_MAX_DATA_URL_MB cap that
    ImageLoader applies to images, before vLLM decodes them."""
    connector_class = getattr(media_connector_module, "DynamoMediaConnector", None)
    if connector_class is None:
        pytest.skip("vLLM is not installed")

    monkeypatch.setenv("DYN_MM_MAX_DATA_URL_MB", "1")
    parent_load = AsyncMock()
    monkeypatch.setattr(connector_class.__mro__[1], "load_from_url_async", parent_load)
    connector = object.__new__(connector_class)
    connector.media_io_kwargs = {}
    connector.decoded_budget = None

    oversized = "data:audio/wav;base64," + "A" * (1024 * 1024)
    with pytest.raises(MediaSizeError, match="DYN_MM_MAX_DATA_URL_MB"):
        await getattr(connector, fetch)(oversized)
    parent_load.assert_not_awaited()

    await getattr(connector, fetch)("data:audio/wav;base64,AAAA")
    parent_load.assert_awaited_once()


def _stub_media_io(max_bytes=None):
    media_io = MagicMock()
    media_io.get_max_bytes.return_value = max_bytes
    media_io.load_bytes.return_value = "decoded"
    return media_io


@pytest.mark.vllm
@pytest.mark.asyncio
@pytest.mark.parametrize("vllm_max_bytes", [None, 1024])
async def test_connector_downloads_remote_media_with_dynamo_fetcher(
    vllm_max_bytes, monkeypatch
):
    """http(s) media is downloaded with dynamo's fetcher, which checks the URL
    policy on every redirect hop, and decoded by vLLM's media_io, so the
    renderer gets the same objects as from vLLM's own download."""
    connector_class = getattr(media_connector_module, "DynamoMediaConnector", None)
    if connector_class is None:
        pytest.skip("vLLM is not installed")

    monkeypatch.setattr(media_connector_module, "validate_media_url", AsyncMock())
    fetch = AsyncMock(return_value=b"media bytes")
    monkeypatch.setattr(media_connector_module, "fetch_bytes", fetch)
    parent_load = AsyncMock()
    monkeypatch.setattr(connector_class.__mro__[1], "load_from_url_async", parent_load)
    connector = object.__new__(connector_class)
    connector.allowed_media_domains = None
    media_io = _stub_media_io(vllm_max_bytes)

    result = await connector.load_from_url_async(
        "https://example.com/clip.mp4", media_io, fetch_timeout=7
    )

    assert result == "decoded"
    media_io.load_bytes.assert_called_once_with(b"media bytes")
    parent_load.assert_not_awaited()
    (url, timeout), kwargs = fetch.await_args
    assert (url, timeout) == ("https://example.com/clip.mp4", 7)
    assert isinstance(kwargs["policy"], media_connector_module.UrlValidationPolicy)
    expected_cap = media_connector_module.max_media_bytes()
    if vllm_max_bytes is not None:
        expected_cap = min(expected_cap, vllm_max_bytes)
    assert kwargs["max_bytes"] == expected_cap


@pytest.mark.vllm
@pytest.mark.asyncio
async def test_connector_applies_allowed_media_domains_to_remote_media(monkeypatch):
    """Skipping vLLM's download keeps its --allowed-media-domains check."""
    connector_class = getattr(media_connector_module, "DynamoMediaConnector", None)
    if connector_class is None:
        pytest.skip("vLLM is not installed")

    monkeypatch.setattr(media_connector_module, "validate_media_url", AsyncMock())
    fetch = AsyncMock(return_value=b"media bytes")
    monkeypatch.setattr(media_connector_module, "fetch_bytes", fetch)
    connector = object.__new__(connector_class)
    connector.allowed_media_domains = ["example.com"]

    with pytest.raises(ValueError, match="allowed domains"):
        await connector.load_from_url_async(
            "https://other.example.org/a.wav", _stub_media_io()
        )
    fetch.assert_not_awaited()

    assert (
        await connector.load_from_url_async(
            "https://example.com/a.wav", _stub_media_io()
        )
        == "decoded"
    )


@pytest.mark.vllm
@pytest.mark.asyncio
async def test_connector_leaves_local_files_to_vllm(monkeypatch):
    """A local path passes the DYN_MM_LOCAL_PATH check, then vLLM loads it."""
    connector_class = getattr(media_connector_module, "DynamoMediaConnector", None)
    if connector_class is None:
        pytest.skip("vLLM is not installed")

    validate = AsyncMock()
    monkeypatch.setattr(media_connector_module, "validate_media_url", validate)
    fetch = AsyncMock()
    monkeypatch.setattr(media_connector_module, "fetch_bytes", fetch)
    parent_load = AsyncMock(return_value="from vllm")
    monkeypatch.setattr(connector_class.__mro__[1], "load_from_url_async", parent_load)
    connector = object.__new__(connector_class)

    result = await connector.load_from_url_async("file:///media/a.wav", MagicMock())

    assert result == "from vllm"
    validate.assert_awaited_once()
    fetch.assert_not_awaited()


@pytest.mark.vllm
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
