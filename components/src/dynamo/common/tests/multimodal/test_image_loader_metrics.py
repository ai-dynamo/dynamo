# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the shared-cache portion of register_image_loader_metrics."""

import asyncio
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image
from redis.exceptions import RedisError

from dynamo.common.http.url_validator import UrlValidationPolicy
from dynamo.common.multimodal.image_loader import ImageLoader
from dynamo.common.utils.prometheus import register_image_loader_metrics

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

_FETCH_BYTES_PATH = "dynamo.common.multimodal.image_loader.fetch_bytes"
_REDIS_CLUSTER_FACTORY_PATH = (
    "dynamo.common.multimodal.shared_image_cache.RedisCluster.from_url"
)


def _make_png_bytes() -> bytes:
    img = Image.new("RGB", (2, 2), color="red")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


PNG_BYTES = _make_png_bytes()


def _permissive_policy() -> UrlValidationPolicy:
    return UrlValidationPolicy(allow_http=True, allow_private_ips=True)


def _mock_fetch_bytes(
    content: bytes = PNG_BYTES,
    delay: float = 0.0,
    side_effect: Exception | None = None,
) -> AsyncMock:
    async def _fetch(url, timeout, *, policy=None):
        if delay > 0:
            await asyncio.sleep(delay)
        if side_effect is not None:
            raise side_effect
        return content

    return AsyncMock(side_effect=_fetch)


def _parse_metric(
    text: str, name: str, label_substr: str | None = None
) -> float | None:
    """Parse a metric value from Prometheus expfmt text, optionally filtered
    by a label substring (e.g. 'outcome="miss"')."""
    for line in text.split("\n"):
        if line.startswith(name + "{") or line.startswith(name + " "):
            if label_substr is not None and label_substr not in line:
                continue
            parts = line.rsplit(" ", 1)
            if len(parts) == 2:
                return float(parts[1])
    return None


async def test_shared_cache_get_set_duration_histograms(monkeypatch) -> None:
    """Shared-cache timings are exported with bounded outcome labels."""
    monkeypatch.setenv("DYN_MM_SHARED_IMAGE_CACHE_ENABLED", "1")
    monkeypatch.setenv("DYN_MM_SHARED_IMAGE_CACHE_URL", "redis://dragonfly.invalid")
    client = AsyncMock()
    client.get.return_value = None

    endpoint = MagicMock()
    with patch(_REDIS_CLUSTER_FACTORY_PATH, return_value=client):
        loader = ImageLoader(cache_size=4, url_policy=_permissive_policy())
    register_image_loader_metrics(endpoint, loader, "m", "c")
    callback = endpoint.metrics.register_prometheus_expfmt_callback.call_args[0][0]

    with patch(_FETCH_BYTES_PATH, _mock_fetch_bytes()):
        await loader.load_image("https://example.com/shared.png")

    text = callback()
    assert (
        _parse_metric(
            text,
            "dynamo_component_image_shared_cache_get_duration_seconds_count",
            'outcome="miss",size_bucket="unknown"',
        )
        == 1.0
    )
    assert (
        _parse_metric(
            text,
            "dynamo_component_image_shared_cache_set_duration_seconds_count",
            'outcome="success",size_bucket="le_64kib"',
        )
        == 1.0
    )


async def test_shared_cache_error_duration_histograms(monkeypatch) -> None:
    """Redis failures are timed while the loader still fails open."""
    monkeypatch.setenv("DYN_MM_SHARED_IMAGE_CACHE_ENABLED", "1")
    monkeypatch.setenv("DYN_MM_SHARED_IMAGE_CACHE_URL", "redis://dragonfly.invalid")
    client = AsyncMock()
    client.get.side_effect = RedisError("read unavailable")
    client.set.side_effect = RedisError("write unavailable")

    endpoint = MagicMock()
    with patch(_REDIS_CLUSTER_FACTORY_PATH, return_value=client):
        loader = ImageLoader(cache_size=4, url_policy=_permissive_policy())
    register_image_loader_metrics(endpoint, loader, "m", "c")
    callback = endpoint.metrics.register_prometheus_expfmt_callback.call_args[0][0]

    with patch(_FETCH_BYTES_PATH, _mock_fetch_bytes()):
        await loader.load_image("https://example.com/shared-error.png")

    text = callback()
    assert (
        _parse_metric(
            text,
            "dynamo_component_image_shared_cache_get_duration_seconds_count",
            'outcome="error",size_bucket="unknown"',
        )
        == 1.0
    )
    assert (
        _parse_metric(
            text,
            "dynamo_component_image_shared_cache_set_duration_seconds_count",
            'outcome="error",size_bucket="le_64kib"',
        )
        == 1.0
    )
