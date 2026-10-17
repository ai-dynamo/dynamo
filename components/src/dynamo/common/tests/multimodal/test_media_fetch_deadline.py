# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Caller-level DNS deadline and policy regressions with no external network."""
import asyncio
from unittest.mock import AsyncMock

import pytest

from dynamo.common import http
from dynamo.common.http import (
    AiohttpClient,
    HttpStatusError,
    HttpTimeoutError,
    from_env,
)
from dynamo.common.http.media_reference import local_media_reference
from dynamo.common.http.url_validator import UrlValidationError, UrlValidationPolicy
from dynamo.common.multimodal.audio_loader import AudioLoader
from dynamo.common.multimodal.image_loader import ImageLoader
from dynamo.common.multimodal.video_loader import VideoLoader

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.asyncio,
    pytest.mark.timeout(5),
]


async def _load(kind, url, policy):
    if kind == "image":
        return await ImageLoader(url_policy=policy).load_image(url)
    if kind == "audio":
        loader = AudioLoader(url_policy=policy)
        loader._create_vllm_audio_io = lambda: object()
        return await loader._load_audio_with_vllm(url)
    if kind == "video":
        loader = VideoLoader(url_policy=policy)
        loader._create_vllm_video_io = lambda _: object()
        return await loader._load_video_with_vllm(url)
    async with local_media_reference(url, policy) as path:
        return path


@pytest.mark.parametrize("kind", ["image", "audio", "video", "reference"])
async def test_initial_dns_runs_inside_fetch_deadline(monkeypatch, kind):
    config = from_env()
    config.per_call_timeout_override = 0.01
    client = AiohttpClient(config)
    monkeypatch.setattr(http, "_default", client)
    fetch_entered = False
    lookup_cancelled = False
    original = client._fetch_with_revalidation

    async def bounded_fetch(*args, **kwargs):
        nonlocal fetch_entered
        fetch_entered = True
        return await original(*args, **kwargs)

    async def stalled_dns(*args, **kwargs):
        nonlocal lookup_cancelled
        assert fetch_entered, "caller performed DNS before entering the bounded fetch"
        try:
            await asyncio.Future()
        finally:
            lookup_cancelled = True

    monkeypatch.setattr(client, "_fetch_with_revalidation", bounded_fetch)
    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", stalled_dns)
    try:
        expected = HttpStatusError if kind == "image" else HttpTimeoutError
        with pytest.raises(expected) as error:
            await _load(kind, "https://media.example/input", UrlValidationPolicy())
        if kind == "image":
            assert error.value.status == 408
        assert lookup_cancelled
    finally:
        await client.close()


@pytest.mark.parametrize("kind", ["image", "audio", "video", "reference"])
async def test_dns_private_address_rejected_before_download(monkeypatch, kind):
    client = AiohttpClient(from_env())
    monkeypatch.setattr(http, "_default", client)
    dns = AsyncMock(return_value=[(2, 1, 6, "", ("127.0.0.1", 443))])
    download = AsyncMock(side_effect=AssertionError("blocked source must not download"))
    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", dns)
    monkeypatch.setattr(client, "_fetch_body_or_redirect", download)
    try:
        with pytest.raises(UrlValidationError, match="blocked IP"):
            await _load(kind, "https://media.example/input", UrlValidationPolicy())
        dns.assert_awaited_once()
        download.assert_not_awaited()
    finally:
        await client.close()
