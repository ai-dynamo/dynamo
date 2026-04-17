# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import numpy as np
import pytest

import dynamo.common.multimodal.video_loader as video_loader_module
from dynamo.common.multimodal.url_validator import (
    UrlValidationError,
    UrlValidationPolicy,
    validate_media_url,
)
from dynamo.common.multimodal.video_loader import VideoLoader

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


def test_normalize_video_url_converts_local_paths(tmp_path):
    video_path = tmp_path / "sample.webm"
    video_path.write_bytes(b"video")

    policy = UrlValidationPolicy(allowed_local_path=str(tmp_path))

    assert validate_media_url(str(video_path), policy) == video_path.resolve().as_uri()


def test_normalize_video_url_preserves_data_urls():
    data_url = "data:video/webm;base64,Zm9v"
    policy = UrlValidationPolicy()

    assert validate_media_url(data_url, policy) == data_url


def test_normalize_video_url_rejects_bare_path_by_default(tmp_path):
    video_path = tmp_path / "sample.webm"
    video_path.write_bytes(b"video")

    # Default policy has no allowed_local_path -> local paths rejected.
    policy = UrlValidationPolicy()

    with pytest.raises(UrlValidationError, match="Local media paths are not permitted"):
        validate_media_url(str(video_path), policy)


def test_normalize_video_url_rejects_private_ip():
    policy = UrlValidationPolicy()

    with pytest.raises(UrlValidationError):
        validate_media_url("https://169.254.169.254/video.mp4", policy)


def test_normalize_video_url_accepts_file_uri_inside_prefix(tmp_path):
    video_path = tmp_path / "sample.webm"
    video_path.write_bytes(b"video")
    policy = UrlValidationPolicy(allowed_local_path=str(tmp_path))

    file_uri = video_path.resolve().as_uri()
    assert validate_media_url(file_uri, policy) == file_uri


def test_normalize_video_url_rejects_file_uri_outside_prefix(tmp_path):
    allowed = tmp_path / "media"
    allowed.mkdir()
    other = tmp_path / "secret.webm"
    other.write_bytes(b"video")
    policy = UrlValidationPolicy(allowed_local_path=str(allowed))

    with pytest.raises(UrlValidationError, match="outside the allowed directory"):
        validate_media_url(other.resolve().as_uri(), policy)


@pytest.mark.asyncio
async def test_load_video_rejects_http_by_default():
    # Default env policy: http is disabled, so validation should reject this
    # before any fetch is attempted.
    loader = VideoLoader(url_policy=UrlValidationPolicy())

    with pytest.raises(ValueError, match="not allowed"):
        await loader.load_video("http://example.com/x.mp4")


@pytest.mark.asyncio
async def test_load_video_blocks_redirect_to_private_ip():
    """A 302 to a blocked IP must be rejected per-hop, not only the initial URL."""
    loader = VideoLoader(url_policy=UrlValidationPolicy())

    redirect = MagicMock(spec=httpx.Response)
    redirect.status_code = 302
    redirect.is_redirect = True
    redirect.headers = {"location": "https://169.254.169.254/evil"}
    redirect.url = httpx.URL("https://8.8.8.8/v.mp4")
    redirect.aclose = AsyncMock()

    client = MagicMock(spec=httpx.AsyncClient)
    client.build_request = MagicMock(return_value=MagicMock(spec=httpx.Request))
    client.send = AsyncMock(return_value=redirect)

    with patch(
        "dynamo.common.multimodal.video_loader.get_http_client",
        return_value=client,
    ):
        with pytest.raises(ValueError, match="blocked range"):
            await loader.load_video("https://8.8.8.8/v.mp4")


@pytest.mark.asyncio
async def test_load_video_uses_vllm_media_connector():
    loader = VideoLoader()
    # data: scheme is in the default allowlist regardless of env flags.
    loader._url_policy = UrlValidationPolicy()
    frames = np.arange(24, dtype=np.uint8).reshape(1, 2, 4, 3)[:, :, ::-1, :]
    metadata = {"fps": 4.0, "frames_indices": [0], "total_num_frames": 1}
    loader._load_video_with_vllm = AsyncMock(  # type: ignore[method-assign]
        return_value=(frames, metadata)
    )

    loaded_frames, loaded_metadata = await loader.load_video(
        "data:video/webm;base64,Zm9v"
    )

    assert loaded_frames.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(loaded_frames, np.ascontiguousarray(frames))
    assert loaded_metadata == metadata


@pytest.mark.asyncio
async def test_load_video_batch_uses_url_loader():
    loader = VideoLoader()
    first = (
        np.zeros((1, 2, 2, 3), dtype=np.uint8),
        {"fps": 2.0, "frames_indices": [0], "total_num_frames": 1},
    )
    second = (
        np.ones((1, 2, 2, 3), dtype=np.uint8),
        {"fps": 2.0, "frames_indices": [0], "total_num_frames": 1},
    )
    loader.load_video = AsyncMock(side_effect=[first, second])  # type: ignore[method-assign]

    videos = await loader.load_video_batch(
        [
            {"Url": "https://example.com/one.mp4"},
            {"Url": "https://example.com/two.mp4"},
        ]
    )

    np.testing.assert_array_equal(videos[0][0], first[0])
    np.testing.assert_array_equal(videos[1][0], second[0])
    assert videos[0][1] == first[1]
    assert videos[1][1] == second[1]


@pytest.mark.asyncio
async def test_load_video_batch_rejects_decoded_variant_without_frontend_decoding():
    loader = VideoLoader(enable_frontend_decoding=False)

    with pytest.raises(ValueError, match="enable_frontend_decoding=False"):
        await loader.load_video_batch([{"Decoded": {"shape": [1, 2, 2, 3]}}])


@pytest.mark.asyncio
async def test_load_video_batch_reads_decoded_variant_with_metadata(monkeypatch):
    loader = VideoLoader(enable_frontend_decoding=False)
    loader._enable_frontend_decoding = True
    loader._nixl_connector = object()

    decoded_item = {
        "shape": [1, 2, 2, 3],
        "metadata": {"fps": 3.0, "frames_indices": [0], "total_num_frames": 1},
    }
    frames = np.arange(12, dtype=np.uint8).reshape(1, 2, 2, 3)
    read_decoded = AsyncMock(return_value=(frames, decoded_item["metadata"]))
    monkeypatch.setattr(
        video_loader_module, "read_decoded_media_via_nixl", read_decoded
    )

    videos = await loader.load_video_batch([{"Decoded": decoded_item}])

    np.testing.assert_array_equal(videos[0][0], np.ascontiguousarray(frames))
    assert videos[0][1] == decoded_item["metadata"]
    read_decoded.assert_awaited_once_with(
        loader._nixl_connector,
        decoded_item,
        return_metadata=True,
    )
