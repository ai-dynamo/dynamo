# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock

import numpy as np
import pytest

import dynamo.common.multimodal.video_loader as video_loader_module
from dynamo.common.http.url_validator import UrlValidationPolicy
from dynamo.common.multimodal.video_loader import VideoLoader

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


@pytest.mark.asyncio
async def test_load_video_rejects_http_by_default():
    """Wiring smoke: VideoLoader plumbs ``url_policy`` to the validator.

    Validator behavior is covered in ``test_url_validator.py``;
    per-hop SSRF revalidation in ``http/test_http_backends.py``.
    """
    loader = VideoLoader(url_policy=UrlValidationPolicy())

    with pytest.raises(ValueError, match="not allowed"):
        await loader.load_video("http://example.com/x.mp4")


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


@pytest.mark.asyncio
async def test_load_processed_media_loads_named_rdma_and_inline_fields(monkeypatch):
    loader = VideoLoader(enable_frontend_decoding=False)
    loader._nixl_connector = object()
    tensor = np.arange(24, dtype=np.float32).reshape(4, 6)
    read_tensor = AsyncMock(return_value=tensor)
    monkeypatch.setattr(video_loader_module, "read_decoded_media_via_nixl", read_tensor)
    metadata = {
        "modality": "video",
        "fields": {
            "pixel_values_videos": {
                "storage": "rdma",
                "shape": [4, 6],
                "dtype": "FLOAT32",
                "layout": {"kind": "flat", "sizes_key": "patches_per_video"},
            },
            "video_grid_thw": {
                "storage": "inline",
                "dtype": "INT64",
                "data": list(np.asarray([1, 2, 2], dtype="<i8").tobytes()),
                "shape": [1, 3],
                "layout": {"kind": "batched"},
                "keep_on_host": True,
            },
            "timestamps": {
                "storage": "inline",
                "dtype": "FLOAT64",
                "data": list(np.asarray([0.25], dtype="<f8").tobytes()),
                "shape": [1, 1],
                "layout": {"kind": "batched"},
                "keep_on_host": True,
            },
        },
        "feature_token_counts": [1],
        "original_sizes": [[2, 2]],
        "content_hashes": ["abc"],
    }

    (media,) = await loader.load_processed_media_batch([{"Preprocessed": metadata}])

    np.testing.assert_array_equal(media.fields["pixel_values_videos"].value, tensor)
    np.testing.assert_array_equal(media.fields["video_grid_thw"].value, [[1, 2, 2]])
    np.testing.assert_array_equal(media.fields["timestamps"].value, [[0.25]])
    assert media.fields["video_grid_thw"].value.flags.writeable
    assert media.modality == "video"
    read_tensor.assert_awaited_once_with(
        loader._nixl_connector,
        metadata["fields"]["pixel_values_videos"],
        trim_alpha=False,
    )


@pytest.mark.asyncio
async def test_load_processed_media_rejects_mixed_variants():
    loader = VideoLoader(enable_frontend_decoding=False)
    with pytest.raises(ValueError, match="cannot mix"):
        await loader.load_processed_media_batch(
            [
                {"Preprocessed": {}},
                {"Url": "https://example.com/video.mp4"},
            ]
        )
