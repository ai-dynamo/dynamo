# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

from dynamo.vllm.handlers import BaseWorkerHandler

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class _TestWorkerHandler(BaseWorkerHandler):
    async def generate(self, request, context):
        yield {}


def _make_handler(enable_multimodal: bool = True) -> _TestWorkerHandler:
    handler = _TestWorkerHandler.__new__(_TestWorkerHandler)
    handler.enable_multimodal = enable_multimodal
    handler.config = SimpleNamespace(model="Qwen/Qwen3-VL-2B-Instruct")
    handler.embedding_loader = None
    handler.image_loader = SimpleNamespace(load_image_batch=AsyncMock(return_value=[]))
    handler.video_loader = SimpleNamespace(load_video_batch=AsyncMock(return_value=[]))
    handler.audio_loader = SimpleNamespace(
        load_audio_batch=AsyncMock(return_value=[]),
        load_audio=AsyncMock(return_value=(np.zeros(16000, dtype=np.float32), 16000.0)),
    )
    return handler


@pytest.mark.asyncio
async def test_extract_multimodal_data_loads_video_url_items():
    handler = _make_handler()
    video = (
        np.zeros((2, 4, 4, 3), dtype=np.uint8),
        {"fps": 2.0, "frames_indices": [0, 1], "total_num_frames": 2},
    )
    handler.video_loader.load_video_batch = AsyncMock(return_value=[video])

    result = await handler._extract_multimodal_data(
        {"multi_modal_data": {"video_url": [{"Url": "https://example.com/video.mp4"}]}},
        "req-1",
        context=None,
    )

    assert result is not None
    assert result["video"] is video
    handler.image_loader.load_image_batch.assert_not_awaited()


@pytest.mark.asyncio
async def test_extract_multimodal_data_merges_image_embeddings_with_video():
    handler = _make_handler()
    image_mm_data = {"image": {"image_embeds": object()}}
    video = (
        np.ones((3, 4, 4, 3), dtype=np.uint8),
        {"fps": 2.0, "frames_indices": [0, 1, 2], "total_num_frames": 3},
    )
    handler.embedding_loader = SimpleNamespace(
        load_multimodal_embeddings=AsyncMock(return_value=image_mm_data)
    )
    handler.video_loader.load_video_batch = AsyncMock(return_value=[video])

    result = await handler._extract_multimodal_data(
        {
            "multi_modal_data": {
                "image_url": [{"Url": "https://example.com/image.png"}],
                "video_url": [{"Url": "https://example.com/video.mp4"}],
            }
        },
        "req-2",
        context=None,
    )

    assert result is not None
    assert result["image"] is image_mm_data["image"]
    assert result["video"] is video
    handler.image_loader.load_image_batch.assert_not_awaited()


@pytest.mark.asyncio
async def test_extract_multimodal_data_falls_back_to_image_loader_for_decoded_images():
    handler = _make_handler()
    image = object()
    video = (
        np.full((1, 2, 2, 3), 7, dtype=np.uint8),
        {"fps": 2.0, "frames_indices": [0], "total_num_frames": 1},
    )
    handler.embedding_loader = SimpleNamespace(
        load_multimodal_embeddings=AsyncMock(return_value={"image": "unused"})
    )
    handler.image_loader.load_image_batch = AsyncMock(return_value=[image])
    handler.video_loader.load_video_batch = AsyncMock(return_value=[video])

    result = await handler._extract_multimodal_data(
        {
            "multi_modal_data": {
                "image_url": [{"Decoded": {"shape": [1, 1, 3]}}],
                "video_url": [{"Url": "https://example.com/video.mp4"}],
            }
        },
        "req-3",
        context=None,
    )

    assert result is not None
    assert result["image"] is image
    assert result["video"] is video
    handler.embedding_loader.load_multimodal_embeddings.assert_not_awaited()
    handler.image_loader.load_image_batch.assert_awaited_once()


@pytest.mark.asyncio
async def test_extract_multimodal_data_rejects_requests_when_disabled():
    handler = _make_handler(enable_multimodal=False)

    with pytest.raises(ValueError, match="multimodal processing is not enabled"):
        await handler._extract_multimodal_data(
            {
                "multi_modal_data": {
                    "video_url": [{"Url": "https://example.com/video.mp4"}]
                }
            },
            "req-4",
            context=None,
        )


# --- use_audio_in_video tests ---


@pytest.mark.asyncio
async def test_extract_audio_from_video_when_use_audio_in_video():
    """Audio is extracted from video URLs when use_audio_in_video=True."""
    handler = _make_handler()
    video = (
        np.zeros((2, 4, 4, 3), dtype=np.uint8),
        {"fps": 2.0, "frames_indices": [0, 1], "total_num_frames": 2},
    )
    audio = (np.zeros(16000, dtype=np.float32), 16000.0)
    handler.video_loader.load_video_batch = AsyncMock(return_value=[video])
    handler.audio_loader.load_audio = AsyncMock(return_value=audio)

    result = await handler._extract_multimodal_data(
        {"multi_modal_data": {"video_url": [{"Url": "https://example.com/video.mp4"}]}},
        "req-aiv-1",
        context=None,
        mm_processor_kwargs={"use_audio_in_video": True},
    )

    assert result is not None
    assert result["video"] is video
    assert result["audio"] is audio
    handler.audio_loader.load_audio.assert_awaited_once_with(
        "https://example.com/video.mp4"
    )


@pytest.mark.asyncio
async def test_no_audio_from_video_without_flag():
    """Without use_audio_in_video, no audio is extracted from video URLs."""
    handler = _make_handler()
    video = (
        np.zeros((2, 4, 4, 3), dtype=np.uint8),
        {"fps": 2.0, "frames_indices": [0, 1], "total_num_frames": 2},
    )
    handler.video_loader.load_video_batch = AsyncMock(return_value=[video])

    result = await handler._extract_multimodal_data(
        {"multi_modal_data": {"video_url": [{"Url": "https://example.com/video.mp4"}]}},
        "req-aiv-2",
        context=None,
    )

    assert result is not None
    assert result["video"] is video
    assert "audio" not in result
    handler.audio_loader.load_audio.assert_not_awaited()


@pytest.mark.asyncio
async def test_audio_from_video_multiple_videos_preserves_order():
    """With multiple videos, audio is extracted per-video in the same order."""
    handler = _make_handler()
    video_a = (np.zeros((1, 4, 4, 3), dtype=np.uint8), {"fps": 1.0})
    video_b = (np.ones((1, 4, 4, 3), dtype=np.uint8), {"fps": 1.0})
    audio_a = (np.zeros(8000, dtype=np.float32), 16000.0)
    audio_b = (np.ones(8000, dtype=np.float32), 16000.0)

    handler.video_loader.load_video_batch = AsyncMock(return_value=[video_a, video_b])
    handler.audio_loader.load_audio = AsyncMock(side_effect=[audio_a, audio_b])

    video_items = [
        {"Url": "https://example.com/a.mp4"},
        {"Url": "https://example.com/b.mp4"},
    ]
    result = await handler._extract_multimodal_data(
        {"multi_modal_data": {"video_url": video_items}},
        "req-aiv-3",
        context=None,
        mm_processor_kwargs={"use_audio_in_video": True},
    )

    assert result is not None
    assert result["video"] == [video_a, video_b]
    assert result["audio"] == [audio_a, audio_b]


@pytest.mark.asyncio
async def test_audio_from_video_raises_on_silent_video():
    """A video without an audio track raises — silent videos break 1:1 ordering."""
    handler = _make_handler()
    video = (
        np.zeros((2, 4, 4, 3), dtype=np.uint8),
        {"fps": 2.0, "frames_indices": [0, 1], "total_num_frames": 2},
    )
    handler.video_loader.load_video_batch = AsyncMock(return_value=[video])
    handler.audio_loader.load_audio = AsyncMock(
        side_effect=RuntimeError("no audio stream")
    )

    with pytest.raises(RuntimeError, match="no audio stream"):
        await handler._extract_multimodal_data(
            {
                "multi_modal_data": {
                    "video_url": [{"Url": "https://example.com/silent.mp4"}]
                }
            },
            "req-aiv-4",
            context=None,
            mm_processor_kwargs={"use_audio_in_video": True},
        )


@pytest.mark.asyncio
async def test_audio_from_video_raises_on_non_url_video():
    """A decoded (non-URL) video item raises when use_audio_in_video is set."""
    handler = _make_handler()
    video = (np.zeros((2, 4, 4, 3), dtype=np.uint8), {"fps": 2.0})
    handler.video_loader.load_video_batch = AsyncMock(return_value=[video])

    with pytest.raises(ValueError, match="non-URL video item"):
        await handler._extract_multimodal_data(
            {"multi_modal_data": {"video_url": [{"Decoded": {"shape": [2, 4, 4, 3]}}]}},
            "req-aiv-decoded",
            context=None,
            mm_processor_kwargs={"use_audio_in_video": True},
        )


@pytest.mark.asyncio
async def test_audio_from_video_merges_with_standalone_audio():
    """Standalone audio_url items and video-extracted audio are both included."""
    handler = _make_handler()
    video = (np.zeros((2, 4, 4, 3), dtype=np.uint8), {"fps": 2.0})
    standalone_audio = (np.zeros(8000, dtype=np.float32), 16000.0)
    video_audio = (np.ones(8000, dtype=np.float32), 16000.0)

    handler.video_loader.load_video_batch = AsyncMock(return_value=[video])
    handler.audio_loader.load_audio_batch = AsyncMock(return_value=[standalone_audio])
    handler.audio_loader.load_audio = AsyncMock(return_value=video_audio)

    result = await handler._extract_multimodal_data(
        {
            "multi_modal_data": {
                "video_url": [{"Url": "https://example.com/video.mp4"}],
                "audio_url": [{"Url": "https://example.com/narration.wav"}],
            }
        },
        "req-aiv-5",
        context=None,
        mm_processor_kwargs={"use_audio_in_video": True},
    )

    assert result is not None
    assert result["audio"] == [standalone_audio, video_audio]


@pytest.mark.asyncio
async def test_build_prompt_includes_mm_processor_kwargs():
    """mm_processor_kwargs is included in the TokensPrompt."""
    handler = _make_handler()
    mm_kwargs = {"use_audio_in_video": True}

    prompt, _, error = handler._build_prompt_from_request(
        {"token_ids": [1, 2, 3]},
        "req-prompt-1",
        multi_modal_data=None,
        mm_processor_kwargs=mm_kwargs,
    )

    assert error is None
    assert prompt["mm_processor_kwargs"] is mm_kwargs


@pytest.mark.asyncio
async def test_build_prompt_excludes_mm_processor_kwargs_when_none():
    """mm_processor_kwargs is not added to TokensPrompt when None."""
    handler = _make_handler()

    prompt, _, error = handler._build_prompt_from_request(
        {"token_ids": [1, 2, 3]},
        "req-prompt-2",
        multi_modal_data=None,
    )

    assert error is None
    assert "mm_processor_kwargs" not in prompt


# --- extra_args extraction tests (KV router path) ---


@pytest.mark.asyncio
async def test_extract_audio_from_video_with_mm_kwargs_in_extra_args():
    """mm_processor_kwargs nested in extra_args (KV router path) triggers
    audio extraction the same way as top-level mm_processor_kwargs."""
    handler = _make_handler()
    video = (
        np.zeros((2, 4, 4, 3), dtype=np.uint8),
        {"fps": 2.0, "frames_indices": [0, 1], "total_num_frames": 2},
    )
    audio = (np.zeros(16000, dtype=np.float32), 16000.0)
    handler.video_loader.load_video_batch = AsyncMock(return_value=[video])
    handler.audio_loader.load_audio = AsyncMock(return_value=audio)

    result = await handler._extract_multimodal_data(
        {"multi_modal_data": {"video_url": [{"Url": "https://example.com/video.mp4"}]}},
        "req-extra-1",
        context=None,
        mm_processor_kwargs={"use_audio_in_video": True},
    )

    assert result is not None
    assert result["audio"] is audio


@pytest.mark.asyncio
async def test_no_audio_extraction_when_extra_args_lacks_mm_kwargs():
    """Without mm_processor_kwargs (neither top-level nor in extra_args),
    no audio is extracted from video URLs."""
    handler = _make_handler()
    video = (
        np.zeros((2, 4, 4, 3), dtype=np.uint8),
        {"fps": 2.0, "frames_indices": [0, 1], "total_num_frames": 2},
    )
    handler.video_loader.load_video_batch = AsyncMock(return_value=[video])

    result = await handler._extract_multimodal_data(
        {"multi_modal_data": {"video_url": [{"Url": "https://example.com/video.mp4"}]}},
        "req-extra-2",
        context=None,
        mm_processor_kwargs=None,
    )

    assert result is not None
    assert "audio" not in result
    handler.audio_loader.load_audio.assert_not_awaited()
