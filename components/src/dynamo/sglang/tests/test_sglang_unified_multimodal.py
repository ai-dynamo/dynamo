# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
    pytest.mark.skipif(
        importlib.util.find_spec("sglang") is None,
        reason="sglang not installed in this container",
    ),
]


def _engine(*, backend_decoding: bool = False, model_type: str = "qwen3_vl"):
    from dynamo.sglang.llm_engine import SglangLLMEngine

    engine = SglangLLMEngine.__new__(SglangLLMEngine)
    engine._backend_decoding = backend_decoding
    engine._decoded_media_model_type = model_type
    engine._decoded_mm_processor = None
    engine._get_input_param = lambda request: {
        "input_ids": list(request.get("token_ids", []))
    }
    return engine


async def test_unified_worker_rejects_legacy_frontend_decoding():
    from dynamo.sglang.llm_engine import _validate_unified_media_mode

    with pytest.raises(ValueError, match="legacy worker"):
        _validate_unified_media_mode(SimpleNamespace(frontend_decoding=True))


async def test_url_passthrough_when_backend_decoding_disabled():
    engine = _engine()
    request = {
        "token_ids": [1, 2],
        "multi_modal_data": {
            "image_url": [{"Url": "https://example.com/first.png"}],
            "video_url": [
                {"Url": "https://example.com/first.mp4"},
                {"Url": "https://example.com/second.mp4"},
            ],
        },
    }

    input_param, kwargs = await engine._prepare_multimodal_input(request)

    assert input_param == {"input_ids": [1, 2]}
    assert kwargs == {
        "image_data": ["https://example.com/first.png"],
        "video_data": [
            "https://example.com/first.mp4",
            "https://example.com/second.mp4",
        ],
    }


async def test_decoded_video_uses_expanded_ids_and_one_processor_output():
    engine = _engine(backend_decoding=True)
    descriptor = {"shape": [8, 4, 4, 3]}
    processor_output = {
        "format": "processor_output",
        "pixel_values_videos": "pixels",
        "video_grid_thw": "grid",
    }
    build = AsyncMock(return_value=([10, 11, 12], processor_output, "video_data"))
    engine._decoded_mm_processor = SimpleNamespace(build=build)
    request = {
        "token_ids": [1, 2],
        "extra_args": {"formatted_prompt": "<video>describe"},
        "multi_modal_data": {"video_url": [{"Decoded": descriptor}]},
    }

    input_param, kwargs = await engine._prepare_multimodal_input(request)

    build.assert_awaited_once_with("<video>describe", [], [descriptor])
    assert input_param == {"input_ids": [10, 11, 12]}
    assert kwargs == {"video_data": [processor_output]}


async def test_decoded_mixed_image_video_preserves_modality_order():
    engine = _engine(backend_decoding=True)
    images = [{"id": "image-1"}, {"id": "image-2"}]
    videos = [{"id": "video-1"}, {"id": "video-2"}]
    processor_output = {
        "format": "processor_output",
        "pixel_values": "images",
        "pixel_values_videos": "videos",
    }
    build = AsyncMock(return_value=([99], processor_output, "image_data"))
    engine._decoded_mm_processor = SimpleNamespace(build=build)
    request = {
        "token_ids": [1],
        "extra_args": {"formatted_prompt": "mixed"},
        "multi_modal_data": {
            "image_url": [{"Decoded": item} for item in images],
            "video_url": [{"Decoded": item} for item in videos],
        },
    }

    input_param, kwargs = await engine._prepare_multimodal_input(request)

    build.assert_awaited_once_with("mixed", images, videos)
    assert input_param == {"input_ids": [99]}
    assert kwargs == {"image_data": [processor_output]}


async def test_mixed_url_and_decoded_variants_are_rejected():
    engine = _engine(backend_decoding=True)
    request = {
        "token_ids": [1],
        "extra_args": {"formatted_prompt": "mixed"},
        "multi_modal_data": {
            "image_url": [{"Url": "https://example.com/image.png"}],
            "video_url": [{"Decoded": {"shape": [2, 2, 2, 3]}}],
        },
    }

    with pytest.raises(ValueError, match="cannot mix Url and Decoded"):
        await engine._prepare_multimodal_input(request)


async def test_decoded_media_requires_formatted_prompt():
    engine = _engine(backend_decoding=True)
    engine._decoded_mm_processor = SimpleNamespace(build=AsyncMock())
    request = {
        "token_ids": [1],
        "multi_modal_data": {"video_url": [{"Decoded": {"shape": [2, 2, 2, 3]}}]},
    }

    with pytest.raises(ValueError, match="extra_args.formatted_prompt"):
        await engine._prepare_multimodal_input(request)
    engine._decoded_mm_processor.build.assert_not_awaited()


async def test_decoded_media_is_gated_to_qwen3_vl_models():
    engine = _engine(backend_decoding=True, model_type="llava")
    engine._decoded_mm_processor = SimpleNamespace(build=AsyncMock())
    request = {
        "token_ids": [1],
        "extra_args": {"formatted_prompt": "video"},
        "multi_modal_data": {"video_url": [{"Decoded": {"shape": [2, 2, 2, 3]}}]},
    }

    with pytest.raises(ValueError, match="qwen3_vl/qwen3_vl_moe"):
        await engine._prepare_multimodal_input(request)
    engine._decoded_mm_processor.build.assert_not_awaited()
