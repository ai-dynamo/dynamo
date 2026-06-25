# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, AsyncGenerator, Dict
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from dynamo.common.constants import DisaggregationMode, EmbeddingTransferMode
from dynamo.sglang.backend_args import DynamoSGLangConfig
from dynamo.sglang.request_handlers.llm.decode_handler import DecodeWorkerHandler

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]


def _make_config(
    *,
    frontend_decoding: bool = False,
    multimodal_encode_worker: bool = False,
    multimodal_worker: bool = False,
) -> DynamoSGLangConfig:
    # ConfigBase has no kwargs __init__; sibling tests (test_backend_args.py)
    # construct via no-arg + setattr.
    config = DynamoSGLangConfig()
    config.use_sglang_tokenizer = False
    config.multimodal_encode_worker = multimodal_encode_worker
    config.multimodal_worker = multimodal_worker
    config.embedding_transfer_mode = EmbeddingTransferMode.NIXL_WRITE
    config.embedding_worker = False
    config.image_diffusion_worker = False
    config.video_generation_worker = False
    config.enable_rl = False
    config.frontend_decoding = frontend_decoding
    return config


def test_validate_rejects_frontend_decoding_with_encode_worker():
    config = _make_config(frontend_decoding=True, multimodal_encode_worker=True)
    with pytest.raises(ValueError, match="--frontend-decoding is incompatible"):
        config.validate()


def test_validate_rejects_frontend_decoding_with_multimodal_worker():
    config = _make_config(frontend_decoding=True, multimodal_worker=True)
    with pytest.raises(ValueError, match="--frontend-decoding is incompatible"):
        config.validate()


def test_validate_accepts_frontend_decoding_alone():
    config = _make_config(frontend_decoding=True)
    config.validate()


class _Context:
    id_value: str = "test-request"
    trace_id: str = "test-trace"

    def id(self) -> str:
        return self.id_value

    def is_stopped(self) -> bool:
        return False


def _new_decode_handler(*, enable_frontend_decoding: bool):
    """Build a DecodeWorkerHandler without invoking sgl.Engine.

    Mirrors the pattern in test_sglang_decode_handler.py — bypass __init__ and
    manually set the few attributes the methods we exercise actually read.
    """
    handler = DecodeWorkerHandler.__new__(DecodeWorkerHandler)
    handler.use_sglang_tokenizer = False
    handler.enable_trace = False
    handler.serving_mode = DisaggregationMode.AGGREGATED
    handler.config = SimpleNamespace(
        server_args=SimpleNamespace(served_model_name="test-model")
    )
    handler._routed_experts_kwargs = {}
    handler._enable_frontend_decoding = enable_frontend_decoding
    handler._image_loader = None
    handler._decoded_mm_processor = None
    handler._mm_hashes_supported = False

    @asynccontextmanager
    async def no_cancellation_monitor(*args, **kwargs):
        yield None

    handler._cancellation_monitor = no_cancellation_monitor

    handler._get_input_param = lambda req: {"input_ids": req.get("token_ids", [])}
    handler._resolve_lora = lambda req: None
    handler._session_kwargs = lambda req: {}
    handler._priority_kwargs = lambda priority: {}

    return handler


async def _empty_stream() -> AsyncGenerator[Dict[str, Any], None]:
    if False:  # pragma: no cover — never yields
        yield {}


@pytest.mark.asyncio
async def test_aggregated_fd_off_passes_url_strings():
    """Without --frontend-decoding, image_url items pass through as URL strings."""
    handler = _new_decode_handler(enable_frontend_decoding=False)

    captured: Dict[str, Any] = {}

    async def fake_async_generate(**kwargs):
        captured.update(kwargs)
        return _empty_stream()

    handler.engine = SimpleNamespace(async_generate=fake_async_generate)

    request = {
        "token_ids": [1, 2, 3],
        "multi_modal_data": {"image_url": [{"Url": "https://example.com/a.jpg"}]},
    }

    async for _ in handler.generate(request, _Context()):
        pass

    assert captured["image_data"] == ["https://example.com/a.jpg"]


@pytest.mark.asyncio
async def test_aggregated_fd_on_loads_decoded_variants_to_pil():
    """With --frontend-decoding, Decoded items are loaded via ImageLoader and
    forwarded as PIL Images (not strings) to engine.async_generate."""
    handler = _new_decode_handler(enable_frontend_decoding=True)

    decoded_metadata = {
        "shape": [4, 4, 3],
        "dtype": "uint8",
        "agent_metadata": "stub",
        "remote_descriptor": "stub",
    }
    pil_stub = Image.new("RGB", (4, 4), (255, 0, 0))

    image_loader = SimpleNamespace(
        load_image_batch=AsyncMock(return_value=[pil_stub]),
    )
    handler._image_loader = image_loader

    captured: Dict[str, Any] = {}

    async def fake_async_generate(**kwargs):
        captured.update(kwargs)
        return _empty_stream()

    handler.engine = SimpleNamespace(async_generate=fake_async_generate)

    request = {
        "token_ids": [1, 2, 3],
        "multi_modal_data": {"image_url": [{"Decoded": decoded_metadata}]},
    }

    async for _ in handler.generate(request, _Context()):
        pass

    image_loader.load_image_batch.assert_awaited_once_with(
        [{"Decoded": decoded_metadata}]
    )
    assert captured["image_data"] == [pil_stub]


@pytest.mark.asyncio
async def test_aggregated_fd_on_no_images_passes_none():
    """FD on, but request has no images — image_data must be None (not [])."""
    handler = _new_decode_handler(enable_frontend_decoding=True)
    handler._image_loader = SimpleNamespace(
        load_image_batch=AsyncMock(
            side_effect=AssertionError(
                "load_image_batch must not run when there are no images"
            )
        ),
    )

    captured: Dict[str, Any] = {}

    async def fake_async_generate(**kwargs):
        captured.update(kwargs)
        return _empty_stream()

    handler.engine = SimpleNamespace(async_generate=fake_async_generate)

    request = {"token_ids": [1, 2, 3], "multi_modal_data": {}}

    async for _ in handler.generate(request, _Context()):
        pass

    assert captured["image_data"] is None


@pytest.mark.asyncio
async def test_aggregated_fd_on_decoded_video_uses_combined_processor():
    """FD on with a Decoded video: the worker runs the combined HF processor and
    forwards the expanded input_ids + a single processor_output dict, not raw
    token_ids / video URLs."""
    handler = _new_decode_handler(enable_frontend_decoding=True)
    handler._image_loader = SimpleNamespace(
        load_image_batch=AsyncMock(
            side_effect=AssertionError("image path must not run for decoded video")
        ),
    )

    video_desc = {"shape": [8, 4, 4, 3], "dtype": "uint8"}
    combined_dict = {"format": "processor_output", "pixel_values_videos": "stub"}
    build_mock = AsyncMock(return_value=([10, 11, 12], combined_dict, "video_data"))
    handler._decoded_mm_processor = SimpleNamespace(build=build_mock)

    captured: Dict[str, Any] = {}

    async def fake_async_generate(**kwargs):
        captured.update(kwargs)
        return _empty_stream()

    handler.engine = SimpleNamespace(async_generate=fake_async_generate)

    request = {
        "token_ids": [1, 2, 3],
        "extra_args": {"formatted_prompt": "<video>describe"},
        "multi_modal_data": {"video_url": [{"Decoded": video_desc}]},
    }

    async for _ in handler.generate(request, _Context()):
        pass

    build_mock.assert_awaited_once_with("<video>describe", [], [video_desc])
    # Worker-expanded ids drive prefill; the single dict rides video_data.
    assert captured["input_ids"] == [10, 11, 12]
    assert captured["video_data"] == [combined_dict]
    assert "image_data" not in captured


@pytest.mark.asyncio
async def test_aggregated_fd_on_decoded_video_rejects_mixed_url_media():
    """A Url item alongside decoded video would lose its placeholder fill — the
    combined path must reject it loudly rather than silently drop it."""
    handler = _new_decode_handler(enable_frontend_decoding=True)
    handler._decoded_mm_processor = SimpleNamespace(
        build=AsyncMock(side_effect=AssertionError("build must not run on mixed media"))
    )
    handler.engine = SimpleNamespace(
        async_generate=AsyncMock(side_effect=AssertionError("must not generate"))
    )

    request = {
        "token_ids": [1, 2, 3],
        "extra_args": {"formatted_prompt": "<image><video>describe"},
        "multi_modal_data": {
            "image_url": [{"Url": "https://example.com/a.jpg"}],
            "video_url": [{"Decoded": {"shape": [8, 4, 4, 3], "dtype": "uint8"}}],
        },
    }

    with pytest.raises(ValueError, match="cannot mix Url media"):
        async for _ in handler.generate(request, _Context()):
            pass
