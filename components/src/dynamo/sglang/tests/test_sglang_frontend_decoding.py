# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import gc
import json
import weakref
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, AsyncGenerator, Dict
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
import torch
from PIL import Image

from dynamo.common.constants import DisaggregationMode, EmbeddingTransferMode
from dynamo.common.memory.multimodal_embedding_cache_manager import (
    MultimodalEmbeddingCacheManager,
)
from dynamo.common.multimodal import TransferRequest
from dynamo.sglang.backend_args import DynamoSGLangConfig
from dynamo.sglang.request_handlers.llm.decode_handler import (
    DecodeWorkerHandler,
    FrontendDecodedVideo,
)
from dynamo.sglang.request_handlers.multimodal.encode_worker_handler import (
    Modality,
    MultimodalEncodeWorkerHandler,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.multimodal,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]


def _make_config(
    *,
    frontend_decoding: bool = False,
    multimodal_encode_worker: bool = False,
    multimodal_worker: bool = False,
    dedicated_mm_encoder: bool = False,
) -> DynamoSGLangConfig:
    # ConfigBase has no kwargs __init__; sibling tests (test_backend_args.py)
    # construct via no-arg + setattr.
    config = DynamoSGLangConfig()
    config.use_sglang_tokenizer = False
    config.multimodal_encode_worker = multimodal_encode_worker
    config.multimodal_worker = multimodal_worker
    config.enable_multimodal = bool(
        multimodal_encode_worker or multimodal_worker or dedicated_mm_encoder
    )
    config.dedicated_mm_encoder = dedicated_mm_encoder
    config.embedding_transfer_mode = EmbeddingTransferMode.NIXL_WRITE
    config.embedding_worker = False
    config.image_diffusion_worker = False
    config.video_generation_worker = False
    config.enable_rl = False
    config.frontend_decoding = frontend_decoding
    return config


def test_validate_accepts_frontend_decoding_with_encode_worker():
    config = _make_config(frontend_decoding=True, multimodal_encode_worker=True)

    with pytest.warns(DeprecationWarning, match="--multimodal-encode-worker"):
        config.validate()

    assert config.enable_multimodal is True


def test_validate_rejects_frontend_decoding_with_multimodal_worker():
    config = _make_config(frontend_decoding=True, multimodal_worker=True)
    with pytest.raises(ValueError, match="not supported on internal EPD workers"):
        config.validate()


def test_validate_rejects_frontend_decoding_with_dedicated_mm_encoder():
    config = _make_config(frontend_decoding=True, dedicated_mm_encoder=True)
    with pytest.raises(ValueError, match="not supported on internal EPD workers"):
        config.validate()


def test_validate_accepts_frontend_decoding_alone():
    config = _make_config(frontend_decoding=True)
    config.validate()


@pytest.mark.asyncio
async def test_encode_worker_prepares_mixed_url_and_decoded_images_in_order():
    handler = MultimodalEncodeWorkerHandler.__new__(MultimodalEncodeWorkerHandler)
    handler._embedding_cache = MultimodalEmbeddingCacheManager(1024 * 1024)
    handler._decoded_content_hash_warning_emitted = False
    decoded_metadata = {
        "shape": [4, 4, 3],
        "dtype": "UINT8",
        "content_hash": "0123456789abcdef",
    }
    decoded_image = Image.new("RGB", (4, 4), (255, 0, 0))
    handler._image_loader = SimpleNamespace(
        load_image_batch=AsyncMock(return_value=[decoded_image])
    )

    image_items, video_urls = handler._extract_media_inputs(
        {
            "multi_modal_data": {
                "image_url": [
                    {"Url": "https://example.com/image.png"},
                    {"Decoded": decoded_metadata},
                ]
            }
        }
    )
    image_inputs, cache_keys, prechecked_entries = await handler._prepare_image_inputs(
        image_items
    )

    assert image_inputs == ["https://example.com/image.png", decoded_image]
    assert cache_keys == [
        handler._url_hash("https://example.com/image.png"),
        "0123456789abcdef",
    ]
    assert prechecked_entries == {1: None}
    assert video_urls == []
    handler._image_loader.load_image_batch.assert_awaited_once_with(
        [{"Decoded": decoded_metadata}]
    )


@pytest.mark.asyncio
async def test_encode_worker_rejects_decoded_images_without_frontend_decoding():
    handler = MultimodalEncodeWorkerHandler.__new__(MultimodalEncodeWorkerHandler)
    handler._embedding_cache = None
    handler._decoded_content_hash_warning_emitted = False
    handler._image_loader = None
    handler._video_loader = None

    with pytest.raises(ValueError, match="--frontend-decoding is not enabled"):
        await handler._prepare_image_inputs(
            [{"Decoded": {"shape": [4, 4, 3], "dtype": "UINT8"}}]
        )


@pytest.mark.asyncio
@pytest.mark.skipif(Modality is None, reason="SGLang Modality is required")
async def test_encode_worker_caches_frontend_decoded_image(caplog):
    handler = MultimodalEncodeWorkerHandler.__new__(MultimodalEncodeWorkerHandler)
    handler._decoded_content_hash_warning_emitted = False
    handler._missing_video_cache_key_config_warned = False
    handler._cache_publisher = None
    handler._embedding_cache = MultimodalEmbeddingCacheManager(1024 * 1024)
    handler.image_token_id = 42
    handler.video_token_id = None

    content_hash = "7a9bbcb11a898630"
    decoded_metadata = {
        "shape": [4, 4, 3],
        "dtype": "UINT8",
        "content_hash": content_hash,
    }
    decoded_image = Image.new("RGB", (4, 4), (0, 255, 0))
    handler._image_loader = SimpleNamespace(
        load_image_batch=AsyncMock(return_value=[decoded_image])
    )
    handler.encoder = SimpleNamespace(
        _encode=AsyncMock(
            return_value=(
                torch.tensor([1, 2, 2]),
                torch.arange(12, dtype=torch.float32).reshape(4, 3),
                None,
            )
        )
    )

    transfer_future = asyncio.get_running_loop().create_future()
    transfer_future.set_result(None)

    class _EmbeddingSender:
        async def send_embeddings(self, embeddings):
            return (
                TransferRequest(
                    embeddings_shape=list(embeddings.shape),
                    embedding_dtype_str=str(embeddings.dtype),
                    serialized_request={"kind": "test"},
                ),
                transfer_future,
            )

    class _PdWorkerClient:
        def __init__(self):
            self.request = None

        async def round_robin(self, request_json, context=None):
            self.request = json.loads(request_json)

            async def responses():
                yield json.dumps({"token_ids": [7], "finished": True, "text": ""})

            return responses()

    handler.embedding_sender = _EmbeddingSender()
    handler.pd_worker_client = _PdWorkerClient()

    raw_request = {
        "token_ids": [1, handler.image_token_id, 2],
        "stop_conditions": {"max_tokens": 8},
        "sampling_options": {"temperature": 0.0},
        "multi_modal_data": {
            "image_url": [{"Decoded": decoded_metadata}],
        },
    }

    first_outputs = [
        output async for output in handler.generate(raw_request, context=None)
    ]
    second_outputs = [
        output async for output in handler.generate(raw_request, context=None)
    ]

    assert first_outputs == [{"token_ids": [7]}]
    assert second_outputs == first_outputs
    handler._image_loader.load_image_batch.assert_awaited_once_with(
        [{"Decoded": decoded_metadata}]
    )
    handler.encoder._encode.assert_awaited_once_with([decoded_image], Modality.IMAGE)
    assert handler._embedding_cache.get(content_hash) is not None
    assert "bypass the Dynamo embedding cache" not in caplog.text

    pd_request = handler.pd_worker_client.request
    assert pd_request["request"]["token_ids"] == [1, 42, 42, 42, 42, 2]
    assert pd_request["multimodal_inputs"][0]["image_grid_thw"] == [1, 2, 2]
    assert pd_request["multimodal_inputs"][0]["num_mm_tokens"] == 4
    assert "Decoded" not in json.dumps(pd_request)


@pytest.mark.asyncio
async def test_encode_worker_missing_decoded_hash_bypasses_cache_and_warns_once(caplog):
    handler = MultimodalEncodeWorkerHandler.__new__(MultimodalEncodeWorkerHandler)
    handler._decoded_content_hash_warning_emitted = False
    handler._embedding_cache = MultimodalEmbeddingCacheManager(1024 * 1024)
    decoded_metadata = {"shape": [4, 4, 3], "dtype": "UINT8"}
    decoded_image = Image.new("RGB", (4, 4), (0, 0, 255))
    handler._image_loader = SimpleNamespace(
        load_image_batch=AsyncMock(return_value=[decoded_image])
    )

    for _ in range(2):
        (
            image_inputs,
            cache_keys,
            prechecked_entries,
        ) = await handler._prepare_image_inputs([{"Decoded": decoded_metadata}])
        assert image_inputs == [decoded_image]
        assert cache_keys == [None]
        assert prechecked_entries == {}

    warning = "descriptor has a missing or invalid canonical content_hash"
    assert caplog.text.count(warning) == 1
    assert "compatible Dynamo versions" in caplog.text


@pytest.mark.parametrize(
    "content_hash",
    [
        "0x12345678901234",
        "+123456789abcdef",
        "01_23456789abcde",
        "0123456789abcdeF",
        "",
        None,
    ],
)
def test_encode_worker_rejects_noncanonical_decoded_content_hash(content_hash):
    assert (
        MultimodalEncodeWorkerHandler._decoded_content_cache_key(
            {"content_hash": content_hash}
        )
        is None
    )


@pytest.mark.asyncio
async def test_encode_worker_skips_image_cache_keys_when_cache_is_disabled():
    handler = MultimodalEncodeWorkerHandler.__new__(MultimodalEncodeWorkerHandler)
    handler._embedding_cache = None
    handler._decoded_content_hash_warning_emitted = False
    handler._image_loader = None
    handler._video_loader = None
    handler._url_hash = Mock(side_effect=AssertionError("URL hash must not run"))

    image_inputs, cache_keys, prechecked_entries = await handler._prepare_image_inputs(
        [{"Url": "data:image/png;base64," + "A" * 4096}]
    )

    assert image_inputs == ["data:image/png;base64," + "A" * 4096]
    assert cache_keys == [None]
    assert prechecked_entries == {}
    handler._url_hash.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("media_key", ["image_url", "video_url"])
async def test_encode_worker_rejects_ambiguous_media_variant(media_key):
    handler = MultimodalEncodeWorkerHandler.__new__(MultimodalEncodeWorkerHandler)
    handler._embedding_cache = None
    handler._decoded_content_hash_warning_emitted = False
    handler._image_loader = None
    handler._video_loader = None
    request = {
        "multi_modal_data": {
            media_key: [
                {
                    "Url": "https://example.com/media",
                    "Decoded": {"shape": [4, 4, 3], "dtype": "UINT8"},
                }
            ]
        }
    }

    message = f"Unsupported {media_key[:-4]} data variant"
    image_items, video_items = handler._extract_media_inputs(request)
    if media_key == "image_url":
        with pytest.raises(ValueError, match=message):
            await handler._prepare_image_inputs(image_items)
    else:
        with pytest.raises(ValueError, match=message):
            await handler._prepare_video_inputs(video_items)


@pytest.mark.asyncio
@pytest.mark.skipif(Modality is None, reason="SGLang Modality is required")
async def test_encode_worker_releases_decoded_image_before_streaming():
    handler = MultimodalEncodeWorkerHandler.__new__(MultimodalEncodeWorkerHandler)
    handler._decoded_content_hash_warning_emitted = False
    handler._missing_video_cache_key_config_warned = False
    handler._cache_publisher = None
    handler._embedding_cache = None
    handler.image_token_id = 42
    handler.video_token_id = None
    handler._video_loader = None

    class _ImageLoader:
        image_ref = None

        async def load_image_batch(self, items):
            image = Image.new("RGB", (4, 4), (1, 2, 3))
            self.image_ref = weakref.ref(image)
            return [image]

    class _Encoder:
        async def _encode(self, media_inputs, modality):
            assert len(media_inputs) == 1
            assert isinstance(media_inputs[0], Image.Image)
            assert modality == Modality.IMAGE
            return (
                torch.tensor([1, 2, 2]),
                torch.arange(12, dtype=torch.float32).reshape(4, 3),
                None,
            )

    transfer_future = asyncio.get_running_loop().create_future()
    transfer_future.set_result(None)

    class _EmbeddingSender:
        async def send_embeddings(self, embeddings):
            return (
                TransferRequest(
                    embeddings_shape=list(embeddings.shape),
                    embedding_dtype_str=str(embeddings.dtype),
                    serialized_request={"kind": "test"},
                ),
                transfer_future,
            )

    image_loader = _ImageLoader()

    class _PdWorkerClient:
        async def round_robin(self, request_json, context=None):
            gc.collect()
            assert image_loader.image_ref is not None
            assert image_loader.image_ref() is None

            async def responses():
                yield json.dumps({"token_ids": [7], "finished": True, "text": ""})

            return responses()

    handler._image_loader = image_loader
    handler.encoder = _Encoder()
    handler.embedding_sender = _EmbeddingSender()
    handler.pd_worker_client = _PdWorkerClient()

    raw_request = {
        "token_ids": [1, handler.image_token_id, 2],
        "stop_conditions": {"max_tokens": 8},
        "sampling_options": {"temperature": 0.0},
        "multi_modal_data": {
            "image_url": [
                {
                    "Decoded": {
                        "shape": [4, 4, 3],
                        "dtype": "UINT8",
                        "content_hash": "0123456789abcdef",
                    }
                }
            ]
        },
    }

    outputs = [output async for output in handler.generate(raw_request, context=None)]
    assert outputs == [{"token_ids": [7]}]


def test_extract_mm_hashes_flattens_sglang_item_order():
    request = {
        "extra_args": {
            "mm_hashes_by_modality": {
                # SGLang groups multimodal items by modality even when their
                # prompt offsets are interleaved I-V-I.
                "image": ["image-1", "image-2"],
                "video": ["video-1"],
            }
        }
    }

    assert DecodeWorkerHandler._extract_mm_hashes(request) == [
        "image-1",
        "image-2",
        "video-1",
    ]


def test_extract_mm_hashes_preserves_legacy_image_only_protocol():
    request = {"extra_args": {"mm_hashes": ["image-1", "image-2"]}}

    assert DecodeWorkerHandler._extract_mm_hashes(request) == [
        "image-1",
        "image-2",
    ]


def test_extract_mm_hashes_rejects_malformed_group():
    request = {
        "extra_args": {
            "mm_hashes_by_modality": {
                "image": ["image-1"],
                "video": [123],
            }
        }
    }

    assert DecodeWorkerHandler._extract_mm_hashes(request) is None


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
    handler._video_loader = None
    handler._mm_hashes_supported = False

    @asynccontextmanager
    async def no_cancellation_monitor(*args, **kwargs):
        yield None

    handler._cancellation_monitor = no_cancellation_monitor

    handler._get_input_param = lambda req: {"input_ids": req.get("token_ids", [])}
    handler._resolve_lora = lambda req: None
    handler._priority_kwargs = lambda priority: {}

    return handler


async def _empty_stream() -> AsyncGenerator[Dict[str, Any], None]:
    if False:  # pragma: no cover — never yields
        yield {}


@pytest.mark.parametrize(("frame_indices", "frame_count"), [([120], 1), ([0, 0], 2)])
def test_frontend_decoded_video_falls_back_to_duration_fps(frame_indices, frame_count):
    frames = np.zeros((frame_count, 4, 4, 3), dtype=np.uint8)
    video = FrontendDecodedVideo(
        frames,
        {"fps": 24.0, "duration": 10.0, "frames_indices": frame_indices},
    )

    assert video.avg_fps == frame_count / 10.0


@pytest.mark.asyncio
async def test_aggregated_fd_off_passes_media_url_strings():
    """Without frontend decoding, media URL items pass through as strings."""
    handler = _new_decode_handler(enable_frontend_decoding=False)

    captured: Dict[str, Any] = {}

    async def fake_async_generate(**kwargs):
        captured.update(kwargs)
        return _empty_stream()

    handler.engine = SimpleNamespace(async_generate=fake_async_generate)

    request = {
        "token_ids": [1, 2, 3],
        "multi_modal_data": {
            "image_url": [{"Url": "https://example.com/a.jpg"}],
            "audio_url": [{"Url": "https://example.com/a.wav"}],
        },
    }

    async for _ in handler.generate(request, _Context()):
        pass

    assert captured["image_data"] == ["https://example.com/a.jpg"]
    assert captured["audio_data"] == ["https://example.com/a.wav"]


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
async def test_aggregated_fd_on_loads_decoded_video_frames():
    handler = _new_decode_handler(enable_frontend_decoding=True)
    handler._mm_hashes_supported = True
    frames = np.zeros((4, 4, 4, 3), dtype=np.uint8)
    metadata = {
        "fps": 24.0,
        "duration": 10.0,
        "frames_indices": [0, 80, 160, 239],
        "total_num_frames": 240,
    }
    video_loader = SimpleNamespace(
        load_video_batch=AsyncMock(return_value=[(frames, metadata)])
    )
    handler._image_loader = SimpleNamespace(load_image_batch=AsyncMock())
    handler._video_loader = video_loader

    captured: Dict[str, Any] = {}

    async def fake_async_generate(**kwargs):
        captured.update(kwargs)
        return _empty_stream()

    handler.engine = SimpleNamespace(async_generate=fake_async_generate)
    decoded_metadata = {"shape": [4, 4, 4, 3], "dtype": "uint8"}
    request = {
        "token_ids": [1, 2, 3],
        "multi_modal_data": {"video_url": [{"Decoded": decoded_metadata}]},
        "extra_args": {"mm_hashes_by_modality": {"video": ["video-hash"]}},
    }

    async for _ in handler.generate(request, _Context()):
        pass

    video_loader.load_video_batch.assert_awaited_once_with(
        [{"Decoded": decoded_metadata}]
    )
    assert len(captured["video_data"]) == 1
    video = captured["video_data"][0]
    assert isinstance(video, np.ndarray)
    assert video.avg_fps == pytest.approx(72 / 239)
    np.testing.assert_array_equal(video.get_frames_at([0, 1]), frames[[0, 1]])
    assert captured["mm_hashes"] == ["video-hash"]


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
