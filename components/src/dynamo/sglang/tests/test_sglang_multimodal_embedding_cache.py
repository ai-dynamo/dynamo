# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SGLang multimodal embedding cache behavior."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch

from dynamo.common.memory.multimodal_embedding_cache_manager import (
    CachedEmbedding,
    MultimodalEmbeddingCacheManager,
)
from dynamo.common.multimodal import TransferRequest
from dynamo.sglang.request_handlers.multimodal.encode_worker_handler import (
    Modality,
    MultimodalEncodeWorkerHandler,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_1,  # sglang tests run on GPU-enabled workers
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
    pytest.mark.skipif(Modality is None, reason="SGLang Modality is required"),
]


@pytest.fixture
def cache_handler() -> MultimodalEncodeWorkerHandler:
    """Create a lightweight handler instance for cache-path unit tests."""

    class _DummyEncoder:
        def __init__(self) -> None:
            self.encode_mock = AsyncMock()
            self.model_type = "test_video_model"
            self.vision_config = {
                "video": {"fps": 2.0, "max_frames": 768, "min_frames": 4}
            }
            self.video_processor = SimpleNamespace(temporal_patch_size=2)

        async def _encode(self, mm_items, modality):
            return await self.encode_mock(mm_items, modality)

    handler = MultimodalEncodeWorkerHandler.__new__(MultimodalEncodeWorkerHandler)

    def _set_token_ids_for_test(image_token_id: int, video_token_id: int) -> None:
        handler.image_token_id = image_token_id
        handler.video_token_id = video_token_id

    handler.set_token_ids_for_test = _set_token_ids_for_test
    handler.set_token_ids_for_test(151655, 151656)
    handler._missing_video_cache_key_config_warned = False
    handler._embedding_cache = MultimodalEmbeddingCacheManager(
        capacity_bytes=32 * 1024 * 1024
    )
    handler._cache_publisher = None
    handler.encoder = _DummyEncoder()
    return handler


@pytest.mark.asyncio
async def test_encode_with_cache_partial_hit_and_reuse(
    cache_handler: MultimodalEncodeWorkerHandler,
) -> None:
    """Partial-hit should encode only misses and preserve URL order in output."""

    urls = [
        "http://example.com/a.jpg",
        "http://example.com/b.jpg",
        "http://example.com/c.jpg",
    ]

    cached_tensor = torch.full((4, 3), fill_value=-1.0)
    cache_handler._embedding_cache.set(
        cache_handler._url_hash(urls[1]),
        CachedEmbedding(tensor=cached_tensor, image_grid_thw=[1, 2, 2]),
    )

    encoded = torch.arange(12 * 3, dtype=torch.float32).reshape(12, 3)
    cache_handler.encoder.encode_mock.return_value = (
        torch.tensor([[1, 2, 4], [1, 2, 2]]),
        encoded,
        None,
    )

    grid, full_embeddings, entries = await cache_handler._encode_with_cache(
        urls, Modality.IMAGE
    )

    cache_handler.encoder.encode_mock.assert_awaited_once_with(
        [urls[0], urls[2]], Modality.IMAGE
    )

    assert grid.tolist() == [[1, 2, 4], [1, 2, 2], [1, 2, 2]]
    assert [entry.image_grid_thw for entry in entries] == [
        [1, 2, 4],
        [1, 2, 2],
        [1, 2, 2],
    ]
    assert torch.equal(full_embeddings[:8], encoded[:8])
    assert torch.equal(full_embeddings[8:12], cached_tensor)
    assert torch.equal(full_embeddings[12:16], encoded[8:12])
    new_cached_entry = cache_handler._embedding_cache.get(
        cache_handler._url_hash(urls[0])
    )
    assert new_cached_entry is not None
    assert new_cached_entry.image_grid_thw == [1, 2, 4]
    assert new_cached_entry.video_grid_thw is None

    grid2, full_embeddings2, entries2 = await cache_handler._encode_with_cache(
        urls, Modality.IMAGE
    )
    assert cache_handler.encoder.encode_mock.await_count == 1
    assert grid2.tolist() == grid.tolist()
    assert [entry.image_grid_thw for entry in entries2] == [
        entry.image_grid_thw for entry in entries
    ]
    assert torch.equal(full_embeddings2, full_embeddings)


def test_publish_cache_delta_delegates_to_publisher(
    cache_handler: MultimodalEncodeWorkerHandler,
) -> None:
    publisher = SimpleNamespace(publish_delta=AsyncMock())
    # publish_delta is sync in production bindings; AsyncMock still tracks calls.
    publisher.publish_delta = lambda added, removed: setattr(
        publisher, "last_call", (added, removed)
    )
    cache_handler._cache_publisher = publisher

    cache_handler._publish_cache_delta(["a"], ["b"])

    assert publisher.last_call == (["a"], ["b"])


@pytest.mark.asyncio
async def test_encode_with_cache_all_hit_no_remote_call(
    cache_handler: MultimodalEncodeWorkerHandler,
) -> None:
    """All-cache-hit path should not call encoder at all."""
    urls = ["http://example.com/x.jpg", "http://example.com/y.jpg"]
    x = torch.ones(2, 3)
    y = torch.ones(1, 3) * 9

    cache_handler._embedding_cache.set(
        cache_handler._url_hash(urls[0]),
        CachedEmbedding(tensor=x, image_grid_thw=[1, 1, 2]),
    )
    cache_handler._embedding_cache.set(
        cache_handler._url_hash(urls[1]),
        CachedEmbedding(tensor=y, image_grid_thw=[1, 1, 1]),
    )

    grid, full_embeddings, entries = await cache_handler._encode_with_cache(
        urls, Modality.IMAGE
    )
    cache_handler.encoder.encode_mock.assert_not_called()
    assert grid.tolist() == [[1, 1, 2], [1, 1, 1]]
    assert [entry.image_grid_thw for entry in entries] == [
        [1, 1, 2],
        [1, 1, 1],
    ]
    assert torch.equal(full_embeddings, torch.cat([x, y], dim=0))


@pytest.mark.asyncio
async def test_video_requests_reuse_cached_embeddings(
    cache_handler: MultimodalEncodeWorkerHandler,
) -> None:
    """Second identical video request should reuse cached embeddings."""

    video_url = "https://example.com/clip.mp4"
    video_token_id = cache_handler.video_token_id

    cache_handler.encoder.encode_mock.return_value = (
        torch.tensor([2, 3, 4]),
        torch.arange(24, dtype=torch.float32).reshape(6, 4),
        {
            "second_per_grid_ts": [0.5],
            "video_timestamps": [[0.25, 0.75]],
        },
    )

    transfer_future = asyncio.get_running_loop().create_future()
    transfer_future.set_result(None)

    class _DummyEmbeddingSender:
        async def send_embeddings(self, embeddings):
            self.embeddings = embeddings
            return (
                TransferRequest(
                    embeddings_shape=list(embeddings.shape),
                    embedding_dtype_str=str(embeddings.dtype),
                    serialized_request={"kind": "mock-transfer"},
                ),
                transfer_future,
            )

    class _DummyPdWorkerClient:
        def __init__(self) -> None:
            self.requests = []

        async def round_robin(self, request_json, context=None):
            self.requests.append((json.loads(request_json), context))

            async def _responses():
                yield json.dumps({"token_ids": [7], "finished": True, "text": ""})

            return _responses()

    cache_handler.embedding_sender = _DummyEmbeddingSender()
    cache_handler.pd_worker_client = _DummyPdWorkerClient()

    raw_request = {
        "token_ids": [101, video_token_id, 102],
        "stop_conditions": {"max_tokens": 8},
        "sampling_options": {"temperature": 0.0},
        "multi_modal_data": {"video_url": [{"Url": video_url}]},
    }

    outputs = []
    async for item in cache_handler.generate(raw_request, context=None):
        outputs.append(item)

    outputs_second = []
    async for item in cache_handler.generate(raw_request, context=None):
        outputs_second.append(item)

    cache_handler.encoder.encode_mock.assert_awaited_once_with(
        [video_url], Modality.VIDEO
    )

    assert outputs == [{"token_ids": [7]}]
    assert outputs_second == [{"token_ids": [7]}]

    cached_entry = cache_handler._embedding_cache.get(
        cache_handler._media_cache_key(video_url, Modality.VIDEO, cache_handler.encoder)
    )
    assert cached_entry is not None
    assert cached_entry.video_grid_thw == [2, 3, 4]
    assert cached_entry.second_per_grid_ts == 0.5
    assert cached_entry.video_timestamps == [0.25, 0.75]

    assert len(cache_handler.pd_worker_client.requests) == 2
    for pd_request, request_context in cache_handler.pd_worker_client.requests:
        assert request_context is None
        assert pd_request["request"]["token_ids"] == [101] + [video_token_id] * 6 + [
            102
        ]
        group = pd_request["multimodal_inputs"][0]
        assert group["video_grid_thw"] == [2, 3, 4]
        assert group["second_per_grid_ts"] == 0.5
        assert group["video_timestamps"] == [0.25, 0.75]
        assert group["num_mm_tokens"] == 6


def test_aux_value_for_item_rejects_mismatched_batched_lists() -> None:
    with pytest.raises(ValueError, match="Auxiliary media metadata length mismatch"):
        MultimodalEncodeWorkerHandler._aux_value_for_item([0.5], 0, 2)

    assert MultimodalEncodeWorkerHandler._aux_value_for_item(0.5, 1, 2) == 0.5
    assert (
        MultimodalEncodeWorkerHandler._aux_value_for_item(torch.tensor([0.5]), 0, 1)
        == 0.5
    )
    assert MultimodalEncodeWorkerHandler._aux_value_for_item([0.25, 0.75], 0, 1) == [
        0.25,
        0.75,
    ]


@pytest.mark.asyncio
async def test_video_cache_key_includes_sampling_config(
    cache_handler: MultimodalEncodeWorkerHandler,
) -> None:
    """Changing video sampling config should force a cache miss for the same URL."""

    video_url = "https://example.com/clip.mp4"
    cache_handler.encoder.encode_mock.return_value = (
        torch.tensor([[2, 3, 4]]),
        torch.arange(24, dtype=torch.float32).reshape(6, 4),
        {
            "second_per_grid_ts": [0.5],
            "video_timestamps": [[0.25, 0.75]],
        },
    )

    first_key = cache_handler._media_cache_key(
        video_url, Modality.VIDEO, cache_handler.encoder
    )
    await cache_handler._encode_with_cache([video_url], Modality.VIDEO)

    cache_handler.encoder.vision_config["video"]["fps"] = 4.0

    second_key = cache_handler._media_cache_key(
        video_url, Modality.VIDEO, cache_handler.encoder
    )
    await cache_handler._encode_with_cache([video_url], Modality.VIDEO)
    await cache_handler._encode_with_cache([video_url], Modality.VIDEO)

    assert first_key != second_key
    assert cache_handler.encoder.encode_mock.await_count == 2
    assert cache_handler._embedding_cache.get(first_key) is not None
    assert cache_handler._embedding_cache.get(second_key) is not None


# --- NVDEC video routing -------------------------------------------------------

_HANDLER_MOD = "dynamo.sglang.request_handlers.multimodal.encode_worker_handler"


@pytest.fixture
def nvdec_handler(cache_handler) -> MultimodalEncodeWorkerHandler:
    """cache_handler wired with the attributes the NVDEC path reads."""
    cache_handler.num_video_frames = 32
    cache_handler._url_policy = SimpleNamespace()
    return cache_handler


def test_nvdec_video_enabled_gating(nvdec_handler, monkeypatch) -> None:
    monkeypatch.setattr(f"{_HANDLER_MOD}.nvdec_available", lambda: True)

    # Disabled explicitly.
    monkeypatch.setenv("DYN_DISABLE_NVDEC", "1")
    assert nvdec_handler._nvdec_video_enabled() is False
    monkeypatch.delenv("DYN_DISABLE_NVDEC", raising=False)

    # No PyNvVideoCodec / CUDA.
    monkeypatch.setattr(f"{_HANDLER_MOD}.nvdec_available", lambda: False)
    assert nvdec_handler._nvdec_video_enabled() is False

    # Available + a model type whose pre-decoded video path crashes upstream.
    monkeypatch.setattr(f"{_HANDLER_MOD}.nvdec_available", lambda: True)
    nvdec_handler.encoder.model_type = "qwen3_vl"
    assert nvdec_handler._nvdec_video_enabled() is False

    # Available + a safe model type.
    nvdec_handler.encoder.model_type = "qwen2_5_vl"
    assert nvdec_handler._nvdec_video_enabled() is True


@pytest.mark.asyncio
async def test_build_encode_inputs_non_video_passthrough(
    nvdec_handler, monkeypatch
) -> None:
    """Image modality never touches NVDEC; URLs pass through unchanged."""
    called = False

    async def _fail(_url):
        nonlocal called
        called = True
        return object()

    monkeypatch.setattr(nvdec_handler, "_maybe_nvdec_decoder", _fail)
    urls = ["https://x/a.jpg", "https://x/b.jpg"]
    out = await nvdec_handler._build_encode_inputs(urls, "IMAGE")
    assert out == urls
    assert called is False


@pytest.mark.asyncio
async def test_build_encode_inputs_swaps_only_nvdec_hits(
    nvdec_handler, monkeypatch
) -> None:
    """H.264/H.265 URLs become decoders; others keep their URL positionally."""
    monkeypatch.setattr(f"{_HANDLER_MOD}.nvdec_available", lambda: True)
    nvdec_handler.encoder.model_type = "qwen2_5_vl"
    decoder = object()  # stand-in for the NvdecVideoDecoder

    async def _decode(url):
        return decoder if url.endswith("h264.mp4") else None

    monkeypatch.setattr(nvdec_handler, "_maybe_nvdec_decoder", _decode)
    urls = ["https://x/h264.mp4", "https://x/vp9.webm"]
    out = await nvdec_handler._build_encode_inputs(urls, "VIDEO")
    assert out[0] is decoder
    assert out[1] == "https://x/vp9.webm"


@pytest.mark.asyncio
async def test_maybe_nvdec_decoder_wraps_h264_only(nvdec_handler, monkeypatch) -> None:
    """H.264 yields a decoder built from the fetched bytes, not decoded frames."""
    decoder = object()
    monkeypatch.setattr(
        f"{_HANDLER_MOD}.validate_media_url",
        AsyncMock(return_value="https://x/clip.mp4"),
    )
    monkeypatch.setattr(f"{_HANDLER_MOD}.fetch_bytes", AsyncMock(return_value=b"bytes"))
    monkeypatch.setattr(f"{_HANDLER_MOD}.probe_video_codec", lambda _b: "h264")
    monkeypatch.setattr(f"{_HANDLER_MOD}.should_use_nvdec", lambda c: c == "h264")
    seen: dict = {}

    def _make(data):
        seen["data"] = data
        return decoder

    monkeypatch.setattr(f"{_HANDLER_MOD}.NvdecVideoDecoder", _make)
    assert await nvdec_handler._maybe_nvdec_decoder("https://x/clip.mp4") is decoder
    # The decoder is handed the validated bytes, so SGLang never refetches the URL.
    assert seen["data"] == b"bytes"


@pytest.mark.asyncio
async def test_maybe_nvdec_decoder_skips_non_hw_codec(
    nvdec_handler, monkeypatch
) -> None:
    """VP9 (not HW-routed) returns None so the URL passthrough is used."""
    monkeypatch.setattr(
        f"{_HANDLER_MOD}.validate_media_url",
        AsyncMock(return_value="https://x/clip.webm"),
    )
    monkeypatch.setattr(f"{_HANDLER_MOD}.fetch_bytes", AsyncMock(return_value=b"bytes"))
    monkeypatch.setattr(f"{_HANDLER_MOD}.probe_video_codec", lambda _b: "vp9")
    monkeypatch.setattr(f"{_HANDLER_MOD}.should_use_nvdec", lambda c: c == "h264")
    assert await nvdec_handler._maybe_nvdec_decoder("https://x/clip.webm") is None


@pytest.mark.asyncio
async def test_maybe_nvdec_decoder_rejects_non_http_scheme(
    nvdec_handler, monkeypatch
) -> None:
    """file:// (or any non-http) is never NVDEC-decoded here."""
    monkeypatch.setattr(
        f"{_HANDLER_MOD}.validate_media_url",
        AsyncMock(return_value="file:///etc/passwd"),
    )
    fetch = AsyncMock()
    monkeypatch.setattr(f"{_HANDLER_MOD}.fetch_bytes", fetch)
    assert await nvdec_handler._maybe_nvdec_decoder("file:///etc/passwd") is None
    fetch.assert_not_called()


@pytest.mark.asyncio
async def test_maybe_nvdec_decoder_falls_back_on_error(
    nvdec_handler, monkeypatch
) -> None:
    """Any failure (fetch/probe/decode) degrades to URL passthrough (None)."""
    monkeypatch.setattr(
        f"{_HANDLER_MOD}.validate_media_url",
        AsyncMock(return_value="https://x/clip.mp4"),
    )
    monkeypatch.setattr(
        f"{_HANDLER_MOD}.fetch_bytes", AsyncMock(side_effect=RuntimeError("boom"))
    )
    assert await nvdec_handler._maybe_nvdec_decoder("https://x/clip.mp4") is None


@pytest.mark.asyncio
async def test_video_routes_through_nvdec_with_cache_disabled(
    nvdec_handler, monkeypatch
) -> None:
    """NVDEC must run when the embedding cache is off -- the default config.

    Regression for the case where the NVDEC conversion lived only inside
    ``_encode_with_cache``. ``--multimodal-embedding-cache-capacity-gb`` defaults
    to 0, so a stock deployment took the no-cache branch, handed SGLang the raw
    URL, and every H.264/H.265 request failed for want of a decoder in the
    codec-compliant image. The earlier serve tests all set a non-zero capacity,
    which hid it.
    """
    nvdec_handler._embedding_cache = None  # the default
    monkeypatch.setattr(f"{_HANDLER_MOD}.nvdec_available", lambda: True)
    nvdec_handler.encoder.model_type = "qwen2_5_vl"

    decoder = object()  # stand-in for the NvdecVideoDecoder

    async def _decode(_url):
        return decoder

    monkeypatch.setattr(nvdec_handler, "_maybe_nvdec_decoder", _decode)

    video_url = "https://example.com/clip.mp4"
    nvdec_handler.encoder.encode_mock.return_value = (
        torch.tensor([2, 3, 4]),
        torch.arange(24, dtype=torch.float32).reshape(6, 4),
        {"second_per_grid_ts": [0.5], "video_timestamps": [[0.25, 0.75]]},
    )

    transfer_future = asyncio.get_running_loop().create_future()
    transfer_future.set_result(None)

    class _DummyEmbeddingSender:
        async def send_embeddings(self, embeddings):
            return (
                TransferRequest(
                    embeddings_shape=list(embeddings.shape),
                    embedding_dtype_str=str(embeddings.dtype),
                    serialized_request={"kind": "mock-transfer"},
                ),
                transfer_future,
            )

    class _DummyPdWorkerClient:
        async def round_robin(self, request_json, context=None):
            async def _responses():
                yield json.dumps({"token_ids": [7], "finished": True, "text": ""})

            return _responses()

    nvdec_handler.embedding_sender = _DummyEmbeddingSender()
    nvdec_handler.pd_worker_client = _DummyPdWorkerClient()

    raw_request = {
        "token_ids": [101, nvdec_handler.video_token_id, 102],
        "stop_conditions": {"max_tokens": 8},
        "sampling_options": {"temperature": 0.0},
        "multi_modal_data": {"video_url": [{"Url": video_url}]},
    }

    async for _ in nvdec_handler.generate(raw_request, context=None):
        pass

    # The decoder reaches SGLang, not the URL. Asserting on the URL would pass
    # against the buggy version.
    nvdec_handler.encoder.encode_mock.assert_awaited_once_with(
        [decoder], Modality.VIDEO
    )


def test_load_video_passthrough_patches_the_encode_server_binding() -> None:
    """The passthrough must reach the binding encode_server actually calls.

    Regression: an earlier revision patched base_processor and sglang.srt.utils
    but not encode_server, which does its own ``from sglang.srt.utils import
    load_video``. Every unit test passed -- they stubbed the decoder factory, so
    load_video was never reached -- while the deployed encode worker rejected
    each video request with "Unsupported video input type" and returned 400.
    """
    encode_server = pytest.importorskip(
        "sglang.srt.disaggregation.encode_server",
        reason="SGLang required to verify the patch target",
    )
    from sglang.srt.utils.video_decoder import VideoDecoderWrapper

    from dynamo.sglang.request_handlers.multimodal.encode_worker_handler import (
        _install_load_video_passthrough,
    )

    _install_load_video_passthrough()
    patched = encode_server.load_video
    assert getattr(patched, "_dynamo_nvdec_passthrough", False), (
        "encode_server.load_video is unpatched; NVDEC decoders will be rejected "
        "by SGLang at request time"
    )

    # A pre-built decoder passes straight through instead of raising.
    sentinel = VideoDecoderWrapper.__new__(VideoDecoderWrapper)
    assert patched(sentinel) is sentinel

    # Anything else still goes to the original implementation.
    with pytest.raises(Exception):
        patched(object())

    _install_load_video_passthrough()  # idempotent
    assert encode_server.load_video is patched
