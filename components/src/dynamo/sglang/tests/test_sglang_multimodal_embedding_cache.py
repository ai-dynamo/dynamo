# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SGLang multimodal embedding cache behavior."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch

from dynamo.common.http.url_validator import (
    UrlValidationError,
    UrlValidationPolicy,
    validate_media_url,
)
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
    pytest.mark.multimodal,
    pytest.mark.gpu_0,
    # These are sub-second unit tests. A generous cap so a hang here fails this
    # test instead of stalling the whole session: an unstubbed network call once
    # cost ~400s of teardown and took the CI container down with it.
    pytest.mark.timeout(60),
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
    pytest.mark.skipif(Modality is None, reason="SGLang Modality is required"),
]


@pytest.fixture
def cache_handler(monkeypatch) -> MultimodalEncodeWorkerHandler:
    """Create a lightweight handler instance for cache-path unit tests.

    Default test-world assumption: a software video decoder exists, so the
    URL/bytes flow under test is reachable -- the codec-compliant test image
    ships none, and without this stub the handler's decoder preflight fires
    before the cache logic these tests exercise. Tests about decoder ABSENCE
    override the stub explicitly.

    URL validation is stubbed out for the same reason: it resolves the
    hostname (``loop.getaddrinfo``) to check the address against the blocked
    ranges, and the CPU test container has no DNS, so a real lookup blocks
    until the job is killed rather than failing. Tests about validation
    itself restore the real function.
    """
    monkeypatch.setattr(f"{_HANDLER_MOD}._software_video_decoder_imports", lambda: True)

    async def _passthrough_url(url, _policy):
        return url

    monkeypatch.setattr(f"{_HANDLER_MOD}.validate_media_url", _passthrough_url)

    # No unit test may reach the network. Giving the handler a real url policy
    # (below) removed the AttributeError that used to abort _maybe_nvdec_decoder
    # before it fetched, so on an NVDEC-capable host these cache tests started
    # issuing a real request for their example.com URL: ~400s of teardown while
    # the event loop waited on the connection, which killed the CI job. The
    # broad except in _maybe_nvdec_decoder turns this into the URL passthrough
    # the cache tests already expected; tests that exercise fetching stub it
    # with their own payload.
    async def _no_network(*_args, **_kwargs):
        raise AssertionError(
            "unit tests must not fetch over the network; stub fetch_bytes"
        )

    monkeypatch.setattr(f"{_HANDLER_MOD}.fetch_bytes", _no_network)

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
    handler._max_input_token_id = 151654
    handler._missing_video_cache_key_config_warned = False
    handler._decoded_content_hash_warning_emitted = False
    handler._image_loader = None
    handler._embedding_cache = MultimodalEmbeddingCacheManager(
        capacity_bytes=32 * 1024 * 1024
    )
    handler._cache_publisher = None
    # The real __init__ always sets this (UrlValidationPolicy.from_env); the
    # video paths read it, so a handler built with __new__ needs it too.
    handler._url_policy = UrlValidationPolicy()
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

    cache_keys = [cache_handler._url_hash(url) for url in urls]
    grid, full_embeddings, entries = await cache_handler._encode_with_cache(
        urls, cache_keys, Modality.IMAGE
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
        urls, cache_keys, Modality.IMAGE
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

    cache_keys = [cache_handler._url_hash(url) for url in urls]
    grid, full_embeddings, entries = await cache_handler._encode_with_cache(
        urls, cache_keys, Modality.IMAGE
    )
    cache_handler.encoder.encode_mock.assert_not_called()
    assert grid.tolist() == [[1, 1, 2], [1, 1, 1]]
    assert [entry.image_grid_thw for entry in entries] == [
        [1, 1, 2],
        [1, 1, 1],
    ]
    assert torch.equal(full_embeddings, torch.cat([x, y], dim=0))


@pytest.mark.asyncio
async def test_encode_with_cache_keeps_prechecked_hit_after_eviction(
    cache_handler: MultimodalEncodeWorkerHandler,
) -> None:
    """A decoded-image hit stays usable after it leaves the bounded LRU."""
    cache_key = "0123456789abcdef"
    cached_tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    cached_entry = CachedEmbedding(
        tensor=cached_tensor,
        image_grid_thw=[1, 1, 2],
    )

    # _prepare_image_inputs holds the returned object while an asynchronous
    # NIXL read materializes other misses. Simulate eviction during that wait.
    cache_handler._embedding_cache = MultimodalEmbeddingCacheManager(
        capacity_bytes=cached_tensor.element_size() * cached_tensor.numel()
    )
    cache_handler._embedding_cache.set(cache_key, cached_entry)
    prechecked_entry = cache_handler._embedding_cache.get(cache_key)
    assert prechecked_entry is cached_entry
    cache_handler._embedding_cache.set(
        "fedcba9876543210",
        CachedEmbedding(
            tensor=torch.ones_like(cached_tensor),
            image_grid_thw=[1, 1, 2],
        ),
    )
    assert cache_handler._embedding_cache.get(cache_key) is None

    grid, embeddings, entries = await cache_handler._encode_with_cache(
        [None],
        [cache_key],
        Modality.IMAGE,
        prechecked_entries={0: prechecked_entry},
    )

    cache_handler.encoder.encode_mock.assert_not_called()
    assert grid.tolist() == [[1, 1, 2]]
    assert torch.equal(embeddings, cached_tensor)
    assert len(entries) == 1
    assert entries[0] is cached_entry


@pytest.mark.asyncio
async def test_encode_with_cache_reencodes_only_unkeyed_items(
    cache_handler: MultimodalEncodeWorkerHandler,
) -> None:
    media_inputs = ["cacheable", "old-decoded-descriptor"]
    cache_keys = ["0123456789abcdef", None]
    first_embeddings = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    cache_handler.encoder.encode_mock.return_value = (
        torch.tensor([[1, 1, 1], [1, 1, 1]]),
        first_embeddings,
        None,
    )

    _, encoded, _ = await cache_handler._encode_with_cache(
        media_inputs, cache_keys, Modality.IMAGE
    )
    assert torch.equal(encoded, first_embeddings)
    assert cache_handler._embedding_cache.keys() == [cache_keys[0]]

    cache_handler.encoder.encode_mock.reset_mock()
    cache_handler.encoder.encode_mock.return_value = (
        torch.tensor([1, 1, 1]),
        torch.tensor([[5.0, 6.0]]),
        None,
    )

    _, encoded_again, _ = await cache_handler._encode_with_cache(
        media_inputs, cache_keys, Modality.IMAGE
    )

    cache_handler.encoder.encode_mock.assert_awaited_once_with(
        [media_inputs[1]], Modality.IMAGE
    )
    assert torch.equal(encoded_again, torch.tensor([[1.0, 2.0], [5.0, 6.0]]))


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


@pytest.mark.asyncio
async def test_video_request_skips_cache_key_when_cache_is_disabled(
    cache_handler: MultimodalEncodeWorkerHandler,
) -> None:
    video_url = "https://example.com/clip.mp4"
    video_token_id = cache_handler.video_token_id
    cache_handler._embedding_cache = None
    cache_handler._image_loader = None
    cache_handler._media_cache_key = Mock(
        side_effect=AssertionError("video cache key must not be computed")
    )
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

    cache_handler.embedding_sender = _DummyEmbeddingSender()
    cache_handler.pd_worker_client = _DummyPdWorkerClient()
    raw_request = {
        "token_ids": [101, video_token_id, 102],
        "stop_conditions": {"max_tokens": 8},
        "sampling_options": {"temperature": 0.0},
        "multi_modal_data": {"video_url": [{"Url": video_url}]},
    }

    outputs = [item async for item in cache_handler.generate(raw_request, context=None)]

    assert outputs == [{"token_ids": [7]}]
    cache_handler._media_cache_key.assert_not_called()
    cache_handler.encoder.encode_mock.assert_awaited_once_with(
        [video_url], Modality.VIDEO
    )


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
    await cache_handler._encode_with_cache([video_url], [first_key], Modality.VIDEO)

    cache_handler.encoder.vision_config["video"]["fps"] = 4.0

    second_key = cache_handler._media_cache_key(
        video_url, Modality.VIDEO, cache_handler.encoder
    )
    await cache_handler._encode_with_cache([video_url], [second_key], Modality.VIDEO)
    await cache_handler._encode_with_cache([video_url], [second_key], Modality.VIDEO)

    assert first_key != second_key
    assert cache_handler.encoder.encode_mock.await_count == 2
    assert cache_handler._embedding_cache.get(first_key) is not None
    assert cache_handler._embedding_cache.get(second_key) is not None


# --- NVDEC video routing -------------------------------------------------------

_HANDLER_MOD = "dynamo.sglang.request_handlers.multimodal.encode_worker_handler"


# A data: URI passes the url policy without touching the network, so the
# decoder-gating tests below exercise gating rather than DNS.
_INLINE_VIDEO = "data:video/mp4;base64,AAAAIGZ0eXBpc29t"
# http:// is refused on scheme alone (allow_http defaults False), so the
# validation tests are deterministic and offline too.
_BLOCKED_URL = "http://169.254.169.254/latest/meta-data/"


@pytest.fixture
def nvdec_handler(cache_handler) -> MultimodalEncodeWorkerHandler:
    """cache_handler wired with the attributes the NVDEC path reads."""
    cache_handler.num_video_frames = 32
    return cache_handler


@pytest.mark.asyncio
async def test_disabled_nvdec_without_software_decoder_is_actionable(
    nvdec_handler, monkeypatch
) -> None:
    """NVDEC off (env/CPU/gated model) + no software decoder: the URLs would
    go straight to SGLang and die deep with the payload-blob error, so
    _build_encode_inputs must raise the actionable error up front.

    This is the deployment class MOST likely to lack a decoder entirely --
    reviewers caught that the preflight originally lived only on the
    NVDEC-enabled path and never ran here.
    """
    from dynamo.common.multimodal.codec_errors import MissingMediaDecoderError

    monkeypatch.setenv("DYN_DISABLE_NVDEC", "1")
    monkeypatch.setattr(
        f"{_HANDLER_MOD}._software_video_decoder_imports", lambda: False
    )

    with pytest.raises(MissingMediaDecoderError) as exc_info:
        await nvdec_handler._build_encode_inputs([_INLINE_VIDEO], "VIDEO")

    assert "install_media_decoders sglang" in str(exc_info.value)


@pytest.mark.asyncio
async def test_disabled_nvdec_with_software_decoder_passes_urls(
    nvdec_handler, monkeypatch
) -> None:
    """NVDEC off but a software decoder exists: URLs pass through unchanged
    (SGLang fetches and decodes them itself, as before)."""
    monkeypatch.setenv("DYN_DISABLE_NVDEC", "1")
    monkeypatch.setattr(f"{_HANDLER_MOD}._software_video_decoder_imports", lambda: True)

    out = await nvdec_handler._build_encode_inputs([_INLINE_VIDEO], "VIDEO")
    assert out == [_INLINE_VIDEO]


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
    """Each URL maps positionally to a decoder, bytes, or the URL itself.

    Three outcomes, not two: a decoder for H.264/H.265, the already-fetched
    bytes when the codec is not hardware-routed, and the URL only when nothing
    was fetched at all.
    """
    monkeypatch.setattr(f"{_HANDLER_MOD}.nvdec_available", lambda: True)
    nvdec_handler.encoder.model_type = "qwen2_5_vl"
    decoder = object()  # stand-in for the NvdecVideoDecoder

    async def _decode(url):
        if url.endswith("h264.mp4"):
            return decoder
        if url.endswith("vp9.webm"):
            return b"vp9-bytes"  # fetched under policy, handed straight over
        return None  # nothing fetched (unsupported scheme)

    monkeypatch.setattr(nvdec_handler, "_maybe_nvdec_decoder", _decode)
    urls = ["https://x/h264.mp4", "https://x/vp9.webm", "ftp://x/other.mkv"]
    out = await nvdec_handler._build_encode_inputs(urls, "VIDEO")
    assert out[0] is decoder
    assert out[1] == b"vp9-bytes"
    assert out[2] == "ftp://x/other.mkv"


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
async def test_maybe_nvdec_decoder_returns_bytes_for_non_hw_codec(
    nvdec_handler, monkeypatch
) -> None:
    """VP9 (not HW-routed) yields the fetched bytes, never the URL.

    Returning the URL made SGLang download the same payload a second time: it
    doubles ingress and latency, fails outright on a one-use signed URL, and --
    because SGLang applies no URL policy of its own, fetching http(s) straight
    through ``get_mm_http_session`` -- means the bytes it decoded were never the
    ones this handler validated, with a redirect free to resolve differently
    between the two fetches.

    Reported by Codex on #11836. Measured on GPU against a counting HTTP server:
    two origin fetches for one request before this change, one after.
    """
    monkeypatch.setattr(
        f"{_HANDLER_MOD}.validate_media_url",
        AsyncMock(return_value="https://x/clip.webm"),
    )
    fetch = AsyncMock(return_value=b"vp9-bytes")
    monkeypatch.setattr(f"{_HANDLER_MOD}.fetch_bytes", fetch)
    monkeypatch.setattr(f"{_HANDLER_MOD}.probe_video_codec", lambda _b: "vp9")
    monkeypatch.setattr(f"{_HANDLER_MOD}.should_use_nvdec", lambda c: c == "h264")
    # The passthrough contract now holds only when SGLang can actually decode
    # the bytes; the codec-compliant test image ships no software decoder, so
    # stub the preflight probe. The absent-decoder leg (actionable error) is
    # covered in test_sglang_multimodal_video.py.
    monkeypatch.setattr(f"{_HANDLER_MOD}._software_video_decoder_imports", lambda: True)

    out = await nvdec_handler._maybe_nvdec_decoder("https://x/clip.webm")

    assert out == b"vp9-bytes"
    assert fetch.await_count == 1  # fetched once, and that copy is what travels on


@pytest.mark.asyncio
async def test_maybe_nvdec_decoder_rejects_non_http_scheme(
    nvdec_handler, monkeypatch
) -> None:
    """A scheme that is neither http(s) nor a permitted local source is skipped.

    Note file:// and data: are *not* in that group -- they are read through
    read_local_media_bytes when the policy allows -- so this uses a scheme with
    no reader at all. Validation is stubbed as succeeding so the assertion is
    about scheme dispatch rather than about the policy refusing it.
    """
    monkeypatch.setattr(
        f"{_HANDLER_MOD}.validate_media_url",
        AsyncMock(return_value="ftp://example.invalid/clip.mp4"),
    )
    fetch = AsyncMock()
    monkeypatch.setattr(f"{_HANDLER_MOD}.fetch_bytes", fetch)
    assert (
        await nvdec_handler._maybe_nvdec_decoder("ftp://example.invalid/clip.mp4")
        is None
    )
    fetch.assert_not_called()
    fetch.assert_not_called()


@pytest.mark.asyncio
async def test_maybe_nvdec_decoder_falls_back_on_error(
    nvdec_handler, monkeypatch
) -> None:
    """A failed FETCH degrades to URL passthrough, because there are no bytes.

    Distinct from a failed decoder build, which returns the bytes it already has
    (see test_decode_failure_falls_back_to_the_fetched_bytes). The URL is the
    fallback only when nothing was retrieved.
    """
    monkeypatch.setattr(
        f"{_HANDLER_MOD}.validate_media_url",
        AsyncMock(return_value="https://x/clip.mp4"),
    )
    monkeypatch.setattr(
        f"{_HANDLER_MOD}.fetch_bytes", AsyncMock(side_effect=RuntimeError("boom"))
    )
    assert await nvdec_handler._maybe_nvdec_decoder("https://x/clip.mp4") is None


@pytest.mark.asyncio
async def test_policy_rejection_is_not_swallowed_into_url_passthrough(
    nvdec_handler, monkeypatch
) -> None:
    """A refused URL must fail the request, not fall through to SGLang.

    SGLang applies no URL policy: it fetches http(s) directly and resolves
    file:// to a bare path. Returning None here restores the original URL, so a
    refusal became an unvalidated fetch. Verified on GPU before the guard
    existed -- a loopback URL the policy rejected was fetched anyway, and a
    refused file:// path resolved to a readable local file.
    """
    from dynamo.common.http.url_validator import UrlValidationError

    monkeypatch.setattr(f"{_HANDLER_MOD}.nvdec_available", lambda: True)
    nvdec_handler.encoder.model_type = "qwen2_5_vl"
    monkeypatch.setattr(
        f"{_HANDLER_MOD}.validate_media_url",
        AsyncMock(side_effect=UrlValidationError("blocked IP")),
    )
    fetch = AsyncMock()
    monkeypatch.setattr(f"{_HANDLER_MOD}.fetch_bytes", fetch)

    with pytest.raises(UrlValidationError):
        await nvdec_handler._maybe_nvdec_decoder(
            "http://169.254.169.254/latest/meta-data"
        )
    fetch.assert_not_called()

    # ...and it propagates through the batch builder rather than being mapped
    # back to the URL there.
    with pytest.raises(UrlValidationError):
        await nvdec_handler._build_encode_inputs(
            ["http://169.254.169.254/latest/meta-data"], "VIDEO"
        )


@pytest.mark.asyncio
async def test_decode_failure_falls_back_to_the_fetched_bytes(
    nvdec_handler, monkeypatch
) -> None:
    """A genuine decode failure degrades to the bytes, not to an error or the URL.

    Two things at once: the refusal guard must not turn a decode failure into a
    raised error, and the fallback must not hand back the URL -- the bytes are
    already here and were fetched under policy, so making SGLang refetch them
    would reintroduce the second request for the one case NVDEC was meant to
    help.
    """
    monkeypatch.setattr(f"{_HANDLER_MOD}.nvdec_available", lambda: True)
    nvdec_handler.encoder.model_type = "qwen2_5_vl"
    monkeypatch.setattr(
        f"{_HANDLER_MOD}.validate_media_url",
        AsyncMock(return_value="https://x/clip.mp4"),
    )
    monkeypatch.setattr(f"{_HANDLER_MOD}.fetch_bytes", AsyncMock(return_value=b"bytes"))
    monkeypatch.setattr(f"{_HANDLER_MOD}.probe_video_codec", lambda _b: "h264")
    monkeypatch.setattr(f"{_HANDLER_MOD}.should_use_nvdec", lambda _c: True)

    def _boom(_data):
        raise RuntimeError("NVDEC init failed")

    monkeypatch.setattr(f"{_HANDLER_MOD}.NvdecVideoDecoder", _boom)
    # Bytes fallback is only valid when SGLang can decode them; stub the
    # preflight probe (the image ships no software decoder).
    monkeypatch.setattr(f"{_HANDLER_MOD}._software_video_decoder_imports", lambda: True)
    assert await nvdec_handler._maybe_nvdec_decoder("https://x/clip.mp4") == b"bytes"
    out = await nvdec_handler._build_encode_inputs([_INLINE_VIDEO], "VIDEO")
    assert out == [b"bytes"]


@pytest.mark.asyncio
@pytest.mark.parametrize("software_decoder_present", [False, True])
async def test_disabled_nvdec_validates_url_before_reporting_decoders(
    nvdec_handler, monkeypatch, software_decoder_present
) -> None:
    """A source the policy refuses must be refused here, whatever the decoders.

    With NVDEC off, SGLang fetches these URLs with its own session and never
    consults our policy, so this is the only place the policy can apply. Two
    failures this pins, both reproduced on the real image: a blocked URL was
    handed back for SGLang to fetch when a decoder was present, and answered
    with "install decord2" -- deployment configuration, in response to a
    request we should refuse -- when one was not.
    """
    monkeypatch.setenv("DYN_DISABLE_NVDEC", "1")
    monkeypatch.setattr(
        f"{_HANDLER_MOD}._software_video_decoder_imports",
        lambda: software_decoder_present,
    )
    # Undo the fixture's passthrough: validation is what this test asserts.
    # The URL below is refused on scheme, so this still performs no lookup.
    monkeypatch.setattr(f"{_HANDLER_MOD}.validate_media_url", validate_media_url)

    with pytest.raises(UrlValidationError):
        await nvdec_handler._build_encode_inputs([_BLOCKED_URL], "VIDEO")


@pytest.mark.asyncio
async def test_decode_failure_without_software_decoder_is_actionable(
    nvdec_handler, monkeypatch
) -> None:
    """NVDEC failed AND no software decoder exists: the bytes would only die
    deep inside SGLang with the payload repr in the message, so the fallback
    must raise the actionable error instead of passing them on."""
    from dynamo.common.multimodal.codec_errors import MissingMediaDecoderError

    monkeypatch.setattr(f"{_HANDLER_MOD}.nvdec_available", lambda: True)
    nvdec_handler.encoder.model_type = "qwen2_5_vl"
    monkeypatch.setattr(
        f"{_HANDLER_MOD}.validate_media_url",
        AsyncMock(return_value="https://x/clip.mp4"),
    )
    monkeypatch.setattr(f"{_HANDLER_MOD}.fetch_bytes", AsyncMock(return_value=b"bytes"))
    monkeypatch.setattr(f"{_HANDLER_MOD}.probe_video_codec", lambda _b: "h264")
    monkeypatch.setattr(f"{_HANDLER_MOD}.should_use_nvdec", lambda _c: True)

    def _boom(_data):
        raise RuntimeError("NVDEC init failed")

    monkeypatch.setattr(f"{_HANDLER_MOD}.NvdecVideoDecoder", _boom)
    monkeypatch.setattr(
        f"{_HANDLER_MOD}._software_video_decoder_imports", lambda: False
    )

    with pytest.raises(MissingMediaDecoderError) as exc_info:
        await nvdec_handler._maybe_nvdec_decoder("https://x/clip.mp4")

    # h264 + NVDEC "available" but failing: the message still leads with the
    # capability/hardware framing and carries the install remedy.
    assert "install_media_decoders sglang" in str(exc_info.value)


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
    # __new__ skips __init__, so set _tmp_path explicitly: the finaliser reads it
    # and would otherwise raise AttributeError during GC, which pytest surfaces
    # as PytestUnraisableExceptionWarning -- an error under this repo's
    # filterwarnings=error.
    sentinel = VideoDecoderWrapper.__new__(VideoDecoderWrapper)
    sentinel._tmp_path = None
    assert patched(sentinel) is sentinel

    # Anything else still goes to the original implementation.
    with pytest.raises(Exception):
        patched(object())

    _install_load_video_passthrough()  # idempotent
    assert encode_server.load_video is patched
