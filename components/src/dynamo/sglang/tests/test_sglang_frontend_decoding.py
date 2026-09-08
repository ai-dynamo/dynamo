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
from dynamo.llm import HttpError
from dynamo.sglang.backend_args import DynamoSGLangConfig
from dynamo.sglang.request_handlers.llm.decode_handler import (
    DecodeWorkerHandler,
    FrontendDecodedVideo,
)
from dynamo.sglang.request_handlers.llm.prefill_handler import PrefillWorkerHandler
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
    handler._max_input_token_id = 41

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
async def test_encode_worker_rejects_unknown_oov_token_before_loading_media():
    handler = MultimodalEncodeWorkerHandler.__new__(MultimodalEncodeWorkerHandler)
    handler.image_token_id = 32000
    handler.video_token_id = None
    handler._max_input_token_id = 31999
    handler._prepare_image_inputs = AsyncMock()
    raw_request = {
        "token_ids": [1, 32000, 2**32 - 1],
        "stop_conditions": {"max_tokens": 8},
        "sampling_options": {"temperature": 0.0},
        "multi_modal_data": {"image_url": [{"Url": "https://example.com/image.png"}]},
    }

    with pytest.raises(HttpError, match="4294967295"):
        async for _ in handler.generate(raw_request, context=None):
            pass

    handler._prepare_image_inputs.assert_not_awaited()


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


@pytest.mark.asyncio
async def test_encode_worker_skips_image_cache_keys_when_cache_is_disabled():
    handler = MultimodalEncodeWorkerHandler.__new__(MultimodalEncodeWorkerHandler)
    handler._embedding_cache = None
    handler._decoded_content_hash_warning_emitted = False
    handler._image_loader = None
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
    if media_key == "image_url":
        image_items, _ = handler._extract_media_inputs(request)
        with pytest.raises(ValueError, match=message):
            await handler._prepare_image_inputs(image_items)
    else:
        with pytest.raises(ValueError, match="Unsupported video_url item"):
            handler._extract_media_inputs(request)


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
    handler._max_input_token_id = 41

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
    handler._first_token_source = None
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
    decoded_metadata = {"shape": [2, 4, 4, 3], "dtype": "uint8"}
    request = {
        "token_ids": [1, 2, 3],
        "multi_modal_data": {"video_url": [{"Decoded": decoded_metadata}]},
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


# NVBug 6418893 — SGLang session radix wiring.
#
# Dynamo used to derive `session_params={"id": <session_id>}` from a request's
# `agent_context.session_id` and pass it to `engine.async_generate`. SGLang
# treats `session_params.id` as an explicit session lifecycle and rejects any id
# that was not created through `open_session`, so every request carrying an
# `agent_context.session_id` failed against an SGLang server. Commit `d245a5be3`
# removed that wiring but added no test guarding its return. These tests assert
# the corrected contract at the engine seam: no handler may synthesize
# `session_params` from `agent_context`. See the "Session identity" note in
# components/src/dynamo/sglang/AGENTS.md.


class _GenerateRecorder:
    """Stands in for ``sgl.Engine``, recording each ``async_generate`` call.

    ``calls`` holds one keyword-argument dict per call, so a test can assert both
    what reached the engine and that the engine was reached exactly once. The
    ``**kwargs`` signature matters: ``filter_supported_async_generate_kwargs``
    inspects it and forwards every kwarg when it finds a var-keyword parameter.
    """

    def __init__(self) -> None:
        self.calls: list[Dict[str, Any]] = []

    async def async_generate(
        self, **kwargs: Any
    ) -> AsyncGenerator[Dict[str, Any], None]:
        self.calls.append(kwargs)
        return _empty_stream()


# The deleted helper read the flag through
# `getattr(server_args, "enable_session_radix_cache", False)`, so an
# attribute-free SimpleNamespace is exactly the fixture its default-False
# behavior would have satisfied silently. Pinning all three states means no
# future server_args flag can quietly reopen the path.
_SESSION_RADIX_SERVER_ARGS = [
    pytest.param({"enable_session_radix_cache": True}, id="radix_flag_true"),
    pytest.param({"enable_session_radix_cache": False}, id="radix_flag_false"),
    pytest.param({}, id="radix_flag_absent"),
]

_ADVERSARIAL_AGENT_CONTEXTS = [
    pytest.param({}, id="no_agent_context_key"),
    pytest.param({"agent_context": None}, id="null_agent_context"),
    pytest.param({"agent_context": {}}, id="empty_agent_context"),
    pytest.param({"agent_context": {"session_id": ""}}, id="empty_session_id"),
    pytest.param({"agent_context": {"session_id": None}}, id="null_session_id"),
    pytest.param({"agent_context": {"session_id": 123}}, id="int_session_id"),
    pytest.param(
        {"agent_context": {"session_id": {"id": "s-1"}}}, id="dict_session_id"
    ),
]

_SESSION_AGENT_CONTEXT = {"agent_context": {"session_id": "s-1"}}


def _set_server_args(handler: Any, extra_fields: Dict[str, Any]) -> None:
    """Rebuild ``handler.config.server_args`` with the given extra attributes."""
    handler.config = SimpleNamespace(
        server_args=SimpleNamespace(served_model_name="test-model", **extra_fields)
    )


def _new_prefill_handler() -> PrefillWorkerHandler:
    """Build a PrefillWorkerHandler without invoking sgl.Engine.

    Same bypass-``__init__`` pattern as ``_new_decode_handler`` above and as
    test_sglang_decode_handler.py.
    """
    handler = PrefillWorkerHandler.__new__(PrefillWorkerHandler)
    handler.use_sglang_tokenizer = False
    handler.enable_trace = False
    handler.serving_mode = DisaggregationMode.PREFILL
    handler.config = SimpleNamespace(
        server_args=SimpleNamespace(served_model_name="test-model")
    )
    handler.bootstrap_host = "127.0.0.1"
    handler.bootstrap_port = 1234
    handler._generate_bootstrap_room = lambda: 7
    handler._consume_tasks = set()

    @asynccontextmanager
    async def no_cancellation_monitor(*args, **kwargs):
        yield None

    handler._cancellation_monitor = no_cancellation_monitor

    handler._get_input_param = lambda req: {"input_ids": req.get("token_ids", [])}
    handler._resolve_lora = lambda req: None
    handler._priority_kwargs = lambda priority: {}

    return handler


async def _capture_aggregated_kwargs(
    request_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """Run one aggregated decode request; return the recorded engine kwargs.

    Asserts the engine was called exactly once. Without that check, an
    "``x`` is absent" assertion would also pass when the handler never reached
    the engine at all.
    """
    handler = _new_decode_handler(enable_frontend_decoding=False)
    recorder = _GenerateRecorder()
    handler.engine = recorder

    request: Dict[str, Any] = {"token_ids": [1, 2, 3], "multi_modal_data": {}}
    request.update(request_overrides)

    async for _ in handler.generate(request, _Context()):
        pass

    assert len(recorder.calls) == 1
    return recorder.calls[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("server_args_fields", _SESSION_RADIX_SERVER_ARGS)
async def test_aggregated_decode_omits_session_params_for_agent_context(
    server_args_fields,
):
    """Aggregated decode must not turn ``agent_context.session_id`` into
    ``session_params`` (NVBug 6418893).

    The absent-key assertion is deliberate, not a stale leftover: reverting
    commit ``d245a5be3`` restores ``_session_kwargs`` and makes this fail.
    """
    handler = _new_decode_handler(enable_frontend_decoding=False)
    _set_server_args(handler, server_args_fields)
    recorder = _GenerateRecorder()
    handler.engine = recorder

    request = {
        "token_ids": [1, 2, 3],
        "multi_modal_data": {},
        **_SESSION_AGENT_CONTEXT,
    }

    async for _ in handler.generate(request, _Context()):
        pass

    assert len(recorder.calls) == 1
    captured = recorder.calls[0]
    assert captured["input_ids"] == [1, 2, 3]
    assert "session_params" not in captured


@pytest.mark.asyncio
@pytest.mark.parametrize("server_args_fields", _SESSION_RADIX_SERVER_ARGS)
async def test_disaggregated_decode_omits_session_params_for_agent_context(
    server_args_fields,
):
    """Disaggregated decode must not attach ``session_params`` either
    (NVBug 6418893).

    Commit ``d245a5be3`` removed a separate ``_session_kwargs`` call site on this
    branch of ``DecodeWorkerHandler.generate``, so it needs its own test.
    """
    handler = _new_decode_handler(enable_frontend_decoding=False)
    handler.serving_mode = DisaggregationMode.DECODE
    _set_server_args(handler, server_args_fields)
    recorder = _GenerateRecorder()
    handler.engine = recorder

    request = {
        "token_ids": [1, 2, 3],
        "multi_modal_data": {},
        "bootstrap_info": {
            "bootstrap_host": "127.0.0.1",
            "bootstrap_port": 1234,
            "bootstrap_room": 7,
        },
        **_SESSION_AGENT_CONTEXT,
    }

    async for _ in handler.generate(request, _Context()):
        pass

    assert len(recorder.calls) == 1
    captured = recorder.calls[0]
    assert captured["input_ids"] == [1, 2, 3]
    assert captured["bootstrap_room"] == 7
    assert "session_params" not in captured


@pytest.mark.asyncio
@pytest.mark.parametrize("server_args_fields", _SESSION_RADIX_SERVER_ARGS)
async def test_prefill_omits_session_params_for_agent_context(server_args_fields):
    """Prefill must not attach ``session_params`` either (NVBug 6418893).

    The third ``_session_kwargs`` call site removed by commit ``d245a5be3`` was
    in ``PrefillWorkerHandler.generate``.
    """
    handler = _new_prefill_handler()
    _set_server_args(handler, server_args_fields)
    recorder = _GenerateRecorder()
    handler.engine = recorder

    request = {
        "token_ids": [1, 2, 3],
        "sampling_options": {},
        "stop_conditions": {},
        **_SESSION_AGENT_CONTEXT,
    }

    async for _ in handler.generate(request, _Context()):
        pass

    assert len(recorder.calls) == 1
    captured = recorder.calls[0]
    assert captured["input_ids"] == [1, 2, 3]
    assert "session_params" not in captured


@pytest.mark.asyncio
@pytest.mark.parametrize("request_overrides", _ADVERSARIAL_AGENT_CONTEXTS)
async def test_aggregated_decode_tolerates_adversarial_agent_context(
    request_overrides,
):
    """Missing, empty, and wrong-typed session ids reach the engine unchanged.

    The no-raise half carries as much weight as the absent key. The removed
    ``_session_id`` helper guarded on ``isinstance(session_id, str)``; a
    reintroduction that drops the guard surfaces as an exception rather than a
    wrong keyword argument, and only running these inputs catches that variant.
    An exception anywhere in ``generate`` fails the test.
    """
    captured = await _capture_aggregated_kwargs(request_overrides)

    assert captured["input_ids"] == [1, 2, 3]
    assert "session_params" not in captured


@pytest.mark.asyncio
async def test_agent_context_contributes_no_engine_kwargs():
    """Control: an ``agent_context`` changes nothing about the engine payload.

    Asserting only that ``session_params`` is absent would still pass if a
    handler grew some other ``agent_context``-derived keyword argument.
    Comparing the whole key set against an otherwise identical request pins that
    ``agent_context`` contributes nothing. This control stays green under
    refactors of the kwargs assembly and goes red only when ``agent_context``
    starts contributing a key.
    """
    with_context = await _capture_aggregated_kwargs(_SESSION_AGENT_CONTEXT)
    without_context = await _capture_aggregated_kwargs({})

    assert set(with_context) == set(without_context)
    assert "session_params" not in with_context
