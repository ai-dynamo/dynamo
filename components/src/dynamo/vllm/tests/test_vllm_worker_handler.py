# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for WorkerHandler in combination with multimodal handling."""
# [gluo FIXME] This suite of tests is added for MultimodalPDWorkerHandler,
# which is now removed. Yet the concept of this tests is still valid that
# we need to have unit tests for the worker handlers.
# Need to revisit the tests and update them to test the worker handlers.

import json
from collections import defaultdict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

import dynamo.vllm.handlers as mod
from dynamo.common.memory.multimodal_embedding_cache_manager import (
    MultimodalEmbeddingCacheManager,
)
from dynamo.vllm.multimodal_utils.protocol import (
    PatchedTokensPrompt,
    vLLMMultimodalRequest,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


# ── Helpers ──────────────────────────────────────────────────────────


def _make_config(
    model: str = "test-model",
    is_prefill_worker: bool = False,
    enable_multimodal: bool = True,
    multimodal_embedding_cache_capacity_gb: float = 0,
    disaggregation_mode: str | None = None,
) -> MagicMock:
    """Create a mock Config with the fields used by MultimodalPDWorkerHandler."""
    from dynamo.vllm.constants import DisaggregationMode, EmbeddingTransferMode

    config = MagicMock()
    config.model = model
    config.is_prefill_worker = is_prefill_worker
    if disaggregation_mode is not None:
        config.disaggregation_mode = getattr(DisaggregationMode, disaggregation_mode)
    elif is_prefill_worker:
        config.disaggregation_mode = DisaggregationMode.PREFILL
    else:
        config.disaggregation_mode = DisaggregationMode.AGGREGATED
    # NIXL_WRITE / NIXL_READ modes require GPU, the tests may run in CPU-only environments,
    # so set to LOCAL mode.
    config.embedding_transfer_mode = EmbeddingTransferMode.LOCAL
    config.enable_multimodal = enable_multimodal
    config.multimodal_embedding_cache_capacity_gb = (
        multimodal_embedding_cache_capacity_gb
    )
    config.engine_args.create_model_config.return_value.get_diff_sampling_param.return_value = (
        {}
    )
    return config


def _make_handler(
    config: MagicMock | None = None,
    encode_worker_client: MagicMock | None = None,
    decode_worker_client: MagicMock | None = None,
) -> mod.DecodeWorkerHandler:
    """Construct a handler with BaseWorkerHandler.__init__ bypassed."""
    if config is None:
        config = _make_config()
    model_config = MagicMock(enable_prompt_embeds=True)
    with patch.object(mod.BaseWorkerHandler, "__init__", return_value=None):
        handler = mod.DecodeWorkerHandler(
            runtime=MagicMock(),
            config=config,
            engine=MagicMock(),
            default_sampling_params={},
            model_config=model_config,
            encode_worker_client=encode_worker_client,
        )
    handler.model_config = model_config
    return handler


def _make_raw_frontend_request(image_urls: list[str] | None = None) -> dict:
    """Build a raw dict that mimics what the Rust frontend sends."""
    mm_data = None
    if image_urls:
        mm_data = {
            "image_url": [{"Url": url} for url in image_urls],
        }
    return {
        "token_ids": [1, 2, 3],
        "multi_modal_data": mm_data,
        "sampling_options": {},
        "stop_conditions": {},
        "output_options": {},
    }


def _make_vllm_request(request_id: str = "req-1") -> vLLMMultimodalRequest:
    """Build a minimal vLLMMultimodalRequest."""
    from vllm.sampling_params import SamplingParams

    return vLLMMultimodalRequest(
        engine_prompt=PatchedTokensPrompt(prompt_token_ids=[1, 2, 3]),
        sampling_params=SamplingParams(),
        request_id=request_id,
        multimodal_inputs=[],
    )


def _make_engine_response(request_id: str = "req-1", finished: bool = True):
    """Create a mock engine response with the fields _format_engine_output needs."""
    resp = MagicMock()
    resp.request_id = request_id
    resp.prompt = "test"
    resp.prompt_token_ids = [1, 2, 3]
    resp.prompt_logprobs = None
    resp.outputs = []
    resp.finished = finished
    resp.metrics = None
    resp.kv_transfer_params = {"do_remote_decode": False}
    return resp


# ── Tests ────────────────────────────────────────────────────────────


@pytest.mark.skip(reason="Need to revisit tests, see comment at top of the file")
class TestInit:
    def test_embedding_cache_created_when_capacity_set(self):
        capacity_gb = 0.1
        handler = _make_handler(
            config=_make_config(multimodal_embedding_cache_capacity_gb=capacity_gb)
        )
        assert isinstance(
            handler.embedding_cache_manager, MultimodalEmbeddingCacheManager
        )
        expected_bytes = int(capacity_gb * 1024**3)
        assert handler.embedding_cache_manager._capacity_bytes == expected_bytes


@pytest.mark.skip(reason="Need to revisit tests, see comment at top of the file")
class TestParseFrontendRequest:
    def test_extracts_token_ids_and_sampling_params(self):
        """Parses token_ids and sampling_params from raw frontend dict."""
        handler = _make_handler()
        handler.default_sampling_params = {}

        raw = _make_raw_frontend_request()
        request, image_urls = handler._parse_frontend_request(raw)

        assert request.engine_prompt["prompt_token_ids"] == [1, 2, 3]
        assert image_urls == []

    def test_extracts_image_urls(self):
        """Extracts image URLs from multi_modal_data."""
        handler = _make_handler()
        handler.default_sampling_params = {}

        raw = _make_raw_frontend_request(image_urls=["http://a.png", "http://b.png"])
        request, image_urls = handler._parse_frontend_request(raw)

        assert image_urls == ["http://a.png", "http://b.png"]


@pytest.mark.skip(reason="Need to revisit tests, see comment at top of the file")
class TestLoadMultimodalData:
    @pytest.mark.asyncio
    async def test_no_encode_client_returns_empty(self):
        """Without encode client -> returns empty dict."""
        handler = _make_handler(encode_worker_client=None)
        mm_data = await handler._load_multimodal_data(["http://img.png"], "req-1")
        assert len(mm_data) == 0

    @pytest.mark.asyncio
    async def test_no_images_returns_empty(self):
        """With encode client but no images -> returns empty dict."""
        handler = _make_handler(encode_worker_client=MagicMock())
        mm_data = await handler._load_multimodal_data([], "req-1")
        assert len(mm_data) == 0

    @pytest.mark.asyncio
    async def test_delegates_to_load_multimodal_embeddings(self):
        """With encode client -> delegates to load_multimodal_embeddings."""
        mock_client = MagicMock()
        handler = _make_handler(encode_worker_client=mock_client)

        fake_mm_data = defaultdict(list, {"image": torch.randn(1, 10)})  # type: ignore
        with patch.object(
            handler.embedding_loader,
            "load_multimodal_embeddings",
            new_callable=AsyncMock,
            return_value=fake_mm_data,
        ) as mock_load:
            result = await handler._load_multimodal_data(["http://img.png"], "req-1")

        mock_load.assert_awaited_once()
        assert result is fake_mm_data

    @pytest.mark.asyncio
    async def test_passes_model(self):
        """Model name is forwarded."""
        mock_client = MagicMock()
        handler = _make_handler(encode_worker_client=mock_client)

        with patch.object(
            handler.embedding_loader,
            "load_multimodal_embeddings",
            new_callable=AsyncMock,
            return_value=defaultdict(list),
        ) as mock_load:
            await handler._load_multimodal_data(["http://img.png"], "req-1")

        assert mock_load.call_args.kwargs["model"] == handler.config.model


@pytest.mark.skip(reason="Need to revisit tests, see comment at top of the file")
class TestGenerateAgg:
    @pytest.mark.asyncio
    async def test_streams_serialized_responses(self):
        """_generate_agg yields dicts formatted by _format_engine_output."""
        handler = _make_handler()
        request = _make_vllm_request()
        engine_resp = _make_engine_response()

        output = MagicMock()
        output.token_ids = [10, 11]
        output.finish_reason = "stop"
        output.stop_reason = None
        engine_resp.outputs = [output]

        async def fake_generate(**kwargs):
            yield engine_resp

        handler.engine_client = MagicMock()
        handler.engine_client.generate = fake_generate

        chunks = []
        async for chunk in handler._generate_agg(request, {"image": []}):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0]["token_ids"] == [10, 11]
        assert chunks[0]["finish_reason"] == "stop"


@pytest.mark.skip(reason="Need to revisit tests, see comment at top of the file")
class TestGenerateDisagg:
    @pytest.mark.asyncio
    async def test_prefills_then_forwards_to_decode(self):
        """_generate_disagg prefills locally, then round-robins to decode worker."""
        config = _make_config(model="test-model", is_prefill_worker=True)
        decode_client = MagicMock()
        handler = _make_handler(config=config, decode_worker_client=decode_client)
        handler.engine_client = MagicMock()

        prefill_resp = _make_engine_response()
        prefill_resp.kv_transfer_params = {"block_ids": [0, 1]}

        async def fake_generate(**kwargs):
            yield prefill_resp

        handler.engine_client.generate = fake_generate

        decode_json = json.dumps(
            {
                "request_id": "req-1",
                "prompt": "test",
                "prompt_token_ids": [1, 2, 3],
                "outputs": [
                    {
                        "index": 0,
                        "text": "",
                        "token_ids": [42],
                        "cumulative_logprob": None,
                        "logprobs": None,
                        "finish_reason": "stop",
                        "stop_reason": None,
                    }
                ],
                "finished": True,
                "kv_transfer_params": {"block_ids": [0, 1]},
            }
        )
        decode_resp = MagicMock()
        decode_resp.data.return_value = decode_json

        async def fake_round_robin(payload, context=None):
            async def _stream():
                yield decode_resp

            return _stream()

        decode_client.round_robin = fake_round_robin

        request = _make_vllm_request()
        chunks = []
        async for chunk in handler._generate_disagg(request, {"image": []}):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert isinstance(chunks[0], dict)
        assert chunks[0]["token_ids"] == [42]
        assert chunks[0]["finish_reason"] == "stop"


# ── Decode worker multimodal branching tests ───────────────────────


def _make_decode_handler(
    model: str = "test-model",
    disaggregation_mode: str = "DECODE",
) -> mod.DecodeWorkerHandler:
    """Construct a DecodeWorkerHandler with mocked internals."""
    config = _make_config(model=model, disaggregation_mode=disaggregation_mode)
    model_config = MagicMock(enable_prompt_embeds=True)
    with patch.object(mod.BaseWorkerHandler, "__init__", return_value=None):
        handler = mod.DecodeWorkerHandler(
            runtime=MagicMock(),
            config=config,
            engine=MagicMock(),
            default_sampling_params={},
            model_config=model_config,
        )
    handler.config = config
    handler.model_config = model_config
    handler.enable_multimodal = True
    handler.image_loader = MagicMock()
    handler.embedding_loader = None
    handler.model_max_len = 4096
    handler.default_sampling_params = {}
    handler.kv_event_publisher = None
    handler.otel_tracing_enabled = False
    handler.input_param_manager = MagicMock()
    handler.input_param_manager.get_extra_params.return_value = {}
    return handler


@pytest.mark.asyncio(loop_scope="function")
class TestDecodeWorkerMultimodalBranching:
    """Tests for the mode-aware multimodal branching in _generate_token_mode."""

    async def test_decode_only_qwen_with_mm_data_no_prefill_result_errors(self):
        """Decode-only Qwen worker receiving mm request without prefill_result -> error."""
        handler = _make_decode_handler(
            model="Qwen/Qwen3-VL-2B-Instruct",
            disaggregation_mode="DECODE",
        )
        request = {
            "token_ids": [1, 2, 3],
            "multi_modal_data": {"image_url": [{"Url": "http://img.png"}]},
            "sampling_options": {},
            "stop_conditions": {},
            "output_options": {},
        }
        context = MagicMock()
        chunks = []
        async for chunk in handler._generate_token_mode(request, context, "req-1"):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0]["status"] == "error"
        assert "without prefill result" in chunks[0]["message"]

    async def test_decode_only_qwen_missing_embedding_params_errors(self):
        """Decode-only Qwen VL with prefill_result but no embedding_params -> error."""
        handler = _make_decode_handler(
            model="Qwen/Qwen3-VL-2B-Instruct",
            disaggregation_mode="DECODE",
        )
        request = {
            "token_ids": [1, 2, 3],
            "multi_modal_data": {"image_url": [{"Url": "http://img.png"}]},
            "sampling_options": {},
            "stop_conditions": {},
            "output_options": {},
            "prefill_result": {
                "disaggregated_params": {
                    "kv_transfer_params": {"block_ids": [0]},
                    # embedding_params intentionally missing
                },
            },
        }
        context = MagicMock()
        chunks = []
        async for chunk in handler._generate_token_mode(request, context, "req-1"):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0]["status"] == "error"
        assert "embedding metadata" in chunks[0]["message"]

    async def test_decode_only_non_qwen_proceeds_without_embedding_params(self):
        """Decode-only non-Qwen with prefill_result but no embedding_params -> proceeds.

        Non-Qwen models don't need embedding_params — the KV cache from
        prefill already contains the vision context.
        """
        handler = _make_decode_handler(
            model="llava-hf/llava-1.5-7b-hf",
            disaggregation_mode="DECODE",
        )
        handler._build_prompt_from_request = MagicMock(
            return_value=(None, None, {"status": "error", "message": "test stop"})
        )
        request = {
            "token_ids": [1, 2, 3],
            "multi_modal_data": {"image_url": [{"Url": "http://img.png"}]},
            "sampling_options": {},
            "stop_conditions": {},
            "output_options": {},
            "prefill_result": {
                "disaggregated_params": {
                    "kv_transfer_params": {"block_ids": [0]},
                },
            },
        }
        context = MagicMock()
        chunks = []
        async for chunk in handler._generate_token_mode(request, context, "req-1"):
            chunks.append(chunk)

        # Should reach _build_prompt_from_request (not error at decode guard)
        assert len(chunks) == 1
        assert chunks[0]["message"] == "test stop"

    async def test_aggregated_mode_calls_extract_multimodal_data(self):
        """Aggregated mode handler calls _extract_multimodal_data normally."""
        handler = _make_decode_handler(disaggregation_mode="AGGREGATED")
        handler._extract_multimodal_data = AsyncMock(return_value=None)

        # Return an error from _build_prompt_from_request so _generate_token_mode
        # yields it and returns early — no need to mock the engine.
        handler._build_prompt_from_request = MagicMock(
            return_value=(None, None, {"status": "error", "message": "test stop"})
        )

        request = {
            "token_ids": [1, 2, 3],
            "multi_modal_data": {"image_url": [{"Url": "http://img.png"}]},
            "sampling_options": {},
            "stop_conditions": {},
            "output_options": {},
        }
        context = MagicMock()

        chunks = []
        async for chunk in handler._generate_token_mode(request, context, "req-1"):
            chunks.append(chunk)

        handler._extract_multimodal_data.assert_awaited_once()
        assert len(chunks) == 1
        assert chunks[0]["status"] == "error"


# ── Prefill _build_embedding_params tests ──────────────────────────


def _make_prefill_handler(model: str = "test-model") -> mod.PrefillWorkerHandler:
    """Construct a PrefillWorkerHandler with mocked internals."""
    config = _make_config(
        model=model, is_prefill_worker=True, disaggregation_mode="PREFILL"
    )
    model_config = MagicMock(enable_prompt_embeds=True)
    with patch.object(mod.BaseWorkerHandler, "__init__", return_value=None):
        handler = mod.PrefillWorkerHandler(
            runtime=MagicMock(),
            config=config,
            engine=MagicMock(),
            default_sampling_params={},
            model_config=model_config,
        )
    handler.config = config
    handler.model_config = model_config
    return handler


class TestBuildEmbeddingParams:
    """Tests for PrefillWorkerHandler._build_embedding_params."""

    def test_dict_image_data_produces_embedding_params(self):
        """Dict-style image data with image_embeds + image_grid_thw -> valid params."""
        handler = _make_prefill_handler(model="Qwen/Qwen3-VL-2B-Instruct")
        mm_data = {
            "image": {
                "image_embeds": torch.randn(1, 256, 1024),
                "image_grid_thw": torch.tensor([[1, 16, 16]]),
            }
        }
        result = handler._build_embedding_params(mm_data)

        assert result is not None
        assert "image_grid_thw" in result
        assert "embeddings_shape" in result
        assert result["embeddings_shape"] == [1, 256, 1024]

    def test_pil_image_qwen_computes_grid(self):
        """PIL image for Qwen VL with grid params -> computes valid embedding_params."""
        from PIL import Image

        from dynamo.vllm.multimodal_utils.models.qwen import QwenGridParams

        handler = _make_prefill_handler(model="Qwen/Qwen3-VL-2B-Instruct")
        # Qwen3-VL: patch=16, merge=2, factor=32
        handler._qwen_grid_params = QwenGridParams(
            patch_size=16,
            merge_size=2,
            factor=32,
            min_pixels=65536,
            max_pixels=16777216,
            vision_hidden_dim=2048,
        )

        img = Image.new("RGB", (640, 480))
        result = handler._build_embedding_params({"image": img})

        assert result is not None
        assert result["image_grid_thw"] == [[1, 30, 40]]
        # total_tokens = 1*30*40 // 4 = 300
        assert result["embeddings_shape"] == [300, 2048]

    def test_pil_multi_image_qwen_computes_grid(self):
        """Multiple PIL images for Qwen VL -> computes combined embedding_params."""
        from PIL import Image

        from dynamo.vllm.multimodal_utils.models.qwen import QwenGridParams

        handler = _make_prefill_handler(model="Qwen/Qwen3-VL-2B-Instruct")
        handler._qwen_grid_params = QwenGridParams(
            patch_size=16,
            merge_size=2,
            factor=32,
            min_pixels=65536,
            max_pixels=16777216,
            vision_hidden_dim=2048,
        )

        imgs = [Image.new("RGB", (640, 480)), Image.new("RGB", (320, 320))]
        result = handler._build_embedding_params({"image": imgs})

        assert result is not None
        assert len(result["image_grid_thw"]) == 2
        assert result["image_grid_thw"][0] == [1, 30, 40]
        assert result["embeddings_shape"][1] == 2048

    def test_pil_image_qwen_params_unavailable_returns_none(self):
        """Qwen VL with no grid params -> returns None (fallback)."""
        from PIL import Image

        handler = _make_prefill_handler(model="Qwen/Qwen3-VL-2B-Instruct")
        handler._qwen_grid_params = None

        img = Image.new("RGB", (640, 480))
        result = handler._build_embedding_params({"image": img})
        assert result is None

    def test_pil_image_list_non_qwen_returns_none(self):
        """PIL image list for non-Qwen model -> returns None."""
        handler = _make_prefill_handler(model="llava-hf/llava-1.5-7b-hf")
        mm_data = {"image": [MagicMock()]}

        result = handler._build_embedding_params(mm_data)
        assert result is None

    def test_no_image_data_returns_none(self):
        """No image data -> returns None."""
        handler = _make_prefill_handler(model="Qwen/Qwen3-VL-2B-Instruct")
        mm_data = {}

        result = handler._build_embedding_params(mm_data)
        assert result is None
