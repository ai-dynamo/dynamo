# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for WorkerHandler in combination with multimodal handling."""
# [gluo FIXME] This suite of tests is added for MultimodalPDWorkerHandler,
# which is now removed. Yet the concept of this tests is still valid that
# we need to have unit tests for the worker handlers.
# Need to revisit the tests and update them to test the worker handlers.

import asyncio
import base64
import json
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
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
    pytest.mark.unit,
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
    # BaseWorkerHandler.__init__ is bypassed above; the decode generate path
    # registers per-request deferred-abort guards here.
    handler._deferred_aborts = {}
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


@pytest.mark.asyncio
async def test_clear_kv_blocks_resets_vllm_external_cache():
    handler = _make_handler()
    handler.engine_client = SimpleNamespace(
        reset_prefix_cache=AsyncMock(return_value=True)
    )

    chunks = [chunk async for chunk in handler.clear_kv_blocks()]

    assert chunks == [{"status": "success", "message": "KV cache cleared"}]
    handler.engine_client.reset_prefix_cache.assert_awaited_once_with(
        reset_connector=True
    )


@pytest.mark.asyncio
async def test_clear_kv_blocks_reports_reset_failure():
    handler = _make_handler()
    handler.engine_client = SimpleNamespace(
        reset_prefix_cache=AsyncMock(return_value=False)
    )

    chunks = [chunk async for chunk in handler.clear_kv_blocks()]

    assert chunks == [{"status": "error", "message": "KV cache reset failed"}]
    handler.engine_client.reset_prefix_cache.assert_awaited_once_with(
        reset_connector=True
    )


class TestReasoningParserForwarding:
    def test_request_reasoning_metadata_reads_extra_args(self):
        request = {
            "extra_args": {
                "reasoning_ended": False,
                "reasoning_parser_kwargs": {
                    "chat_template_kwargs": {"reasoning_effort": "high"}
                },
            }
        }

        assert mod._request_reasoning_metadata(request) == (
            False,
            {"chat_template_kwargs": {"reasoning_effort": "high"}},
        )

    def test_generate_signature_support_is_cached(self, monkeypatch):
        class EngineClient:
            def generate(
                self,
                prompt,
                sampling_params,
                request_id,
                *,
                reasoning_ended=None,
                reasoning_parser_kwargs=None,
            ):
                pass

        engine_client = EngineClient()
        signature_calls = 0
        original_signature = mod.inspect.signature

        def counting_signature(obj):
            nonlocal signature_calls
            signature_calls += 1
            return original_signature(obj)

        monkeypatch.setattr(mod.inspect, "signature", counting_signature)

        assert mod._engine_generate_reasoning_kwargs(
            engine_client,
            False,
            {"chat_template_kwargs": {"reasoning_effort": "high"}},
        ) == {
            "reasoning_ended": False,
            "reasoning_parser_kwargs": {
                "chat_template_kwargs": {"reasoning_effort": "high"}
            },
        }
        assert mod._engine_generate_reasoning_kwargs(
            engine_client,
            True,
            {"chat_template_kwargs": {"reasoning_effort": "low"}},
        ) == {
            "reasoning_ended": True,
            "reasoning_parser_kwargs": {
                "chat_template_kwargs": {"reasoning_effort": "low"}
            },
        }
        assert signature_calls == 1

    @pytest.mark.asyncio
    async def test_generate_tokens_forwards_reasoning_parser_metadata(self):
        from vllm.sampling_params import SamplingParams

        handler = _make_handler()
        calls = {}

        async def fake_generate(
            prompt,
            sampling_params,
            request_id,
            *,
            lora_request=None,
            data_parallel_rank=None,
            trace_headers=None,
            priority=0,
            reasoning_ended=None,
            reasoning_parser_kwargs=None,
        ):
            calls["reasoning_ended"] = reasoning_ended
            calls["reasoning_parser_kwargs"] = reasoning_parser_kwargs
            if False:
                yield None

        handler.engine_client = MagicMock()
        handler.engine_client.generate = fake_generate

        chunks = []
        async for chunk in handler.generate_tokens(
            PatchedTokensPrompt(prompt_token_ids=[1]),
            SamplingParams(max_tokens=1),
            "req-1",
            reasoning_ended=False,
            reasoning_parser_kwargs={
                "chat_template_kwargs": {"reasoning_effort": "high"}
            },
        ):
            chunks.append(chunk)

        assert chunks == []
        assert calls == {
            "reasoning_ended": False,
            "reasoning_parser_kwargs": {
                "chat_template_kwargs": {"reasoning_effort": "high"}
            },
        }

    @pytest.mark.asyncio
    async def test_generate_tokens_drops_reasoning_metadata_for_old_vllm(self):
        from vllm.sampling_params import SamplingParams

        handler = _make_handler()
        calls = {}

        async def fake_generate(
            prompt,
            sampling_params,
            request_id,
            *,
            lora_request=None,
            data_parallel_rank=None,
            trace_headers=None,
            priority=0,
        ):
            calls["called"] = True
            if False:
                yield None

        handler.engine_client = MagicMock()
        handler.engine_client.generate = fake_generate

        chunks = []
        async for chunk in handler.generate_tokens(
            PatchedTokensPrompt(prompt_token_ids=[1]),
            SamplingParams(max_tokens=1),
            "req-1",
            reasoning_ended=True,
            reasoning_parser_kwargs={
                "chat_template_kwargs": {"reasoning_effort": "low"}
            },
        ):
            chunks.append(chunk)

        assert chunks == []
        assert calls == {"called": True}

    @pytest.mark.asyncio
    async def test_generate_tokens_emits_routed_experts_on_final_chunk(self):
        from vllm.sampling_params import SamplingParams

        handler = _make_handler()
        handler._extract_logprobs = MagicMock(return_value=(None, None))

        routed_experts = np.array([[[1]], [[2]]], dtype=np.int16)

        async def fake_generate(*args, **kwargs):
            yield SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        index=0,
                        token_ids=[11],
                        routed_experts=None,
                        finish_reason=None,
                        stop_reason=None,
                    )
                ],
                prompt_token_ids=[1, 2],
                prompt_logprobs=None,
            )
            # DELTA output_kind: each chunk carries only its own new token(s),
            # and generate_tokens passes output.token_ids through verbatim — so
            # the second chunk is the delta [12], not the cumulative [11, 12].
            yield SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        index=0,
                        token_ids=[12],
                        routed_experts=routed_experts,
                        finish_reason="stop",
                        stop_reason=None,
                    )
                ],
                prompt_token_ids=[1, 2],
                prompt_logprobs=None,
            )

        handler.engine_client = MagicMock()
        handler.engine_client.generate = fake_generate

        chunks = []
        async for chunk in handler.generate_tokens(
            PatchedTokensPrompt(prompt_token_ids=[1]),
            SamplingParams(max_tokens=2),
            "req-1",
        ):
            chunks.append(chunk)

        assert [chunk["token_ids"] for chunk in chunks] == [[11], [12]]
        # routed_experts must ride engine_data (where the Rust postprocessor's
        # build_response_nvext reads it from), not disaggregated_params. It is
        # only emitted on the final chunk.
        assert "engine_data" not in chunks[0]
        assert "disaggregated_params" not in chunks[1]
        routed = chunks[1]["engine_data"]["routed_experts"]
        assert routed["shape"] == [2, 1, 1]
        assert routed["dtype"] == "int16"
        # start defaults to 0 when routed_experts_prompt_start is unset.
        assert routed["start"] == 0
        decoded = np.frombuffer(
            base64.b64decode(routed["data"]), dtype=np.dtype(routed["dtype"])
        )
        np.testing.assert_array_equal(decoded, routed_experts.reshape(-1))

    @pytest.mark.asyncio
    async def test_generate_tokens_routed_experts_start_echoes_prompt_start(self):
        """routed_experts.start echoes SamplingParams.routed_experts_prompt_start
        (the offset vLLM trimmed) so the RL consumer can align the completion."""
        from vllm.sampling_params import SamplingParams

        # routed_experts_prompt_start is an RL-patch field; set it post-construct
        # so the test runs against stock vLLM too (the worker reads it via
        # getattr, defaulting to 0 when absent).
        sampling_params = SamplingParams(max_tokens=1)
        try:
            sampling_params.routed_experts_prompt_start = 5
        except (AttributeError, TypeError):
            pytest.skip("installed vLLM has no routed_experts_prompt_start support")

        handler = _make_handler()
        handler._extract_logprobs = MagicMock(return_value=(None, None))
        routed_experts = np.array([[[3]], [[4]]], dtype=np.int16)

        # Prompt longer than the requested start, so the echoed offset is not
        # clamped to the prompt length.
        prompt_token_ids = [1, 2, 3, 4, 5, 6]

        async def fake_generate(*args, **kwargs):
            yield SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        index=0,
                        token_ids=[12],
                        routed_experts=routed_experts,
                        finish_reason="stop",
                        stop_reason=None,
                    )
                ],
                prompt_token_ids=prompt_token_ids,
                prompt_logprobs=None,
            )

        handler.engine_client = MagicMock()
        handler.engine_client.generate = fake_generate

        chunks = []
        async for chunk in handler.generate_tokens(
            PatchedTokensPrompt(prompt_token_ids=prompt_token_ids),
            sampling_params,
            "req-2",
        ):
            chunks.append(chunk)

        assert chunks[-1]["engine_data"]["routed_experts"]["start"] == 5

    @pytest.mark.asyncio
    async def test_generate_tokens_routed_experts_start_clamped_to_prompt_len(self):
        """An out-of-range routed_experts_prompt_start is clamped to the prompt
        length (vLLM clamps the returned rows the same way)."""
        from vllm.sampling_params import SamplingParams

        sampling_params = SamplingParams(max_tokens=1)
        try:
            sampling_params.routed_experts_prompt_start = 99
        except (AttributeError, TypeError):
            pytest.skip("installed vLLM has no routed_experts_prompt_start support")

        handler = _make_handler()
        handler._extract_logprobs = MagicMock(return_value=(None, None))
        routed_experts = np.array([[[3]], [[4]]], dtype=np.int16)

        async def fake_generate(*args, **kwargs):
            yield SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        index=0,
                        token_ids=[12],
                        routed_experts=routed_experts,
                        finish_reason="stop",
                        stop_reason=None,
                    )
                ],
                prompt_token_ids=[1, 2, 3],
                prompt_logprobs=None,
            )

        handler.engine_client = MagicMock()
        handler.engine_client.generate = fake_generate

        chunks = []
        async for chunk in handler.generate_tokens(
            PatchedTokensPrompt(prompt_token_ids=[1, 2, 3]),
            sampling_params,
            "req-3",
        ):
            chunks.append(chunk)

        # start=99 clamped to prompt_len=3
        assert chunks[-1]["engine_data"]["routed_experts"]["start"] == 3


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
    handler._deferred_aborts = {}
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
        result = handler._build_embedding_params(mm_data, [1, 2, 3])

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
        result = handler._build_embedding_params({"image": img}, [1, 2, 3])

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
        result = handler._build_embedding_params({"image": imgs}, [1, 2, 3])

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
        result = handler._build_embedding_params({"image": img}, [1, 2, 3])
        assert result is None

    def test_pil_image_list_llava_returns_expanded_prompt_token_ids(self):
        """PIL image list for LLaVA model -> returns expanded prompt token ids."""
        handler = _make_prefill_handler(model="llava-hf/llava-1.5-7b-hf")
        mm_data = {"image": [MagicMock()]}

        result = handler._build_embedding_params(mm_data, [1, 2, 3])
        assert result["expanded_prompt_token_ids"] == [1, 2, 3]

    def test_no_image_data_returns_none(self):
        """No image data -> returns None."""
        handler = _make_prefill_handler(model="Qwen/Qwen3-VL-2B-Instruct")
        mm_data = {}

        result = handler._build_embedding_params(mm_data, [1, 2, 3])
        assert result is None


# ── Deferred abort (disagg decode KV-transfer safety) tests ────────


class TestDeferredAbort:
    """Tests for ``_DeferredAbort`` used in disaggregated decode mode.

    Purpose: when a request is cancelled before the decode worker has
    received the first token, the underlying NIXL KV transfer may still be
    in flight. Calling ``engine_client.abort(request_id)`` at that moment
    can crash EngineCore. ``_DeferredAbort`` delays the real abort call
    until the first engine output has been signalled.
    """

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_abort_before_first_token_does_not_fire_immediately(self):
        """abort() before first token should NOT call engine_client.abort yet.

        abort() awaits the deferred task to completion so the engine abort
        cannot be dropped under concurrent cancellation, so spawn it as a
        task to observe the mid-flight state, then use close() to cancel the
        parked waiter and let abort() return.
        """
        engine_client = MagicMock()
        engine_client.abort = AsyncMock()
        guard = mod._DeferredAbort(engine_client, "req-1")

        # abort() before first token returns promptly; the real abort is
        # deferred to a background task and engine.abort is NOT called yet.
        await asyncio.wait_for(guard.abort(), timeout=1.0)
        engine_client.abort.assert_not_called()
        assert guard._abort_task is not None
        assert not guard._abort_task.done()

        # Cleanup: close() cancels the parked deferred waiter without firing abort.
        await guard.close()
        engine_client.abort.assert_not_called()

    @pytest.mark.asyncio
    async def test_abort_after_first_token_fires_immediately(self):
        """abort() after signal_first_token should call engine_client.abort."""
        engine_client = MagicMock()
        engine_client.abort = AsyncMock()
        guard = mod._DeferredAbort(engine_client, "req-2")

        guard.signal_first_token()
        await guard.abort()

        engine_client.abort.assert_awaited_once_with("req-2")

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_deferred_background_task_fires_after_first_token(self):
        """Background task should call abort once first token is signalled."""
        engine_client = MagicMock()
        engine_client.abort = AsyncMock()
        guard = mod._DeferredAbort(engine_client, "req-3")

        # abort() returns promptly (deferred); engine.abort not called yet.
        await asyncio.wait_for(guard.abort(), timeout=1.0)
        engine_client.abort.assert_not_called()
        assert guard._abort_task is not None
        assert not guard._abort_task.done()

        # Signalling first token wakes the deferred waiter, which runs abort().
        guard.signal_first_token()
        await guard._abort_task

        engine_client.abort.assert_awaited_once_with("req-3")

    @pytest.mark.asyncio
    async def test_signal_first_token_is_idempotent(self):
        """Calling signal_first_token multiple times is safe."""
        engine_client = MagicMock()
        engine_client.abort = AsyncMock()
        guard = mod._DeferredAbort(engine_client, "req-4")

        guard.signal_first_token()
        guard.signal_first_token()
        await guard.abort()

        engine_client.abort.assert_awaited_once_with("req-4")

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_monitor_abort_routes_through_guard(self):
        """_monitor_abort should call guard.abort() instead of engine_client.abort()."""
        handler = _make_handler()
        handler.engine_client = MagicMock()
        handler.engine_client.abort = AsyncMock()
        handler.shutdown_event = None

        killed_future = asyncio.get_event_loop().create_future()
        killed_future.set_result(None)
        context = MagicMock()
        context.async_killed_or_stopped.return_value = killed_future

        guard = mod._DeferredAbort(handler.engine_client, "req-5")
        # _monitor_abort awaits guard.abort() to completion. With first_token
        # not yet received, that await blocks on the deferred waiter; spawn it
        # as a task to observe state and then signal first_token to unblock.
        monitor_task = asyncio.create_task(
            handler._monitor_abort(
                context, "req-5", is_prefill=False, abort_guard=guard
            )
        )
        await asyncio.sleep(0)
        handler.engine_client.abort.assert_not_called()
        assert not monitor_task.done()

        guard.signal_first_token()
        await monitor_task

        handler.engine_client.abort.assert_awaited_once_with("req-5")

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_monitor_abort_direct_when_no_guard(self):
        """Without a guard, _monitor_abort should call engine_client.abort directly."""
        handler = _make_handler()
        handler.engine_client = MagicMock()
        handler.engine_client.abort = AsyncMock()
        handler.shutdown_event = None

        killed_future = asyncio.get_event_loop().create_future()
        killed_future.set_result(None)
        context = MagicMock()
        context.async_killed_or_stopped.return_value = killed_future

        await handler._monitor_abort(context, "req-6", is_prefill=False)

        handler.engine_client.abort.assert_awaited_once_with("req-6")

    # close() cleanup tests: case 1b safety

    @pytest.mark.asyncio
    async def test_close_without_pending_abort_is_noop(self):
        """close() with no deferred abort must not call engine_client.abort."""
        engine_client = MagicMock()
        engine_client.abort = AsyncMock()
        guard = mod._DeferredAbort(engine_client, "req-close-1")

        await guard.close()

        engine_client.abort.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_close_cancels_waiter_without_abort_when_no_first_token(self):
        """Case 1b: close() must cancel the waiter without firing abort.

        The abort guard exists to avoid calling engine_client.abort before the
        first engine output arrives (unsafe during NIXL KV transfer). The
        cleanup path must preserve that invariant.
        """
        engine_client = MagicMock()
        engine_client.abort = AsyncMock()
        guard = mod._DeferredAbort(engine_client, "req-close-1b")

        # abort() awaits the deferred waiter, so spawn as a task to observe
        # the parked state.
        abort_task = asyncio.create_task(guard.abort())
        await asyncio.sleep(0)
        engine_client.abort.assert_not_called()
        assert guard._abort_task is not None
        assert not guard._abort_task.done()

        await guard.close()
        await abort_task

        engine_client.abort.assert_not_called()
        assert guard._abort_task.done()
        assert guard._abort_task.cancelled()

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_close_awaits_deferred_abort_when_first_token_received(self):
        """close() after first token must let the now-safe deferred abort finish."""
        engine_client = MagicMock()
        engine_client.abort = AsyncMock()
        guard = mod._DeferredAbort(engine_client, "req-close-first-tok")

        abort_task = asyncio.create_task(guard.abort())
        await asyncio.sleep(0)
        engine_client.abort.assert_not_called()

        # Signal first token; the deferred waiter wakes and runs engine.abort,
        # which unblocks abort_task. close() then observes the completed task.
        guard.signal_first_token()
        await abort_task
        await guard.close()

        engine_client.abort.assert_awaited_once_with("req-close-first-tok")
        assert guard._abort_task is not None
        assert guard._abort_task.done()
        assert not guard._abort_task.cancelled()

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_close_observes_already_completed_deferred_abort(self):
        """close() is safe when the background waiter already ran to completion."""
        engine_client = MagicMock()
        engine_client.abort = AsyncMock()
        guard = mod._DeferredAbort(engine_client, "req-close-done")

        await asyncio.wait_for(guard.abort(), timeout=1.0)
        guard.signal_first_token()
        await guard._abort_task

        assert guard._abort_task is not None
        assert guard._abort_task.done()
        engine_client.abort.assert_awaited_once_with("req-close-done")

        # close() must not re-issue abort and must not raise.
        await guard.close()
        engine_client.abort.assert_awaited_once_with("req-close-done")

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_generate_token_mode_closes_guard_on_no_output(self):
        """_generate_token_mode awaits guard cleanup when decode yields nothing.

        Verifies that in decode-only mode, when generate_tokens exits without
        yielding any output, _generate_token_mode still awaits the deferred
        abort guard's close() method, and does not call engine_client.abort
        in the pre-first-token window.
        """
        config = _make_config(disaggregation_mode="DECODE")
        handler = _make_handler(config=config)
        handler.engine_client = MagicMock()
        handler.engine_client.abort = AsyncMock()
        handler.shutdown_event = None
        handler.runtime = MagicMock()
        handler.config = config
        handler.default_sampling_params = {}
        handler.model_max_len = None
        handler.input_param_manager = MagicMock()
        handler.input_param_manager.get_input_param.return_value = [1, 2, 3]
        handler._resolve_lora_request = MagicMock(return_value=None)
        handler._get_mm_processor_kwargs = MagicMock(return_value={})
        handler._build_prompt_from_request = MagicMock(
            return_value=(MagicMock(), None, None)
        )

        # Capture the guard created inside the handler and wrap close() so
        # the test can assert that the handler awaited it.
        created_guards: list[mod._DeferredAbort] = []
        real_deferred_abort = mod._DeferredAbort

        def _capture(engine_client, request_id, on_engine_dead=None):
            g = real_deferred_abort(engine_client, request_id, on_engine_dead)
            g.close = AsyncMock(wraps=g.close)
            created_guards.append(g)
            return g

        killed_future = asyncio.get_event_loop().create_future()
        killed_future.set_result(None)
        context = MagicMock()
        context.async_killed_or_stopped.return_value = killed_future

        async def _empty_gen(*args, **kwargs):
            # Decode yields nothing: the case-1b shape.
            await asyncio.sleep(0)
            if False:
                yield None
            return

        handler.generate_tokens = _empty_gen

        request = {
            "token_ids": [1, 2, 3],
            "sampling_options": {},
            "stop_conditions": {},
            "output_options": {},
            "prefill_result": None,
            "routing": {},
            "model": "test-model",
        }

        with patch.object(mod, "_DeferredAbort", side_effect=_capture):
            async for _ in handler._generate_token_mode(
                request, context, "req-decode-1b"
            ):
                pass

        assert len(created_guards) == 1
        guard = created_guards[0]

        for _ in range(5):
            await asyncio.sleep(0)

        # _generate_token_mode must have awaited guard cleanup, and must not
        # have called engine_client.abort in the pre-first-token window.
        guard.close.assert_awaited_once()
        handler.engine_client.abort.assert_not_called()
        if guard._abort_task is not None:
            assert guard._abort_task.done()


class TestClassifyEmbeddingInput:
    """Unit tests for the embedding input classifier.

    Covers the four OpenAI-spec input shapes (str / list[str] / list[int] /
    list[list[int]]) and the previous bug where `[1, 2, 3]` was silently
    coerced to three text prompts via `str(item)`. Pure-function logic, no
    async / vLLM engine needed.
    """

    def test_single_string(self):
        assert mod._classify_embedding_input("hello") == ["hello"]

    def test_list_of_strings(self):
        result = mod._classify_embedding_input(["a", "b", "c"])
        assert result == ["a", "b", "c"]

    def test_list_of_ints_is_one_tokenized_prompt(self):
        # The bug: previously this returned ["1", "2", "3"] (three text
        # prompts). Correct behavior is one tokenized prompt.
        result = mod._classify_embedding_input([1, 2, 3])
        assert result == [[1, 2, 3]]

    def test_list_of_list_of_ints_is_batch_of_tokenized_prompts(self):
        result = mod._classify_embedding_input([[1, 2], [3, 4, 5]])
        assert result == [[1, 2], [3, 4, 5]]

    def test_mixed_str_and_int_rejected(self):
        with pytest.raises(TypeError, match="mixes str and non-str"):
            mod._classify_embedding_input(["hello", 42])

    def test_mixed_int_and_str_rejected(self):
        with pytest.raises(TypeError, match="mixes int and non-int"):
            mod._classify_embedding_input([1, "two"])

    def test_mixed_list_of_lists_with_str_rejected(self):
        with pytest.raises(TypeError, match="must be a list of"):
            mod._classify_embedding_input([[1, 2], "three"])

    def test_inner_list_with_non_int_rejected(self):
        with pytest.raises(TypeError, match="must be a list of"):
            mod._classify_embedding_input([[1, 2], [3.5, 4]])

    def test_bool_is_not_treated_as_int(self):
        # `bool` is a subclass of `int`; token ids must be real ints.
        with pytest.raises(TypeError):
            mod._classify_embedding_input([True, False])

    def test_empty_list_rejected(self):
        with pytest.raises(ValueError, match="must be non-empty"):
            mod._classify_embedding_input([])

    def test_unsupported_top_level_type_rejected(self):
        with pytest.raises(TypeError, match="Invalid 'input' type"):
            mod._classify_embedding_input({"text": "hi"})

    def test_unsupported_element_type_rejected(self):
        with pytest.raises(TypeError, match="Unsupported 'input' element"):
            mod._classify_embedding_input([3.14, 2.71])


class TestEmbeddingWorkerHandlerCancellation:
    """Tests for partial-failure cleanup in ``EmbeddingWorkerHandler.generate``.

    Each prompt's encode runs as its own asyncio task in ``asyncio.gather``.
    If one task raises, gather re-raises but does NOT cancel the others by
    default -- they keep consuming vLLM engine capacity for output that the
    handler would discard. The handler's ``try/finally`` block must cancel
    every still-pending task and await them with ``return_exceptions=True``
    before propagating the failure to the frontend.
    """

    def _make_embedding_handler(self) -> "mod.EmbeddingWorkerHandler":
        """Construct an ``EmbeddingWorkerHandler`` with the engine monitor
        stubbed so it does not spawn a real background task during the test.
        """
        with patch.object(mod, "VllmEngineMonitor"):
            handler = mod.EmbeddingWorkerHandler(
                runtime=MagicMock(),
                engine=MagicMock(),
                config=MagicMock(served_model_name="test-model"),
                shutdown_event=None,
            )
        # Replace the engine client wholesale: each test installs its own
        # ``encode`` async generator behaviour. ``abort`` may be called by
        # the per-prompt ``_monitor_abort`` task on cancellation paths.
        handler.engine_client = MagicMock()
        handler.engine_client.abort = AsyncMock()
        return handler

    def _make_context(self) -> MagicMock:
        """Mock dynamo.Context whose ``async_killed_or_stopped()`` never
        resolves (so the per-prompt ``_monitor_abort`` task parks until
        cancelled by the wrapping ``_abort_monitor`` context manager).
        """
        context = MagicMock()
        context.id.return_value = "test-req"
        context.async_killed_or_stopped.return_value = (
            asyncio.get_event_loop().create_future()
        )
        return context

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_partial_failure_cancels_in_flight_encodes(self):
        """When one prompt's encode raises, the siblings must be cancelled.

        Mocks ``engine_client.encode`` as a per-prompt async generator. For
        the failing index it raises immediately; for the others it parks on
        ``asyncio.sleep`` and records cancellation when it occurs. Without
        the ``finally``-cancel pass the test would hang on the asleep
        siblings until the 5s pytest timeout fires.
        """
        handler = self._make_embedding_handler()
        context = self._make_context()
        cancelled: set[int] = set()
        started: dict[int, asyncio.Event] = {i: asyncio.Event() for i in range(4)}

        async def fake_encode(prompt, pooling_params, request_id):
            idx = int(request_id.rsplit("-", 1)[-1])
            started[idx].set()
            if idx == 1:
                raise RuntimeError("boom")
            try:
                # Would hang past the 5s pytest timeout if not cancelled.
                await asyncio.sleep(60)
                yield MagicMock()  # unreachable
            except asyncio.CancelledError:
                cancelled.add(idx)
                raise

        handler.engine_client.encode = fake_encode

        request = {"input": ["a", "b", "c", "d"], "model": "test-model"}
        with pytest.raises(RuntimeError, match="boom"):
            async for _ in handler.generate(request, context):
                pass

        # Sibling encodes (0, 2, 3) must have started (so we know they were
        # in flight when idx=1 failed) AND been observed as cancelled.
        assert started[0].is_set()
        assert started[2].is_set()
        assert started[3].is_set()
        assert cancelled == {
            0,
            2,
            3,
        }, f"sibling encodes were not cancelled: cancelled={cancelled}"

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_happy_path_yields_response_without_cancelling(self):
        """When every prompt succeeds, the ``finally`` block sees no pending
        tasks and exits cleanly; the handler yields the OpenAI-shaped
        response. Guards against the cleanup pass accidentally aborting
        completed tasks.
        """
        handler = self._make_embedding_handler()
        context = self._make_context()
        aborted: list[str] = []
        handler.engine_client.abort = AsyncMock(side_effect=aborted.append)

        async def fake_encode(prompt, pooling_params, request_id):
            output = MagicMock()
            output.outputs.data = torch.tensor([0.1, 0.2, 0.3])
            output.prompt_token_ids = [1, 2, 3]
            yield output

        handler.engine_client.encode = fake_encode

        request = {"input": ["a", "b"], "model": "test-model"}
        responses = [r async for r in handler.generate(request, context)]

        assert len(responses) == 1
        response = responses[0]
        assert response["object"] == "list"
        assert response["model"] == "test-model"
        assert len(response["data"]) == 2
        assert response["data"][0]["index"] == 0
        assert response["data"][1]["index"] == 1
        # The worker always emits base64 on the internal worker->frontend
        # wire format; the Rust HTTP frontend decodes back to float at the
        # HTTP boundary when the client asks for float. So both data items
        # have the same base64 of [0.1, 0.2, 0.3] here.
        expected_b64 = mod._encode_floats_to_base64([0.1, 0.2, 0.3])
        assert response["data"][0]["embedding"] == expected_b64
        assert response["data"][1]["embedding"] == expected_b64
        # No tasks were in flight at gather completion, so the finally
        # cancel-and-await pass must not have touched the engine.
        assert aborted == []

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_dimensions_forwarded_to_pooling_params(self):
        """``dimensions`` is forwarded to vLLM via ``PoolingParams`` rather
        than applied as post-hoc truncation in the handler.

        vLLM's pooler then does the Matryoshka reduction (truncate +
        re-normalize) and validates that the model supports it. The handler
        must NOT slice the returned vector itself, so it emits exactly what
        the engine produced.
        """
        handler = self._make_embedding_handler()
        context = self._make_context()
        captured: dict = {}

        async def fake_encode(prompt, pooling_params, request_id):
            captured["pooling_params"] = pooling_params
            output = MagicMock()
            output.outputs.data = torch.tensor([0.1, 0.2, 0.3])
            output.prompt_token_ids = [1, 2, 3]
            yield output

        handler.engine_client.encode = fake_encode

        request = {"input": ["hello"], "model": "test-model", "dimensions": 128}
        responses = [r async for r in handler.generate(request, context)]

        pp = captured["pooling_params"]
        assert pp.task == "embed"
        assert pp.dimensions == 128
        # No post-hoc truncation: the handler returns exactly the vector vLLM
        # produced (the 3-float stub here), trusting the pooler to have
        # already applied the dimensionality reduction.
        expected_b64 = mod._encode_floats_to_base64([0.1, 0.2, 0.3])
        assert responses[0]["data"][0]["embedding"] == expected_b64

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_no_dimensions_omits_pooling_dimensions(self):
        """Without ``dimensions`` the handler requests ``task="embed"`` only,
        leaving ``PoolingParams.dimensions`` unset so vLLM returns the model's
        native embedding size.
        """
        handler = self._make_embedding_handler()
        context = self._make_context()
        captured: dict = {}

        async def fake_encode(prompt, pooling_params, request_id):
            captured["pooling_params"] = pooling_params
            output = MagicMock()
            output.outputs.data = torch.tensor([0.1, 0.2, 0.3])
            output.prompt_token_ids = [1, 2, 3]
            yield output

        handler.engine_client.encode = fake_encode

        request = {"input": ["hello"], "model": "test-model"}
        _ = [r async for r in handler.generate(request, context)]

        pp = captured["pooling_params"]
        assert pp.task == "embed"
        assert pp.dimensions is None

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_oversized_dimensions_raises(self):
        """When vLLM silently clamps an oversized ``dimensions`` request (a
        model enabled via ``--hf-overrides '{"is_matryoshka": true}'`` with no
        ``matryoshka_dimensions`` list), the handler raises a clear error
        instead of returning a shorter-than-requested vector.
        """
        handler = self._make_embedding_handler()
        context = self._make_context()

        async def fake_encode(prompt, pooling_params, request_id):
            output = MagicMock()
            # vLLM clamped to the model's native size (3 dims here) even though
            # 2048 was requested.
            output.outputs.data = torch.tensor([0.1, 0.2, 0.3])
            output.prompt_token_ids = [1, 2, 3]
            yield output

        handler.engine_client.encode = fake_encode

        request = {"input": ["hello"], "model": "test-model", "dimensions": 2048}
        with pytest.raises(ValueError, match="exceeds model embedding dimension"):
            async for _ in handler.generate(request, context):
                pass


class TestPadMmHashesTo64:
    """The frontend forwards canonical 16-char hex mm_hashes; vLLM must pad
    them to the 64-char BlockStored form the router's
    parse_mm_hash_from_extra_key keys on."""

    def test_pads_16_char_to_64(self):
        out = mod._pad_mm_hashes_to_64(["0123456789abcdef"])
        assert out == ["0123456789abcdef" + "0" * 48]
        assert len(out[0]) == 64

    def test_already_64_char_unchanged(self):
        h64 = "0123456789abcdef" + "0" * 48
        assert mod._pad_mm_hashes_to_64([h64]) == [h64]

    def test_mixed_and_empty(self):
        h64 = "f" * 64
        assert mod._pad_mm_hashes_to_64([]) == []
        assert mod._pad_mm_hashes_to_64(["abc", h64]) == ["abc" + "0" * 61, h64]


class TestRLAdminRouteHardening:
    """Regressions for the codex round-2 RL admin fixes."""

    @pytest.mark.asyncio
    async def test_admin_rejects_non_dict_body(self):
        handler = _make_handler()
        handler.engine_client = MagicMock()
        for body in ([], "x", 5, ["pause"]):
            for fn in (
                handler.pause_generation,
                handler.resume_generation,
                handler.flush_cache,
                handler.abort_request,
            ):
                resp = await fn(body)
                assert resp["status"] == "error", (fn.__name__, body, resp)
                assert "JSON object" in resp["message"]

    @pytest.mark.asyncio
    async def test_abort_request_surfaces_deferred_abort_failure(self):
        handler = _make_handler()
        handler.engine_client = MagicMock()

        class _FailingGuard:
            def __init__(self):
                self._abort_exc = RuntimeError("engine abort boom")

            async def abort(self):
                return None

        handler._deferred_aborts = {"req-x": _FailingGuard()}
        resp = await handler.abort_request({"request_id": "req-x"})
        assert resp["status"] == "error"
        assert "boom" in resp["message"]

    @pytest.mark.asyncio
    async def test_abort_request_ok_when_deferred_clean(self):
        handler = _make_handler()
        handler.engine_client = MagicMock()

        class _CleanGuard:
            def __init__(self):
                self._abort_exc = None

            async def abort(self):
                return None

        handler._deferred_aborts = {"req-y": _CleanGuard()}
        resp = await handler.abort_request({"request_id": "req-y"})
        assert resp["status"] == "ok"
        assert resp["request_id"] == "req-y"

    @pytest.mark.asyncio
    async def test_deferred_abort_does_not_block_before_first_token(self):
        # abort() before the first token must return promptly (the real abort is
        # deferred to a background task), not hang on the first-token event.
        guard = mod._DeferredAbort(MagicMock(), "req-z")
        await asyncio.wait_for(guard.abort(), timeout=1.0)
        assert guard._abort_exc is None
        await guard.close()

    @pytest.mark.asyncio
    async def test_deferred_abort_escalates_engine_dead(self):
        from vllm.v1.engine.exceptions import EngineDeadError

        escalated = []

        async def boom(_request_id):
            raise EngineDeadError("engine dead")

        engine = MagicMock()
        engine.abort = boom
        guard = mod._DeferredAbort(
            engine, "req-d", on_engine_dead=lambda e: escalated.append(e)
        )
        guard.signal_first_token()  # post-first-token -> immediate abort path
        await guard.abort()
        assert len(escalated) == 1
        assert isinstance(escalated[0], EngineDeadError)
