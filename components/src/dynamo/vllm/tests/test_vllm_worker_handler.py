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
from dynamo.vllm.multimodal_utils.request_processor import (
    PreparedMultimodalInput,
    VllmMultimodalRequestProcessor,
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
    enable_multimodal: bool = True,
    multimodal_embedding_cache_capacity_gb: float = 0,
    disaggregation_mode: str | None = None,
) -> MagicMock:
    """Create a mock Config with the fields used by MultimodalPDWorkerHandler."""
    from dynamo.vllm.constants import DisaggregationMode, EmbeddingTransferMode

    config = MagicMock()
    config.model = model
    if disaggregation_mode is not None:
        config.disaggregation_mode = getattr(DisaggregationMode, disaggregation_mode)
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
    handler._multimodal_request_processor = VllmMultimodalRequestProcessor(
        model=config.model,
        enable_multimodal=config.enable_multimodal,
        use_unified_vision_chunk=False,
    )
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
        config = _make_config(model="test-model", disaggregation_mode="PREFILL")
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
    handler.model_max_len = 4096
    handler.default_sampling_params = {}
    handler.kv_event_publisher = None
    handler.otel_tracing_enabled = False
    handler.input_param_manager = MagicMock()
    handler.input_param_manager.get_extra_params.return_value = {}
    handler._multimodal_request_processor = VllmMultimodalRequestProcessor(
        model=model,
        enable_multimodal=True,
        use_unified_vision_chunk=False,
    )
    handler._deferred_aborts = {}
    # Real BaseWorkerHandler.__init__ (patched out above) sets this; the
    # aggregated branch in _generate_token_mode reads it, so mirror the default.
    handler._custom_encoder = None
    return handler


@pytest.mark.asyncio(loop_scope="function")
class TestDecodeWorkerMultimodalBranching:
    """Tests for the mode-aware multimodal branching in _generate_token_mode."""

    @pytest.mark.parametrize(
        "request_payload",
        [
            {"multi_modal_data": {"image_url": [{"Url": "http://img.png"}]}},
            {"extra_args": {"mm_kwargs_shm": {"modality": "image"}}},
        ],
    )
    async def test_text_mode_rejects_multimodal_input_when_disabled(
        self, request_payload
    ):
        handler = _make_decode_handler(disaggregation_mode="AGGREGATED")
        handler.use_vllm_tokenizer = True
        handler._multimodal_request_processor.enable_multimodal = False

        with pytest.raises(ValueError, match="--enable-multimodal"):
            async for _ in handler.generate(request_payload, MagicMock()):
                pass

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

        assert chunks == [
            {
                "finish_reason": (
                    "error: Decode worker received multimodal request without "
                    "prefill result"
                ),
                "index": 0,
                "token_ids": [],
            }
        ]

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
        assert chunks[0]["token_ids"] == []
        assert "embedding metadata" in chunks[0]["finish_reason"]

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
            return_value=(None, {"status": "error", "message": "test stop"})
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
        """Aggregated mode delegates media loading to the shared processor."""
        handler = _make_decode_handler(disaggregation_mode="AGGREGATED")
        handler._multimodal_request_processor.extract_multimodal_data = AsyncMock(
            return_value=None
        )

        # Return an error from _build_prompt_from_request so _generate_token_mode
        # yields it and returns early — no need to mock the engine.
        handler._build_prompt_from_request = MagicMock(
            return_value=(None, {"status": "error", "message": "test stop"})
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

        handler._multimodal_request_processor.extract_multimodal_data.assert_awaited_once()
        assert len(chunks) == 1
        assert chunks[0]["status"] == "error"

    @pytest.mark.parametrize(
        "mm_processor_kwargs",
        [None, {"use_audio_in_video": True}],
    )
    async def test_handler_prompt_builder_delegates_processor_kwargs(
        self, mm_processor_kwargs
    ):
        handler = _make_decode_handler(disaggregation_mode="AGGREGATED")

        prompt, error = handler._build_prompt_from_request(
            {"token_ids": [1, 2, 3]},
            "request-prompt",
            multi_modal_data=None,
            mm_processor_kwargs=mm_processor_kwargs,
        )

        assert error is None
        if mm_processor_kwargs is None:
            assert "mm_processor_kwargs" not in prompt
        else:
            assert prompt["mm_processor_kwargs"] is mm_processor_kwargs


@pytest.mark.asyncio
async def test_prefill_delegates_mode_policy_to_shared_processor():
    handler = mod.PrefillWorkerHandler.__new__(mod.PrefillWorkerHandler)
    prepared_request = {"token_ids": [9, 10]}
    multi_modal_data = {"image": object()}
    mm_processor_kwargs = {"max_pixels": 4096}
    processor = SimpleNamespace(
        prepare_input=AsyncMock(
            return_value=PreparedMultimodalInput(
                request=prepared_request,
                multi_modal_data=multi_modal_data,
                mm_processor_kwargs=mm_processor_kwargs,
            )
        )
    )
    handler._multimodal_request_processor = processor
    handler._build_prompt_from_request = MagicMock(
        return_value=(None, {"status": "error", "message": "stop"})
    )
    context = MagicMock()

    chunks = [
        chunk
        async for chunk in handler._generate_token_mode(
            {"token_ids": [1, 2]},
            context,
            "request-prefill",
        )
    ]

    assert chunks == [
        {"status": "error", "message": "stop", "disaggregated_params": None}
    ]
    processor.prepare_input.assert_awaited_once_with(
        {"token_ids": [1, 2]},
        "request-prefill",
        context,
        mod.DisaggregationMode.PREFILL,
    )
    handler._build_prompt_from_request.assert_called_once_with(
        prepared_request,
        "request-prefill",
        multi_modal_data,
        log_prefix="Prefill ",
        mm_processor_kwargs=mm_processor_kwargs,
    )


@pytest.mark.asyncio
async def test_prefill_returns_structured_error_when_multimodal_is_disabled():
    handler = mod.PrefillWorkerHandler.__new__(mod.PrefillWorkerHandler)
    processor = SimpleNamespace(
        validate_multimodal_request=MagicMock(
            side_effect=ValueError("use --enable-multimodal")
        )
    )
    handler._multimodal_request_processor = processor
    context = MagicMock()
    context.id.return_value = "request-prefill-disabled"

    chunks = [chunk async for chunk in handler.generate({}, context)]

    assert chunks == [
        {
            "status": "error",
            "message": "use --enable-multimodal",
            "disaggregated_params": None,
        }
    ]


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
        handler._build_prompt_from_request = MagicMock(return_value=(MagicMock(), None))

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
        # vLLM's pooler has already reduced to the requested ``dimensions``, so
        # the stub returns a 128-dim vector (not 3) -- otherwise the handler's
        # oversized-dimensions guard would (correctly) reject it.
        vec = [i * 0.01 for i in range(128)]

        async def fake_encode(prompt, pooling_params, request_id):
            captured["pooling_params"] = pooling_params
            output = MagicMock()
            output.outputs.data = torch.tensor(vec)
            output.prompt_token_ids = [1, 2, 3]
            yield output

        handler.engine_client.encode = fake_encode

        request = {"input": ["hello"], "model": "test-model", "dimensions": 128}
        responses = [r async for r in handler.generate(request, context)]

        pp = captured["pooling_params"]
        assert pp.task == "embed"
        assert pp.dimensions == 128
        # No post-hoc truncation: the handler returns exactly the vector vLLM
        # produced (the 128-float stub here), trusting the pooler to have
        # already applied the dimensionality reduction.
        expected_b64 = mod._encode_floats_to_base64(vec)
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
    async def test_raw_text_truncation_forwarded_to_vllm(self):
        """Raw text inputs forward ``truncate_prompt_tokens`` to vLLM's
        tokenizer path, including the ``-1`` sentinel vLLM accepts.
        """
        handler = self._make_embedding_handler()
        context = self._make_context()
        captured: list[dict] = []

        async def fake_encode(
            prompt, pooling_params, request_id, *, tokenization_kwargs=None
        ):
            captured.append(
                {"prompt": prompt, "tokenization_kwargs": tokenization_kwargs}
            )
            output = MagicMock()
            output.outputs.data = torch.tensor([0.1, 0.2, 0.3])
            output.prompt_token_ids = [1, 2, 3]
            yield output

        handler.engine_client.encode = fake_encode

        for truncate_prompt_tokens in (2048, -1):
            request = {
                "input": "hello",
                "model": "test-model",
                "truncate_prompt_tokens": truncate_prompt_tokens,
            }
            _ = [r async for r in handler.generate(request, context)]

        assert captured == [
            {
                "prompt": "hello",
                "tokenization_kwargs": {"truncate_prompt_tokens": 2048},
            },
            {
                "prompt": "hello",
                "tokenization_kwargs": {"truncate_prompt_tokens": -1},
            },
        ]

    @pytest.mark.parametrize(
        ("truncate_prompt_tokens", "error_type", "match"),
        [
            ("2048", TypeError, "Invalid 'truncate_prompt_tokens' type"),
            (True, TypeError, "Invalid 'truncate_prompt_tokens' type"),
            (-2, ValueError, "truncate_prompt_tokens must be >= -1"),
        ],
    )
    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_truncate_prompt_tokens_rejects_invalid_values(
        self, truncate_prompt_tokens, error_type, match
    ):
        """Invalid truncation values fail before the request reaches vLLM."""
        handler = self._make_embedding_handler()
        context = self._make_context()

        request = {
            "input": "hello",
            "model": "test-model",
            "truncate_prompt_tokens": truncate_prompt_tokens,
        }
        with pytest.raises(error_type, match=match):
            async for _ in handler.generate(request, context):
                pass

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_truncation_uses_default_encode_shape_when_not_tokenizing(self):
        """Pretokenized inputs stay on the default encode path because callers
        already control token-id truncation before reaching vLLM.
        """
        handler = self._make_embedding_handler()
        context = self._make_context()
        prompts = []

        async def fake_encode(prompt, pooling_params, request_id):
            prompts.append(prompt)
            output = MagicMock()
            output.outputs.data = torch.tensor([0.1, 0.2, 0.3])
            output.prompt_token_ids = [1, 2, 3]
            yield output

        handler.engine_client.encode = fake_encode

        request = {"input": "hello", "model": "test-model"}
        _ = [r async for r in handler.generate(request, context)]

        request = {
            "input": [1, 2, 3],
            "model": "test-model",
            "truncate_prompt_tokens": 2,
        }
        _ = [r async for r in handler.generate(request, context)]

        assert prompts[0] == "hello"
        assert prompts[1]["prompt_token_ids"] == [1, 2, 3]

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
    async def test_distributed_update_can_match_async_rl_semantics(self):
        handler = _make_handler()
        handler._pause_lock = asyncio.Lock()
        handler._paused = False
        handler.engine_client = MagicMock()
        handler.engine_client.collective_rpc = AsyncMock()
        handler.engine_client.reset_prefix_cache = AsyncMock()

        resp = await handler.update_weights_from_distributed(
            {
                "allow_unpaused": True,
                "reset_prefix_cache": False,
                "engine_rpc": "update_weights",
                "weight_version": 7,
                "update_info": {"names": ["weight"]},
            }
        )

        assert resp == {"status": "ok", "version": 7}
        handler.engine_client.collective_rpc.assert_awaited_once_with(
            "update_weights",
            kwargs={"update_info": {"names": ["weight"]}},
        )
        handler.engine_client.reset_prefix_cache.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_distributed_update_rejects_unpaused_cache_reset(self):
        handler = _make_handler()
        handler._pause_lock = asyncio.Lock()
        handler._paused = False
        handler.engine_client = MagicMock()
        handler.engine_client.collective_rpc = AsyncMock()
        handler.engine_client.reset_prefix_cache = AsyncMock()

        resp = await handler.update_weights_from_distributed(
            {"allow_unpaused": True, "engine_rpc": "update_weights"}
        )

        assert resp["status"] == "error"
        assert "cannot reset the prefix cache" in resp["message"]
        handler.engine_client.collective_rpc.assert_not_awaited()
        handler.engine_client.reset_prefix_cache.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_distributed_update_preserves_safe_defaults(self):
        handler = _make_handler()
        handler._pause_lock = asyncio.Lock()
        handler.engine_client = MagicMock()
        handler.engine_client.collective_rpc = AsyncMock()
        handler.engine_client.reset_prefix_cache = AsyncMock()

        handler._paused = False
        resp = await handler.update_weights_from_distributed({})
        assert resp["status"] == "error"
        handler.engine_client.collective_rpc.assert_not_awaited()

        handler._paused = True
        resp = await handler.update_weights_from_distributed(
            {"engine_rpc": "finish_weight_update"}
        )
        assert resp["status"] == "ok"
        handler.engine_client.collective_rpc.assert_awaited_once_with(
            "finish_weight_update", kwargs={}
        )
        handler.engine_client.reset_prefix_cache.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_init_weights_update_group_succeeds_within_timeout(self):
        handler = _make_handler()
        handler._pause_lock = asyncio.Lock()
        handler.engine_client = MagicMock()
        handler.engine_client.collective_rpc = AsyncMock()

        resp = await handler.init_weights_update_group(
            {
                "engine_rpc": "init_weight_transfer_engine",
                "init_info": {"master_address": "trainer", "master_port": 29500},
            }
        )

        assert resp == {
            "status": "ok",
            "message": "Weight update group initialized",
        }
        handler.engine_client.collective_rpc.assert_awaited_once_with(
            "init_weight_transfer_engine",
            kwargs={
                "init_info": {
                    "master_address": "trainer",
                    "master_port": 29500,
                }
            },
        )
        assert not handler._pause_lock.locked()

    def test_init_weights_update_group_timeout_defaults_to_30_seconds(
        self, monkeypatch
    ):
        monkeypatch.delenv("DYN_RL_INIT_WEIGHTS_TIMEOUT_S", raising=False)

        assert mod._rl_init_weights_timeout_s() == 30.0

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_init_weights_update_group_timeout_exits_worker(self, monkeypatch):
        handler = _make_handler()
        handler._pause_lock = asyncio.Lock()
        handler.runtime = MagicMock()
        handler.engine_client = MagicMock()

        rpc_cancelled = asyncio.Event()

        async def blocked_rpc(*_args, **_kwargs):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                rpc_cancelled.set()
                raise

        handler.engine_client.collective_rpc = AsyncMock(side_effect=blocked_rpc)
        monkeypatch.setenv("DYN_RL_INIT_WEIGHTS_TIMEOUT_S", "0.01")
        exit_mock = MagicMock(side_effect=SystemExit(1))
        monkeypatch.setattr(mod.os, "_exit", exit_mock)

        with pytest.raises(SystemExit) as exc_info:
            await handler.init_weights_update_group(
                {"engine_rpc": "init_weight_transfer_engine"}
            )

        assert exc_info.value.code == 1
        assert rpc_cancelled.is_set()
        handler.runtime.shutdown.assert_called_once_with()
        exit_mock.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_init_weights_update_group_returns_inner_timeout_error(
        self, monkeypatch
    ):
        handler = _make_handler()
        handler._pause_lock = asyncio.Lock()
        handler.runtime = MagicMock()
        handler.engine_client = MagicMock()
        handler.engine_client.collective_rpc = AsyncMock(
            side_effect=TimeoutError("transport timed out")
        )
        exit_mock = MagicMock(side_effect=SystemExit(1))
        monkeypatch.setattr(mod.os, "_exit", exit_mock)

        resp = await handler.init_weights_update_group(
            {"engine_rpc": "init_weight_transfer_engine"}
        )

        assert resp == {"status": "error", "message": "transport timed out"}
        handler.runtime.shutdown.assert_not_called()
        exit_mock.assert_not_called()
        assert not handler._pause_lock.locked()

    @pytest.mark.asyncio
    async def test_init_weights_update_group_engine_dead_exits_worker(
        self, monkeypatch
    ):
        from vllm.v1.engine.exceptions import EngineDeadError

        handler = _make_handler()
        handler._pause_lock = asyncio.Lock()
        handler.runtime = MagicMock()
        handler.engine_client = MagicMock()
        handler.engine_client.collective_rpc = AsyncMock(
            side_effect=EngineDeadError("engine dead")
        )
        exit_mock = MagicMock(side_effect=SystemExit(1))
        monkeypatch.setattr(mod.os, "_exit", exit_mock)

        with pytest.raises(SystemExit) as exc_info:
            await handler.init_weights_update_group(
                {"engine_rpc": "init_weight_transfer_engine"}
            )

        assert exc_info.value.code == 1
        handler.runtime.shutdown.assert_called_once_with()
        exit_mock.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_init_weights_update_group_returns_ordinary_errors(self):
        handler = _make_handler()
        handler._pause_lock = asyncio.Lock()
        handler.runtime = MagicMock()
        handler.engine_client = MagicMock()
        handler.engine_client.collective_rpc = AsyncMock(
            side_effect=RuntimeError("init failed")
        )

        resp = await handler.init_weights_update_group(
            {"engine_rpc": "init_weight_transfer_engine"}
        )

        assert resp == {"status": "error", "message": "init failed"}
        handler.runtime.shutdown.assert_not_called()
        assert not handler._pause_lock.locked()

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
