# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("vllm.v1.engine.async_llm")
pytest.importorskip("vllm.usage.usage_lib")

from dynamo.common.constants import DisaggregationMode  # noqa: E402
from dynamo.vllm import llm_engine as llm_engine_mod  # noqa: E402
from dynamo.vllm.constants import EmbeddingTransferMode  # noqa: E402
from dynamo.vllm.multimodal_utils import (  # noqa: E402
    prefill_worker_utils as prefill_worker_utils_mod,
)
from dynamo.vllm.multimodal_utils.prefill_worker_utils import (  # noqa: E402
    EncoderResultEmbeddingLoader,
)
from dynamo.vllm.multimodal_utils.request_processor import (  # noqa: E402
    PreparedMultimodalPrompt,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.unified,
    pytest.mark.multimodal,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class _Context:
    def id(self):
        return "request-1"

    def trace_headers(self):
        return None


class _EngineClient:
    tokenizer = None

    def __init__(self, responses=()):
        self.responses = responses
        self.calls = []

    def generate(self, *args, **kwargs):
        self.calls.append((args, kwargs))

        async def stream():
            for response in self.responses:
                yield response

        return stream()


def _engine(mode=DisaggregationMode.AGGREGATED, responses=()):
    engine = llm_engine_mod.VllmLLMEngine(
        SimpleNamespace(model="Qwen/Qwen3-VL-2B-Instruct"),
        mode,
        served_model_name="Qwen/Qwen3-VL-2B-Instruct",
        component="backend",
        enable_multimodal=True,
    )
    engine.engine_client = _EngineClient(responses)
    engine._default_sampling_params = {}
    engine._model_max_len = 4096
    engine._dp_range = None
    return engine


def _sampling_params():
    return SimpleNamespace(extra_args=None, max_tokens=16, min_tokens=0)


@pytest.mark.asyncio
async def test_generate_submits_prepared_multimodal_prompt(monkeypatch):
    engine = _engine()
    prompt = {
        "prompt_token_ids": [1, 2, 3],
        "multi_modal_data": {"image": object(), "video": object()},
    }
    effective_request = {"token_ids": [1, 99, 2]}
    processor = SimpleNamespace(
        prepare_prompt=AsyncMock(
            return_value=PreparedMultimodalPrompt(
                prompt=prompt,
                request=effective_request,
            )
        )
    )
    engine._multimodal_request_processor = processor
    build_sampling_params = MagicMock(return_value=_sampling_params())
    monkeypatch.setattr(llm_engine_mod, "build_sampling_params", build_sampling_params)

    chunks = [
        chunk
        async for chunk in engine.generate(
            {
                "token_ids": [1, 2, 3],
                "sampling_options": {},
                "stop_conditions": {},
                "output_options": {},
            },
            _Context(),
        )
    ]

    assert chunks == []
    assert engine.engine_client.calls[0][0][0] is prompt
    assert build_sampling_params.call_args.args[0] is effective_request
    processor.prepare_prompt.assert_awaited_once()


@pytest.mark.asyncio
async def test_prefill_terminal_adds_embedding_handoff(monkeypatch):
    output = SimpleNamespace(
        index=0,
        token_ids=[42],
        finish_reason="length",
        stop_reason=None,
        logprobs=None,
    )
    response = SimpleNamespace(
        outputs=[output],
        prompt_token_ids=[1, 99, 2],
        prompt_logprobs=None,
        kv_transfer_params={"remote_block_ids": [7]},
    )
    engine = _engine(DisaggregationMode.PREFILL, [response])
    image = object()
    handoff = {
        "image_grid_thw": [[1, 2, 2]],
        "embeddings_shape": [1, 16],
    }
    processor = SimpleNamespace(
        prepare_prompt=AsyncMock(
            return_value=PreparedMultimodalPrompt(
                prompt={"prompt_token_ids": [1, 2]},
                request={"token_ids": [1, 2]},
                multi_modal_data={"image": image},
                mm_processor_kwargs={"max_pixels": 1003520},
            )
        ),
        build_prefill_handoff=MagicMock(return_value=handoff),
    )
    engine._multimodal_request_processor = processor
    monkeypatch.setattr(
        llm_engine_mod,
        "build_sampling_params",
        lambda *args, **kwargs: _sampling_params(),
    )

    chunks = [
        chunk
        async for chunk in engine.generate(
            {
                "token_ids": [1, 2],
                "sampling_options": {},
                "stop_conditions": {},
                "output_options": {},
            },
            _Context(),
        )
    ]

    assert chunks[0]["disaggregated_params"] == {
        "kv_transfer_params": {"remote_block_ids": [7]},
        "embedding_params": {
            "image_grid_thw": [[1, 2, 2]],
            "embeddings_shape": [1, 16],
        },
    }
    processor.build_prefill_handoff.assert_called_once_with(
        multi_modal_data={"image": image},
        prompt_token_ids=[1, 99, 2],
        mm_processor_kwargs={"max_pixels": 1003520},
    )


@pytest.mark.asyncio
async def test_llm_engine_rejects_encode_role(monkeypatch):
    config = SimpleNamespace(
        disaggregation_mode=DisaggregationMode.ENCODE,
        route_to_encoder=False,
        headless=False,
    )
    monkeypatch.setattr(
        llm_engine_mod,
        "parse_args",
        lambda argv, fpm_trace_relay_supported=False: config,
    )

    with pytest.raises(ValueError, match="VllmEncodeEngine"):
        await llm_engine_mod.VllmLLMEngine.from_args([])


@pytest.mark.asyncio
async def test_from_args_retains_multimodal_runtime_configuration(monkeypatch):
    config = SimpleNamespace(
        disaggregation_mode=DisaggregationMode.AGGREGATED,
        route_to_encoder=False,
        headless=False,
        served_model_name="model",
        model="model",
        component="backend",
        namespace="deployment",
        enable_rl=False,
        enable_multimodal=True,
        frontend_decoding=True,
        multimodal_embedding_cache_capacity_gb=2.5,
        embedding_transfer_mode=llm_engine_mod.EmbeddingTransferMode.LOCAL,
        dyn_tool_call_parser=None,
        dyn_reasoning_parser=None,
        engine_args=SimpleNamespace(
            model="model", served_model_name=["model"], logprobs_mode="raw_logprobs"
        ),
    )
    worker_config = object()
    from_runtime_config = MagicMock(return_value=worker_config)
    monkeypatch.setattr(
        llm_engine_mod,
        "parse_args",
        lambda argv, fpm_trace_relay_supported=False: config,
    )
    monkeypatch.setattr(
        llm_engine_mod.WorkerConfig,
        "from_runtime_config",
        from_runtime_config,
    )

    engine, actual_worker_config = await llm_engine_mod.VllmLLMEngine.from_args([])

    assert actual_worker_config is worker_config
    assert engine.enable_multimodal is True
    assert engine.frontend_decoding is True
    assert engine.multimodal_embedding_cache_capacity_gb == 2.5
    assert engine._namespace == "deployment"
    assert engine.route_to_encoder is False
    assert engine.embedding_transfer_mode == llm_engine_mod.EmbeddingTransferMode.LOCAL
    worker_overrides = from_runtime_config.call_args.kwargs
    assert worker_overrides["media_decoder"] is not None
    assert worker_overrides["media_fetcher"] is not None


@pytest.mark.asyncio
async def test_from_args_accepts_separate_encoder_routing(monkeypatch):
    config = SimpleNamespace(
        disaggregation_mode=DisaggregationMode.PREFILL,
        route_to_encoder=True,
        headless=False,
        served_model_name="model",
        model="model",
        component="prefill",
        namespace="deployment",
        enable_rl=False,
        enable_multimodal=True,
        frontend_decoding=False,
        multimodal_embedding_cache_capacity_gb=0.0,
        embedding_transfer_mode=llm_engine_mod.EmbeddingTransferMode.NIXL_WRITE,
        dyn_tool_call_parser=None,
        dyn_reasoning_parser=None,
        engine_args=SimpleNamespace(
            model="model", served_model_name=["model"], logprobs_mode="raw_logprobs"
        ),
    )
    monkeypatch.setattr(
        llm_engine_mod,
        "parse_args",
        lambda argv, fpm_trace_relay_supported=False: config,
    )
    monkeypatch.setattr(
        llm_engine_mod.WorkerConfig,
        "from_runtime_config",
        MagicMock(return_value=object()),
    )

    engine, _ = await llm_engine_mod.VllmLLMEngine.from_args([])

    assert engine.route_to_encoder is True
    assert (
        engine.embedding_transfer_mode
        == llm_engine_mod.EmbeddingTransferMode.NIXL_WRITE
    )


@pytest.mark.asyncio
async def test_encoder_result_loader_validates_schema():
    loader = EncoderResultEmbeddingLoader(AsyncMock())
    with pytest.raises(ValueError, match="schema_version"):
        await loader.load_encoder_result(
            {"schema_version": 2, "multimodal_inputs": []},
            model="model",
            request_id="request",
        )


def test_encoder_result_loader_selects_matching_receiver(monkeypatch):
    receiver = object()
    constructor = MagicMock(return_value=receiver)
    monkeypatch.setattr(
        prefill_worker_utils_mod, "NixlWriteEmbeddingReceiver", constructor
    )

    loader = EncoderResultEmbeddingLoader.from_transfer_mode(
        EmbeddingTransferMode.NIXL_WRITE
    )

    assert loader._receiver is receiver


def _encoder_result(*serialized_requests):
    return {
        "schema_version": 1,
        "multimodal_inputs": [
            {
                "serialized_request": {
                    "embeddings_shape": [1, 1],
                    "embedding_dtype_str": "float32",
                    "serialized_request": serialized_request,
                }
            }
            for serialized_request in serialized_requests
        ],
    }


@pytest.mark.asyncio
async def test_encoder_result_loader_releases_partial_receive_successes():
    class PartiallyFailingReceiver:
        def __init__(self):
            self.received = asyncio.Event()
            self.released = []

        async def receive_embeddings(self, request):
            if request.serialized_request == "success":
                self.received.set()
                return 7, prefill_worker_utils_mod.torch.ones((1, 1))
            await self.received.wait()
            raise RuntimeError("receive failed")

        def release_tensor(self, tensor_id):
            self.released.append(tensor_id)

    receiver = PartiallyFailingReceiver()
    loader = EncoderResultEmbeddingLoader(receiver)

    with pytest.raises(RuntimeError, match="receive failed"):
        await loader.load_encoder_result(
            _encoder_result("success", "failure"),
            model="model",
            request_id="request",
        )

    assert receiver.released == [7]


@pytest.mark.asyncio
async def test_encoder_result_loader_releases_local_transfer(monkeypatch):
    class RecordingLocalReceiver(prefill_worker_utils_mod.LocalEmbeddingReceiver):
        def __init__(self):
            self.embedding = prefill_worker_utils_mod.torch.ones((1, 1))
            self.released = []

        async def receive_embeddings(self, request):
            return 11, self.embedding

        def release_tensor(self, tensor_id):
            self.released.append(tensor_id)

    receiver = RecordingLocalReceiver()
    ensure_owned = MagicMock()
    monkeypatch.setattr(prefill_worker_utils_mod, "_ensure_owned_tensors", ensure_owned)
    monkeypatch.setattr(
        prefill_worker_utils_mod,
        "_accumulate_embeddings",
        lambda output, model, dtype, embedding, image_grid_thw: output.update(
            image=embedding
        ),
    )

    result = await EncoderResultEmbeddingLoader(receiver).load_encoder_result(
        _encoder_result("local"),
        model="model",
        request_id="request",
    )

    assert result["image"] is receiver.embedding
    assert receiver.released == [11]
    ensure_owned.assert_not_called()
