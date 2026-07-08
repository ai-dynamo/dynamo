# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from dynamo.common.constants import DisaggregationMode
from dynamo.vllm.multimodal_utils.request_processor import PreparedMultimodalPrompt

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
    from dynamo.vllm import llm_engine as mod

    engine = mod.VllmLLMEngine(
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
    from dynamo.vllm import llm_engine as mod

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
    monkeypatch.setattr(mod, "build_sampling_params", build_sampling_params)

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
@pytest.mark.parametrize(
    ("mode", "route_to_encoder", "message"),
    [
        (DisaggregationMode.ENCODE, False, "ENCODE is not supported"),
        (DisaggregationMode.AGGREGATED, True, "--route-to-encoder is not supported"),
    ],
)
async def test_from_args_rejects_unsupported_multimodal_topologies(
    monkeypatch, mode, route_to_encoder, message
):
    from dynamo.vllm import llm_engine as mod

    config = SimpleNamespace(
        disaggregation_mode=mode,
        route_to_encoder=route_to_encoder,
        headless=False,
    )
    monkeypatch.setattr(
        mod,
        "parse_args",
        lambda argv, fpm_trace_relay_supported=False: config,
    )

    with pytest.raises(NotImplementedError, match=message):
        await mod.VllmLLMEngine.from_args([])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", [DisaggregationMode.PREFILL, DisaggregationMode.DECODE]
)
async def test_from_args_rejects_multimodal_pd_until_followup(monkeypatch, mode):
    from dynamo.vllm import llm_engine as mod

    config = SimpleNamespace(
        disaggregation_mode=mode,
        route_to_encoder=False,
        headless=False,
        enable_multimodal=True,
    )
    monkeypatch.setattr(
        mod,
        "parse_args",
        lambda argv, fpm_trace_relay_supported=False: config,
    )

    with pytest.raises(NotImplementedError, match="multimodal P/D"):
        await mod.VllmLLMEngine.from_args([])


@pytest.mark.asyncio
async def test_from_args_retains_multimodal_runtime_configuration(monkeypatch):
    from dynamo.vllm import llm_engine as mod

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
        dyn_tool_call_parser=None,
        dyn_reasoning_parser=None,
        engine_args=SimpleNamespace(
            model="model", served_model_name=["model"], logprobs_mode="raw_logprobs"
        ),
    )
    worker_config = object()
    from_runtime_config = MagicMock(return_value=worker_config)
    monkeypatch.setattr(
        mod,
        "parse_args",
        lambda argv, fpm_trace_relay_supported=False: config,
    )
    monkeypatch.setattr(
        mod.WorkerConfig,
        "from_runtime_config",
        from_runtime_config,
    )

    engine, actual_worker_config = await mod.VllmLLMEngine.from_args([])

    assert actual_worker_config is worker_config
    assert engine.enable_multimodal is True
    assert engine.frontend_decoding is True
    assert engine.multimodal_embedding_cache_capacity_gb == 2.5
    assert engine._namespace == "deployment"
    worker_overrides = from_runtime_config.call_args.kwargs
    assert worker_overrides["media_decoder"] is not None
    assert worker_overrides["media_fetcher"] is not None
    assert from_runtime_config.call_args.kwargs["model_input"] is mod.ModelInput.Tokens
