# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from dynamo.common.constants import DisaggregationMode
from dynamo.vllm import llm_engine as mod
from dynamo.vllm.multimodal_utils.request_processor import (
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
    engine = _engine()
    prompt = {
        "prompt_token_ids": [1, 2, 3],
        "multi_modal_data": {"image": object(), "video": object()},
    }
    processor = SimpleNamespace(
        prepare_prompt=AsyncMock(return_value=PreparedMultimodalPrompt(prompt=prompt))
    )
    engine._multimodal_request_processor = processor
    monkeypatch.setattr(
        mod, "build_sampling_params", lambda *args, **kwargs: _sampling_params()
    )

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
                multi_modal_data={"image": image},
                mm_processor_kwargs={"max_pixels": 1003520},
            )
        ),
        build_prefill_handoff=MagicMock(return_value=handoff),
    )
    engine._multimodal_request_processor = processor
    monkeypatch.setattr(
        mod, "build_sampling_params", lambda *args, **kwargs: _sampling_params()
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
    config = SimpleNamespace(
        disaggregation_mode=mode,
        route_to_encoder=route_to_encoder,
    )
    monkeypatch.setattr(mod, "parse_args", lambda argv: config)

    with pytest.raises(NotImplementedError, match=message):
        await mod.VllmLLMEngine.from_args([])


@pytest.mark.asyncio
async def test_from_args_retains_multimodal_runtime_configuration(monkeypatch):
    config = SimpleNamespace(
        disaggregation_mode=DisaggregationMode.AGGREGATED,
        route_to_encoder=False,
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
    monkeypatch.setattr(mod, "parse_args", lambda argv: config)
    monkeypatch.setattr(
        mod.WorkerConfig,
        "from_runtime_config",
        lambda *args, **kwargs: worker_config,
    )

    engine, actual_worker_config = await mod.VllmLLMEngine.from_args([])

    assert actual_worker_config is worker_config
    assert engine.enable_multimodal is True
    assert engine.frontend_decoding is True
    assert engine.multimodal_embedding_cache_capacity_gb == 2.5
    assert engine._namespace == "deployment"
