# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from vllm.sampling_params import SamplingParams

pytest.importorskip(
    "vllm.v1.engine.async_llm",
    reason="a full vLLM installation is required by the decoder runtime",
)

from dynamo.vllm.decoder_runtime import VllmDecoderRuntime  # noqa: E402

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
]


def _runtime(engine: MagicMock) -> VllmDecoderRuntime:
    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=128),
        shutdown_timeout=7.0,
    )
    return VllmDecoderRuntime(
        engine=cast(Any, engine),
        vllm_config=cast(Any, config),
        default_sampling_params={"temperature": 0.4},
    )


def test_prepare_sampling_params_uses_prepared_prompt_length() -> None:
    runtime = _runtime(MagicMock())

    sampling_params = runtime.prepare_sampling_params(
        {
            "token_ids": [1],
            "sampling_options": {},
            "stop_conditions": {},
            "output_options": {},
        },
        prompt={"prompt_token_ids": list(range(100))},
    )

    assert sampling_params.max_tokens == 28


@pytest.mark.asyncio
async def test_decode_delegates_and_adapts_native_output() -> None:
    engine = MagicMock()
    sampling_params = SamplingParams(max_tokens=1)

    async def native_stream():
        yield SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    index=0,
                    token_ids=[7],
                    finish_reason="length",
                    stop_reason=None,
                    logprobs=None,
                    routed_experts=None,
                )
            ],
            prompt_token_ids=[1, 2],
            prompt_logprobs=None,
            num_cached_tokens=0,
        )

    engine.generate.return_value = native_stream()
    runtime = _runtime(engine)

    chunks = [
        chunk
        async for chunk in runtime.decode(
            {"prompt_token_ids": [1, 2]},
            sampling_params,
            "request-1",
            engine_options={"priority": 3},
        )
    ]

    assert chunks == [
        {
            "index": 0,
            "token_ids": [7],
            "finish_reason": "length",
            "completion_usage": {
                "prompt_tokens": 2,
                "completion_tokens": 1,
                "total_tokens": 3,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        }
    ]
    engine.generate.assert_called_once_with(
        {"prompt_token_ids": [1, 2]},
        sampling_params,
        "request-1",
        priority=3,
    )


@pytest.mark.asyncio
async def test_decode_stops_after_native_response_without_outputs() -> None:
    engine = MagicMock()

    async def native_stream():
        yield SimpleNamespace(outputs=[])
        yield SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    index=0,
                    token_ids=[7],
                    finish_reason="length",
                    stop_reason=None,
                    logprobs=None,
                    routed_experts=None,
                )
            ],
            prompt_token_ids=[1],
            prompt_logprobs=None,
            num_cached_tokens=0,
        )

    engine.generate.return_value = native_stream()
    runtime = _runtime(engine)

    chunks = [
        chunk
        async for chunk in runtime.decode(
            {"prompt_token_ids": [1]},
            SamplingParams(max_tokens=1),
            "request-1",
        )
    ]

    assert chunks == [
        {
            "finish_reason": "error: No outputs from vLLM engine",
            "index": 0,
            "token_ids": [],
        }
    ]


@pytest.mark.asyncio
async def test_abort_and_shutdown_are_lifecycle_safe() -> None:
    engine = MagicMock()
    engine.abort = AsyncMock()
    runtime = _runtime(engine)

    await runtime.abort("request-1")
    runtime.shutdown()
    runtime.shutdown()
    await runtime.abort("request-2")

    engine.abort.assert_awaited_once_with("request-1")
    engine.shutdown.assert_called_once_with(timeout=7.0)
    with pytest.raises(RuntimeError, match="shut down"):
        _ = runtime.engine
