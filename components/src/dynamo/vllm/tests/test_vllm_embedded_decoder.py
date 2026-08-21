# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip(
    "dynamo._core.backend",
    reason="dynamo._core.backend not built — run maturin develop first",
)
pytest.importorskip(
    "vllm.engine.arg_utils",
    reason="a full vLLM installation is required by the embedded decoder",
)

from vllm.sampling_params import RequestOutputKind  # noqa: E402

from dynamo.llm.exceptions import InvalidArgument  # noqa: E402
from dynamo.vllm.embedded_decoder import EmbeddedVllmDecoder  # noqa: E402

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
]


class _FakeNativeEngine:
    def __init__(self) -> None:
        self.sampling_params: Any = None
        self.abort_ids: list[str] = []
        self.shutdown_calls = 0
        self.failure: Exception | None = None

    async def generate(
        self,
        prompt: dict[str, Any],
        sampling_params: Any,
        request_id: str,
    ) -> AsyncIterator[Any]:
        self.sampling_params = sampling_params
        if self.failure is not None:
            raise self.failure
        yield SimpleNamespace(
            prompt_token_ids=prompt["prompt_token_ids"],
            outputs=[SimpleNamespace(index=0, token_ids=[4, 2], finish_reason="stop")],
        )

    async def abort(self, request_id: str) -> None:
        self.abort_ids.append(request_id)

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def _vllm_config(max_model_len: int = 128) -> MagicMock:
    config = MagicMock()
    config.model_config.max_model_len = max_model_len
    config.cache_config.block_size = 16
    config.scheduler_config.max_num_seqs = 32
    config.scheduler_config.max_num_batched_tokens = 4096
    return config


def _decoder(
    engine: _FakeNativeEngine,
    *,
    max_model_len: int = 128,
    default_sampling_params: dict[str, Any] | None = None,
) -> EmbeddedVllmDecoder:
    return EmbeddedVllmDecoder(
        engine=engine,
        engine_args=MagicMock(),
        vllm_config=_vllm_config(max_model_len),
        default_sampling_params=default_sampling_params or {},
    )


def _request(*, max_tokens: int | None = 8, n: int | None = 1) -> dict[str, Any]:
    stop_conditions: dict[str, Any] = {}
    if max_tokens is not None:
        stop_conditions["max_tokens"] = max_tokens
    return {
        "token_ids": [1, 2, 3],
        "sampling_options": {"n": n, "temperature": 0.25},
        "stop_conditions": stop_conditions,
        "output_options": {},
    }


async def test_generate_final_translates_and_normalizes() -> None:
    engine = _FakeNativeEngine()
    decoder = _decoder(engine)

    chunk = await decoder.generate_final(
        _request(),
        {"prompt_token_ids": [1, 2, 3, 99]},
        "request-1",
    )

    assert engine.sampling_params.temperature == 0.25
    assert engine.sampling_params.max_tokens == 8
    assert engine.sampling_params.detokenize is False
    assert engine.sampling_params.output_kind == RequestOutputKind.FINAL_ONLY
    assert chunk == {
        "token_ids": [4, 2],
        "index": 0,
        "finish_reason": "stop",
        "completion_usage": {
            "prompt_tokens": 4,
            "completion_tokens": 2,
            "total_tokens": 6,
        },
    }


async def test_default_output_limit_uses_prepared_prompt_length() -> None:
    engine = _FakeNativeEngine()
    decoder = _decoder(
        engine,
        max_model_len=10,
        default_sampling_params={"max_tokens": 20},
    )

    await decoder.generate_final(
        _request(max_tokens=None),
        {"prompt_token_ids": [1, 2, 3, 4]},
        "request-1",
    )

    assert engine.sampling_params.max_tokens == 6


@pytest.mark.parametrize(
    ("request_update", "message"),
    [
        ({"sampling_options": {"n": 2}}, "exactly one choice"),
        ({"output_options": {"logprobs": 1}}, "does not support logprobs"),
    ],
)
async def test_rejects_unsupported_final_output_options(
    request_update: dict[str, Any],
    message: str,
) -> None:
    request = _request()
    request.update(request_update)

    with pytest.raises(InvalidArgument, match=message):
        await _decoder(_FakeNativeEngine()).generate_final(
            request,
            {"prompt_token_ids": [1]},
            "request-1",
        )


async def test_failure_aborts_native_request() -> None:
    engine = _FakeNativeEngine()
    engine.failure = RuntimeError("decode failed")

    with pytest.raises(RuntimeError, match="decode failed"):
        await _decoder(engine).generate_final(
            _request(),
            {"prompt_token_ids": [1]},
            "request-1",
        )

    assert engine.abort_ids == ["request-1"]


def test_factory_owns_engine_setup_and_registration() -> None:
    engine_args = MagicMock()
    engine_args.enable_log_requests = True
    engine_args.disable_log_stats = False
    config = _vllm_config()
    config.model_config.get_diff_sampling_param.return_value = {"temperature": 0.5}
    engine_args.create_engine_config.return_value = config
    native_engine = _FakeNativeEngine()

    with patch(
        "dynamo.vllm.embedded_decoder.AsyncLLM.from_vllm_config",
        return_value=native_engine,
    ):
        decoder = EmbeddedVllmDecoder.from_engine_args(engine_args)

    registration = decoder.registration()
    assert registration.context_length == 128
    assert registration.kv_cache_block_size == 16
    assert registration.max_num_seqs == 32
    assert registration.max_num_batched_tokens == 4096

    decoder.shutdown()
    decoder.shutdown()
    assert native_engine.shutdown_calls == 1
