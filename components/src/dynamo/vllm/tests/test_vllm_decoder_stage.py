# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest

pytest.importorskip(
    "dynamo._core.backend",
    reason="dynamo._core.backend not built — run maturin develop first",
)
pytest.importorskip(
    "vllm.engine.arg_utils",
    reason="a full vLLM installation is required by the decoder stage",
)

from vllm.sampling_params import RequestOutputKind  # noqa: E402

from dynamo.llm.exceptions import InvalidArgument  # noqa: E402
from dynamo.vllm.decoder_runtime import VllmDecoderRuntime  # noqa: E402
from dynamo.vllm.decoder_stage import VllmDecoderStage  # noqa: E402
from dynamo.workflow import StageContext  # noqa: E402

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
]


class _FakeEngine:
    tokenizer = None

    def __init__(self) -> None:
        self.sampling_params: Any = None
        self.request_ids: list[str] = []
        self.abort_ids: list[str] = []
        self.failure: Exception | None = None
        self.started: asyncio.Event | None = None
        self.release: asyncio.Event | None = None

    def generate(self, prompt, sampling_params, request_id, **_options):
        self.sampling_params = sampling_params
        self.request_ids.append(request_id)

        async def stream():
            if self.started is not None:
                self.started.set()
            if self.release is not None:
                await self.release.wait()
            if self.failure is not None:
                raise self.failure
            yield SimpleNamespace(
                prompt_token_ids=prompt["prompt_token_ids"],
                prompt_logprobs=None,
                num_cached_tokens=None,
                outputs=[
                    SimpleNamespace(
                        index=0,
                        token_ids=[4, 2],
                        finish_reason="stop",
                        stop_reason=None,
                        logprobs=None,
                        routed_experts=None,
                    )
                ],
            )

        return stream()

    async def abort(self, request_id: str) -> None:
        self.abort_ids.append(request_id)

    def shutdown(self, *, timeout: float) -> None:
        del timeout


def _runtime(
    *, max_model_len: int = 128, defaults: dict[str, Any] | None = None
) -> tuple[VllmDecoderRuntime, _FakeEngine]:
    engine = _FakeEngine()
    runtime = VllmDecoderRuntime(
        engine=cast(Any, engine),
        vllm_config=cast(
            Any,
            SimpleNamespace(
                model_config=SimpleNamespace(max_model_len=max_model_len),
                shutdown_timeout=1.0,
            ),
        ),
        default_sampling_params=defaults,
    )
    return runtime, engine


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


def _context(attempt_id: str = "request-1", stage_id: str = "decoder") -> StageContext:
    return StageContext(
        workflow_name="test-workflow",
        stage_id=stage_id,
        attempt_id=attempt_id,
        deadline=None,
        _cancelled=asyncio.Event(),
    )


async def test_stage_uses_shared_final_decode_and_names_native_request() -> None:
    runtime, engine = _runtime()
    stage = VllmDecoderStage(runtime)

    result = await stage.run(
        {
            "request": _request(),
            "prompt": {"prompt_token_ids": [1, 2, 3, 99]},
        },
        _context(),
    )

    assert engine.request_ids == ["request-1:decoder"]
    assert engine.sampling_params.temperature == 0.25
    assert engine.sampling_params.max_tokens == 8
    assert engine.sampling_params.detokenize is False
    assert engine.sampling_params.output_kind == RequestOutputKind.FINAL_ONLY
    assert result == {
        "chunk": {
            "token_ids": [4, 2],
            "index": 0,
            "finish_reason": "stop",
            "completion_usage": {
                "prompt_tokens": 4,
                "completion_tokens": 2,
                "total_tokens": 6,
                "prompt_tokens_details": None,
            },
        }
    }


async def test_default_output_limit_uses_prepared_prompt_length() -> None:
    runtime, engine = _runtime(max_model_len=10, defaults={"max_tokens": 20})

    await VllmDecoderStage(runtime).run(
        {
            "request": _request(max_tokens=None),
            "prompt": {"prompt_token_ids": [1, 2, 3, 4]},
        },
        _context(),
    )

    assert engine.sampling_params.max_tokens == 6


async def test_rejects_requested_output_beyond_prepared_prompt_budget() -> None:
    runtime, _ = _runtime(max_model_len=10)

    with pytest.raises(InvalidArgument, match="prepared prompt budget"):
        await VllmDecoderStage(runtime).run(
            {
                "request": _request(max_tokens=7),
                "prompt": {"prompt_token_ids": [1, 2, 3, 4]},
            },
            _context(),
        )


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
    runtime, _ = _runtime()

    with pytest.raises(InvalidArgument, match=message):
        await VllmDecoderStage(runtime).run(
            {"request": request, "prompt": {"prompt_token_ids": [1]}},
            _context(),
        )


async def test_failure_aborts_native_request() -> None:
    runtime, engine = _runtime()
    engine.failure = RuntimeError("decode failed")

    with pytest.raises(RuntimeError, match="decode failed"):
        await VllmDecoderStage(runtime).run(
            {
                "request": _request(),
                "prompt": {"prompt_token_ids": [1]},
            },
            _context(),
        )

    assert engine.abort_ids == ["request-1:decoder"]


async def test_external_abort_uses_active_stage_request_id() -> None:
    runtime, engine = _runtime()
    engine.started = asyncio.Event()
    engine.release = asyncio.Event()
    stage = VllmDecoderStage(runtime)
    task = asyncio.create_task(
        stage.run(
            {
                "request": _request(),
                "prompt": {"prompt_token_ids": [1]},
            },
            _context("attempt-7", "generator"),
        )
    )
    await engine.started.wait()

    await stage.abort_attempt("attempt-7")
    engine.release.set()
    await task

    assert engine.abort_ids == ["attempt-7:generator"]
