# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request metadata forwarding tests for ``VllmLLMEngine.generate``."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from dynamo.common.constants import DisaggregationMode

# Guard the engine import directly: a partial vLLM install (e.g., metadata
# present but `vllm.usage` missing) lets `find_spec("vllm")` pass while the
# transitive import fails. Catch `ImportError` to also cover native-lib
# resolution errors (e.g., missing libcuda.so.1 on CPU-only test runners).
try:
    from dynamo.vllm.llm_engine import VllmLLMEngine
    from dynamo.vllm.multimodal_utils.request_processor import (
        VllmMultimodalRequestProcessor,
    )
except ImportError:
    pytest.skip("vllm backend not available", allow_module_level=True)

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
]


class _FakeContext:
    def __init__(self, trace_id: str | None = None, span_id: str | None = None):
        self._trace_id = trace_id
        self._span_id = span_id

    def id(self) -> str:
        return "trace-test-req"

    def trace_headers(self) -> dict[str, str] | None:
        if not self._trace_id or not self._span_id:
            return None
        return {"traceparent": f"00-{self._trace_id}-{self._span_id}-01"}


async def _empty_async_iter():
    """Async iterator that yields nothing. The unreachable ``yield`` is what
    makes this function an async generator rather than a coroutine."""
    for _ in ():
        yield


def _make_engine(engine_client) -> VllmLLMEngine:
    # Couples to VllmLLMEngine.__init__ attribute names — keep in sync.
    engine = VllmLLMEngine.__new__(VllmLLMEngine)
    engine.engine_client = engine_client
    engine._default_sampling_params = SimpleNamespace()
    engine._model_max_len = 1024
    engine.disaggregation_mode = DisaggregationMode.AGGREGATED
    engine.enable_rl = False
    engine._multimodal_request_processor = VllmMultimodalRequestProcessor(
        model="test-model",
        enable_multimodal=False,
    )
    # `_dp_range=None` bypasses the router's DP-rank resolution branch.
    engine._dp_range = None
    return engine


async def _drain(
    engine: VllmLLMEngine,
    ctx: _FakeContext,
    request: dict | None = None,
) -> None:
    async for _ in engine.generate(request or {"token_ids": [1, 2, 3]}, ctx):
        pass


@patch("dynamo.vllm.llm_engine.build_sampling_params")
async def test_forwards_trace_headers_when_context_has_trace(mock_build_sampling):
    mock_build_sampling.return_value = SimpleNamespace()
    captured: dict = {}

    def fake_generate(prompt, sampling_params, request_id, **kwargs):
        captured.update(kwargs)
        return _empty_async_iter()

    trace_id = "a" * 32
    span_id = "b" * 16

    await _drain(
        _make_engine(SimpleNamespace(generate=fake_generate)),
        _FakeContext(trace_id=trace_id, span_id=span_id),
    )

    assert captured["trace_headers"] == {"traceparent": f"00-{trace_id}-{span_id}-01"}


@patch("dynamo.vllm.llm_engine.build_sampling_params")
async def test_omits_trace_headers_when_no_trace_context(mock_build_sampling):
    mock_build_sampling.return_value = SimpleNamespace()
    captured: dict = {}

    def fake_generate(prompt, sampling_params, request_id, **kwargs):
        captured.update(kwargs)
        return _empty_async_iter()

    await _drain(
        _make_engine(SimpleNamespace(generate=fake_generate)),
        _FakeContext(),
    )

    # kwarg omitted (engine_trace_kwargs returns {}).
    assert "trace_headers" not in captured


@patch("dynamo.vllm.llm_engine.build_sampling_params")
async def test_forwards_agent_context_session_id_when_vllm_accepts_it(
    mock_build_sampling,
):
    mock_build_sampling.return_value = SimpleNamespace()
    captured: dict = {}

    def fake_generate(
        prompt,
        sampling_params,
        request_id,
        *,
        data_parallel_rank=None,
        lora_request=None,
        session_id=None,
    ):
        captured["session_id"] = session_id
        return _empty_async_iter()

    request = {
        "token_ids": [1, 2, 3],
        "agent_context": {"session_id": "session-1"},
    }

    await _drain(
        _make_engine(SimpleNamespace(generate=fake_generate)),
        _FakeContext(),
        request,
    )

    assert captured["session_id"] == "session-1"


@patch("dynamo.vllm.llm_engine.build_sampling_params")
async def test_drops_agent_context_session_id_for_old_vllm(mock_build_sampling):
    mock_build_sampling.return_value = SimpleNamespace()
    called = False

    def fake_generate(
        prompt,
        sampling_params,
        request_id,
        *,
        data_parallel_rank=None,
        lora_request=None,
    ):
        nonlocal called
        called = True
        return _empty_async_iter()

    request = {
        "token_ids": [1, 2, 3],
        "agent_context": {"session_id": "session-1"},
    }

    await _drain(
        _make_engine(SimpleNamespace(generate=fake_generate)),
        _FakeContext(),
        request,
    )

    assert called


@patch("dynamo.vllm.llm_engine.build_sampling_params")
async def test_forwards_reasoning_parser_metadata(mock_build_sampling):
    mock_build_sampling.return_value = SimpleNamespace()
    captured: dict = {}

    def fake_generate(
        prompt,
        sampling_params,
        request_id,
        *,
        data_parallel_rank=None,
        lora_request=None,
        reasoning_ended=None,
        reasoning_parser_kwargs=None,
    ):
        captured["reasoning_ended"] = reasoning_ended
        captured["reasoning_parser_kwargs"] = reasoning_parser_kwargs
        return _empty_async_iter()

    engine = _make_engine(SimpleNamespace(generate=fake_generate))
    request = {
        "token_ids": [1, 2, 3],
        "extra_args": {
            "reasoning_ended": False,
            "reasoning_parser_kwargs": {
                "chat_template_kwargs": {"reasoning_effort": "high"}
            },
        },
    }
    async for _ in engine.generate(request, _FakeContext()):
        pass

    assert captured == {
        "reasoning_ended": False,
        "reasoning_parser_kwargs": {
            "chat_template_kwargs": {"reasoning_effort": "high"}
        },
    }


@patch("dynamo.vllm.llm_engine.build_sampling_params")
async def test_drops_reasoning_parser_metadata_for_old_vllm(mock_build_sampling):
    mock_build_sampling.return_value = SimpleNamespace()
    called = False

    def fake_generate(
        prompt,
        sampling_params,
        request_id,
        *,
        data_parallel_rank=None,
        lora_request=None,
    ):
        nonlocal called
        called = True
        return _empty_async_iter()

    engine = _make_engine(SimpleNamespace(generate=fake_generate))
    request = {
        "token_ids": [1, 2, 3],
        "extra_args": {
            "reasoning_parser_kwargs": {
                "chat_template_kwargs": {"enable_thinking": True}
            }
        },
    }
    async for _ in engine.generate(request, _FakeContext()):
        pass

    assert called
