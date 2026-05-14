# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Asserts ``VllmLLMEngine.generate`` forwards ``trace_headers`` to
``engine_client.generate``."""

from __future__ import annotations

import importlib.util

import pytest

if importlib.util.find_spec("vllm") is None:
    pytest.skip("vllm not installed", allow_module_level=True)

from types import SimpleNamespace
from unittest.mock import patch

from dynamo.common.constants import DisaggregationMode
from dynamo.vllm.llm_engine import VllmLLMEngine

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
]


class _FakeContext:
    def __init__(self, trace_id: str | None = None, span_id: str | None = None):
        self.trace_id = trace_id
        self.span_id = span_id

    def id(self) -> str:
        return "trace-test-req"


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
    return engine


async def _drain(engine: VllmLLMEngine, ctx: _FakeContext) -> None:
    async for _ in engine.generate({"token_ids": [1, 2, 3]}, ctx):
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

    assert captured["trace_headers"] == {
        "traceparent": f"00-{trace_id}-{span_id}-01"
    }


@patch("dynamo.vllm.llm_engine.build_sampling_params")
async def test_forwards_none_when_no_trace_context(mock_build_sampling):
    mock_build_sampling.return_value = SimpleNamespace()
    captured: dict = {}

    def fake_generate(prompt, sampling_params, request_id, **kwargs):
        captured.update(kwargs)
        return _empty_async_iter()

    await _drain(
        _make_engine(SimpleNamespace(generate=fake_generate)),
        _FakeContext(),
    )

    assert captured["trace_headers"] is None
