# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Asserts ``TrtllmLLMEngine.generate`` forwards ``trace_headers`` to
``generate_async``."""

from __future__ import annotations

import importlib.util

import pytest

if importlib.util.find_spec("tensorrt_llm") is None:
    pytest.skip("tensorrt_llm not installed", allow_module_level=True)

from types import SimpleNamespace

from dynamo.trtllm.constants import DisaggregationMode
from dynamo.trtllm.llm_engine import TrtllmLLMEngine

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.unit,
    pytest.mark.trtllm,
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


def _make_engine(generate_async) -> TrtllmLLMEngine:
    # Couples to TrtllmLLMEngine.__init__ attribute names — keep in sync.
    engine = TrtllmLLMEngine.__new__(TrtllmLLMEngine)
    engine._engine = SimpleNamespace(llm=SimpleNamespace(generate_async=generate_async))
    engine._default_sampling_params = SimpleNamespace()
    engine.disaggregation_mode = DisaggregationMode.AGGREGATED
    engine.max_seq_len = 1024
    engine._active_requests = {}
    # The AGGREGATED branch writes max_tokens on the returned object;
    # SimpleNamespace allows that.
    engine._override_sampling_params = lambda default, request: SimpleNamespace(
        max_tokens=None
    )
    return engine


async def _drain(engine: TrtllmLLMEngine, ctx: _FakeContext) -> None:
    async for _ in engine.generate({"token_ids": [1, 2, 3]}, ctx):
        pass


async def test_forwards_trace_headers_when_context_has_trace():
    captured: dict = {}

    def fake_generate_async(**kwargs):
        captured.update(kwargs)
        return _empty_async_iter()

    trace_id = "a" * 32
    span_id = "b" * 16

    await _drain(
        _make_engine(fake_generate_async),
        _FakeContext(trace_id=trace_id, span_id=span_id),
    )

    assert captured["trace_headers"] == {
        "traceparent": f"00-{trace_id}-{span_id}-01"
    }


async def test_forwards_none_when_no_trace_context():
    captured: dict = {}

    def fake_generate_async(**kwargs):
        captured.update(kwargs)
        return _empty_async_iter()

    await _drain(_make_engine(fake_generate_async), _FakeContext())

    assert captured["trace_headers"] is None
