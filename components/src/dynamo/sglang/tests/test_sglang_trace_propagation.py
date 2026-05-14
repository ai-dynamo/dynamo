# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Asserts ``SglangLLMEngine.generate`` forwards ``external_trace_header`` to
``async_generate``, gated on ``server_args.enable_trace``."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dynamo.common.constants import DisaggregationMode

# Guard the engine import directly: package metadata can be present without
# submodules, and native-lib loads (e.g., libcuda.so.1) raise `ImportError`
# rather than `ModuleNotFoundError`. Catch both via `ImportError`.
try:
    from dynamo.sglang.llm_engine import SglangLLMEngine
except ImportError:
    pytest.skip("sglang backend not available", allow_module_level=True)

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
]


class _FakeContext:
    def __init__(self, trace_id: str | None = None, span_id: str | None = None):
        self._trace_id = trace_id
        # SGLang's request uses context.trace_id directly as the SGLang RID,
        # so keep the attribute available alongside the trace_headers() method.
        self.trace_id = trace_id
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


def _make_engine(async_generate, enable_trace: bool) -> SglangLLMEngine:
    # Couples to SglangLLMEngine.__init__ attribute names — keep in sync.
    engine = SglangLLMEngine.__new__(SglangLLMEngine)
    engine.engine = SimpleNamespace(async_generate=async_generate)
    engine.serving_mode = DisaggregationMode.AGGREGATED
    engine.enable_trace = enable_trace
    # Single-rank defaults: validate_global_dp_rank(None, 0, 1, ...) -> None.
    engine._dp_start = 0
    engine._dp_size = 1
    # generate() calls these helpers before the call site; stub them out.
    engine._build_sampling_params = lambda request: SimpleNamespace()
    engine._get_input_param = lambda request: {}
    return engine


async def _drain(engine: SglangLLMEngine, ctx: _FakeContext) -> None:
    async for _ in engine.generate({"token_ids": [1, 2, 3]}, ctx):
        pass


async def test_forwards_external_trace_header_when_enabled_and_context_has_trace():
    captured: dict = {}

    async def fake_async_generate(**kwargs):
        captured.update(kwargs)
        return _empty_async_iter()

    trace_id = "a" * 32
    span_id = "b" * 16

    await _drain(
        _make_engine(fake_async_generate, enable_trace=True),
        _FakeContext(trace_id=trace_id, span_id=span_id),
    )

    assert captured["external_trace_header"] == {
        "traceparent": f"00-{trace_id}-{span_id}-01"
    }


async def test_gates_off_when_enable_trace_false():
    captured: dict = {}

    async def fake_async_generate(**kwargs):
        captured.update(kwargs)
        return _empty_async_iter()

    # Even with a trace context, --enable-trace=False must suppress forwarding.
    await _drain(
        _make_engine(fake_async_generate, enable_trace=False),
        _FakeContext(trace_id="a" * 32, span_id="b" * 16),
    )

    assert captured["external_trace_header"] is None
