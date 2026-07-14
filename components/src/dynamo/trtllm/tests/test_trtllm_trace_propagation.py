# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Asserts ``TrtllmLLMEngine.generate`` forwards ``trace_headers`` to
``generate_async``."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from dynamo.trtllm.constants import DisaggregationMode

# Guard the engine import directly: package metadata can be present without
# submodules, and native-lib loads (e.g., libcuda.so.1) raise `ImportError`
# rather than `ModuleNotFoundError`. Catch both via `ImportError`.
try:
    from dynamo.trtllm.llm_engine import TrtllmLLMEngine
except ImportError:
    pytest.skip("tensorrt_llm backend not available", allow_module_level=True)

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.core,
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


def _make_engine(generate_async) -> TrtllmLLMEngine:
    # Couples to TrtllmLLMEngine.__init__ attribute names — keep in sync.
    engine = TrtllmLLMEngine.__new__(TrtllmLLMEngine)
    engine._engine = SimpleNamespace(llm=SimpleNamespace(generate_async=generate_async))
    engine._logits_processor_spec = None
    engine._default_sampling_params = SimpleNamespace()
    engine.disaggregation_mode = DisaggregationMode.AGGREGATED
    engine.max_seq_len = 1024
    engine._active_requests = {}
    engine._additional_metrics = None
    engine._inflight_lock = asyncio.Lock()
    engine._inflight_requests = 0
    engine._no_inflight_requests = asyncio.Event()
    engine._no_inflight_requests.set()
    engine._reject_new_requests = False
    # Match __init__ default; generate_locally reads this before dispatch.
    engine._conversation_affinity = False
    # Single-rank default: validate_global_dp_rank(None, 0, 1, ...) -> None,
    # so scheduling_params stays None and never touches a real SchedulingParams.
    engine._attention_dp_size = 1
    # The AGGREGATED branch writes max_tokens on the returned object;
    # SimpleNamespace allows that.
    engine._override_sampling_params = lambda default, request: SimpleNamespace(
        max_tokens=None
    )
    return engine


async def _drain(
    engine: TrtllmLLMEngine, ctx: _FakeContext, request: dict | None = None
) -> None:
    async for _ in engine.generate(request or {"token_ids": [1, 2, 3]}, ctx):
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

    assert captured["trace_headers"] == {"traceparent": f"00-{trace_id}-{span_id}-01"}


async def test_omits_trace_headers_when_no_trace_context():
    captured: dict = {}

    def fake_generate_async(**kwargs):
        captured.update(kwargs)
        return _empty_async_iter()

    await _drain(_make_engine(fake_generate_async), _FakeContext())

    # kwarg omitted (engine_trace_kwargs returns {}).
    assert "trace_headers" not in captured


async def test_forwards_routing_cache_salt():
    captured: dict = {}

    def fake_generate_async(**kwargs):
        captured.update(kwargs)
        return _empty_async_iter()

    await _drain(
        _make_engine(fake_generate_async),
        _FakeContext(),
        {"token_ids": [1, 2, 3], "routing": {"cache_salt": "tenant-a"}},
    )

    assert captured["cache_salt"] == "tenant-a"


async def test_visible_stop_tokens_use_dynamo_stopping():
    captured: dict = {}

    def fake_generate_async(**kwargs):
        captured.update(kwargs)
        return _empty_async_iter()

    sampling_params = SimpleNamespace(
        max_tokens=None,
        ignore_eos=False,
        min_tokens=None,
        stop_token_ids=[200, 300],
    )
    engine = _make_engine(fake_generate_async)
    engine._override_sampling_params = lambda _default, _request: sampling_params

    await _drain(
        engine,
        _FakeContext(),
        {
            "token_ids": [1, 2, 3],
            "stop_conditions": {
                "max_tokens": 4,
                "stop_token_ids_hidden": [100, 200],
                "stop_token_ids_visible": [200],
            },
        },
    )

    assert captured["sampling_params"] is sampling_params
    assert sampling_params.ignore_eos is True
    assert set(sampling_params.stop_token_ids) == {100, 300}
