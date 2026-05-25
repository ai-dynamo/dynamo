# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the SGLang unified engine's disagg dispatch."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("sglang", reason="sglang not installed in this container")

from dynamo.common.constants import DisaggregationMode  # noqa: E402
from dynamo.sglang.llm_engine import SglangLLMEngine  # noqa: E402

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.unified,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _engine_with_cached_bootstrap() -> SglangLLMEngine:
    # __new__ bypasses real engine init; we only exercise pure-Python helpers.
    engine = SglangLLMEngine.__new__(SglangLLMEngine)
    engine._bootstrap_host = "10.0.0.5"
    engine._bootstrap_port = 12345
    engine.serving_mode = DisaggregationMode.PREFILL
    return engine


def test_prefill_bootstrap_honours_router_provided_triple():
    # Router-resolved triple must be passed through unchanged so the
    # decode peer (which the router writes the same triple onto) finds
    # the matching room.
    engine = _engine_with_cached_bootstrap()
    out = engine._resolve_prefill_bootstrap(
        {
            "token_ids": [1, 2, 3],
            "bootstrap_info": {
                "bootstrap_host": "router.host",
                "bootstrap_port": 9999,
                "bootstrap_room": 42,
            },
        }
    )
    assert out == {
        "bootstrap_host": "router.host",
        "bootstrap_port": 9999,
        "bootstrap_room": 42,
    }


def test_decode_bootstrap_raises_when_router_left_no_info():
    # Both router paths missing → loud failure. Silent zero-fill would
    # leave the decode worker waiting on a phantom KV transfer.
    with pytest.raises(ValueError, match="bootstrap"):
        SglangLLMEngine._resolve_decode_bootstrap({"token_ids": [1, 2, 3]})

@pytest.mark.asyncio
async def test_decode_forwards_decode_and_prefill_dp_ranks_separately():
    calls: dict[str, Any] = {}

    async def fake_stream():
        yield {
            "output_ids": [7],
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "prompt_tokens": 2,
                "completion_tokens": 1,
            },
        }

    class FakeSglangEngine:
        async def async_generate(
            self,
            input_ids=None,
            sampling_params=None,
            stream=False,
            rid=None,
            data_parallel_rank=None,
            disagg_prefill_dp_rank=None,
            bootstrap_host=None,
            bootstrap_port=None,
            bootstrap_room=None,
        ):
            calls.update(locals())
            return fake_stream()

    engine = SglangLLMEngine.__new__(SglangLLMEngine)
    engine.engine = FakeSglangEngine()
    engine.serving_mode = DisaggregationMode.DECODE
    engine.enable_trace = False
    engine._use_sglang_tokenizer = False
    engine._input_param_manager = SimpleNamespace(
        get_input_param=lambda request, use_tokenizer: request["token_ids"]
    )
    engine._dp_start = 0
    engine._dp_size = 8

    context = SimpleNamespace(trace_id="rid-1", is_stopped=lambda: False)
    request = {
        "token_ids": [1, 2],
        "routing": {"dp_rank": 5, "prefill_dp_rank": 2},
        "bootstrap_info": {
            "bootstrap_host": "prefill.host",
            "bootstrap_port": 9000,
            "bootstrap_room": 123,
        },
    }

    chunks = [chunk async for chunk in engine.generate(request, context)]

    assert chunks[-1]["finish_reason"] == "stop"
    assert calls["data_parallel_rank"] == 5
    assert calls["disagg_prefill_dp_rank"] == 2
    assert calls["bootstrap_room"] == 123
