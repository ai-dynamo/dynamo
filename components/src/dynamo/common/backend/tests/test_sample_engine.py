# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the sample engine's disaggregation dispatch."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip(
    "dynamo._core.backend",
    reason="dynamo._core.backend not built — run `maturin develop` first",
)

from dynamo.common.backend.sample_engine import SampleLLMEngine  # noqa: E402
from dynamo.common.constants import DisaggregationMode  # noqa: E402

pytestmark = [
    pytest.mark.unit,
    pytest.mark.unified,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _ctx(stopped: bool = False) -> MagicMock:
    """Minimal Context stand-in. The sample engine only consults
    ``is_stopped`` on each iteration — anything else is unused."""
    ctx = MagicMock()
    ctx.is_stopped.return_value = stopped
    return ctx


async def _collect(engine: SampleLLMEngine, request: dict) -> list[dict]:
    return [chunk async for chunk in engine.generate(request, _ctx())]


async def test_aggregated_mode_emits_max_tokens_chunks():
    # Aggregated mode produces one chunk per token, with the terminal
    # marked length and carrying completion_usage. No disaggregated_params.
    engine = SampleLLMEngine(
        max_tokens=4,
        delay=0.0,
        disaggregation_mode=DisaggregationMode.AGGREGATED,
    )
    chunks = await _collect(engine, {"token_ids": [1, 2]})
    assert len(chunks) == 4
    assert chunks[-1]["finish_reason"] == "length"
    assert "disaggregated_params" not in chunks[-1]


async def test_prefill_mode_caps_to_one_token_with_disaggregated_params():
    engine = SampleLLMEngine(
        max_tokens=16,
        delay=0.0,
        disaggregation_mode=DisaggregationMode.PREFILL,
    )
    # Even when the client asks for 8, prefill caps to 1.
    chunks = await _collect(
        engine,
        {"token_ids": [1, 2, 3], "stop_conditions": {"max_tokens": 8}},
    )
    assert len(chunks) == 1
    terminal = chunks[0]
    assert terminal["finish_reason"] == "length"
    assert terminal["completion_usage"]["completion_tokens"] == 1
    # The synthetic handle on the prefill terminal is what the frontend's
    # PrefillRouter forwards to the decode peer.
    assert "disaggregated_params" in terminal
    assert "sample_handle" in terminal["disaggregated_params"]


async def test_decode_mode_rejects_request_without_prefill_result():
    engine = SampleLLMEngine(
        max_tokens=4,
        delay=0.0,
        disaggregation_mode=DisaggregationMode.DECODE,
    )
    # Decode workers can't proceed without the prefill peer's payload.
    with pytest.raises(ValueError, match="prefill_result"):
        async for _ in engine.generate({"token_ids": [1, 2, 3]}, _ctx()):
            pass


async def test_decode_mode_runs_to_completion_when_prefill_result_provided():
    engine = SampleLLMEngine(
        max_tokens=4,
        delay=0.0,
        disaggregation_mode=DisaggregationMode.DECODE,
    )
    chunks = await _collect(
        engine,
        {
            "token_ids": [1, 2, 3],
            "prefill_result": {"disaggregated_params": {"sample_handle": "from-test"}},
        },
    )
    assert len(chunks) == 4
    terminal = chunks[-1]
    assert terminal["finish_reason"] == "length"
    # Decode does not stamp a fresh disaggregated_params on its response —
    # that's the prefill role.
    assert "disaggregated_params" not in terminal


async def test_from_args_propagates_mode_to_worker_config():
    engine, worker_config = await SampleLLMEngine.from_args(
        ["--disaggregation-mode", "prefill"]
    )
    assert engine.disaggregation_mode is DisaggregationMode.PREFILL
    assert worker_config.disaggregation_mode is DisaggregationMode.PREFILL
