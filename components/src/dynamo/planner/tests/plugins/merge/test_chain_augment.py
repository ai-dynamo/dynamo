# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for chain_augment (PR 4 sub-task 4-5).

Covered patterns:
- Replace: single plugin emits complete PredictionData
- Patch: higher-priority plugin overrides one field only
- Augment: plugins fill disjoint fields
- Passthrough: all plugins emit predictions=None (ACCEPT)
- final break: chain stops at final=true, subsequent plugin never called
- final misuse: non-lowest-priority final => warning + downstream skipped
- final correct: lowest-priority final => no warning
- Multiple finals in chain: first-encountered (non-lowest) wins + warning
- partial-merge preserves earlier fields when later plugin has None
- Empty chain / mixed priority order from caller

Note (deviation from PR 4 doc v1.1 pseudocode): PR 1 as-built does not
expose a REJECT mechanism in ``PredictStageResponse`` (the proto message
has only ``predictions``/``reason``/``final``). So the ``degraded`` field
on ``ChainAugmentOutcome`` is always empty in v1. A future proto revision
can introduce explicit reject; tests here assert ``degraded == []``.
"""

from __future__ import annotations

import pytest

from dynamo.planner.plugins.merge import chain_augment
from dynamo.planner.plugins.types import (
    PipelineContext,
    PredictionData,
    PredictStageResponse,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


class _StubPlugin:
    """Minimal PredictPluginCallable for tests — returns a queued
    ``PredictStageResponse`` on each ``call``; counts invocations."""

    def __init__(self, plugin_id: str, priority: int, responses):
        self.plugin_id = plugin_id
        self.priority = priority
        self._responses = list(responses)
        self.call_count = 0
        self.seen_contexts: list[PipelineContext] = []

    async def call(self, method: str, context: PipelineContext) -> PredictStageResponse:
        assert method == "Predict"
        self.call_count += 1
        self.seen_contexts.append(context)
        return self._responses.pop(0)


def _pd(num_req=None, isl=None, osl=None, source=""):
    return PredictionData(
        predicted_num_req=num_req,
        predicted_isl=isl,
        predicted_osl=osl,
        source=source,
    )


# ---------------------------------------------------------------------------
# Replace / Patch / Augment / Passthrough
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_replace_single_plugin_complete_prediction():
    p = _StubPlugin(
        "p1",
        10,
        [PredictStageResponse(predictions=_pd(num_req=1000, isl=3000, osl=150))],
    )
    out = await chain_augment([p], PipelineContext())
    assert out.prediction is not None
    assert out.prediction.predicted_num_req == 1000
    assert out.prediction.predicted_isl == 3000
    assert out.prediction.predicted_osl == 150
    assert out.final_from == ""
    assert out.degraded == []
    assert out.misuse_warnings == []


@pytest.mark.asyncio
async def test_patch_high_priority_overrides_single_field():
    # Caller passes arbitrary order; chain_augment sorts priority-descending.
    # priority=100 runs first (low precedence); priority=10 runs last (high).
    low = _StubPlugin(
        "low",
        100,
        [PredictStageResponse(predictions=_pd(num_req=1000, isl=3000, osl=150))],
    )
    high = _StubPlugin(
        "high",
        10,
        [PredictStageResponse(predictions=_pd(num_req=1200))],
    )
    out = await chain_augment([high, low], PipelineContext())
    assert out.prediction is not None
    # high overrode num_req; isl/osl preserved from low.
    assert out.prediction.predicted_num_req == 1200
    assert out.prediction.predicted_isl == 3000
    assert out.prediction.predicted_osl == 150


@pytest.mark.asyncio
async def test_augment_disjoint_fields_merge():
    a = _StubPlugin("A", 100, [PredictStageResponse(predictions=_pd(num_req=1000))])
    b = _StubPlugin("B", 10, [PredictStageResponse(predictions=_pd(isl=3000, osl=150))])
    out = await chain_augment([a, b], PipelineContext())
    assert out.prediction is not None
    assert out.prediction.predicted_num_req == 1000
    assert out.prediction.predicted_isl == 3000
    assert out.prediction.predicted_osl == 150


@pytest.mark.asyncio
async def test_passthrough_all_plugins_accept():
    a = _StubPlugin("A", 100, [PredictStageResponse()])
    b = _StubPlugin("B", 10, [PredictStageResponse()])
    out = await chain_augment([a, b], PipelineContext())
    assert out.prediction is None
    assert out.final_from == ""


@pytest.mark.asyncio
async def test_predictions_none_preserves_prior():
    # A emits a full PredictionData; B emits predictions=None (ACCEPT).
    # Running prediction from A persists unchanged after B.
    a = _StubPlugin(
        "A",
        100,
        [PredictStageResponse(predictions=_pd(num_req=1000, isl=3000, osl=150))],
    )
    b = _StubPlugin("B", 10, [PredictStageResponse()])
    out = await chain_augment([a, b], PipelineContext())
    assert out.prediction is not None
    assert out.prediction.predicted_num_req == 1000
    assert out.prediction.predicted_isl == 3000
    assert out.prediction.predicted_osl == 150


@pytest.mark.asyncio
async def test_source_field_merges_new_then_prev():
    a = _StubPlugin("A", 100, [PredictStageResponse(predictions=_pd(num_req=1.0, source="base"))])
    b = _StubPlugin("B", 10, [PredictStageResponse(predictions=_pd(isl=2.0, source="patch"))])
    out = await chain_augment([a, b], PipelineContext())
    assert out.prediction is not None
    assert out.prediction.source == "patch"


@pytest.mark.asyncio
async def test_source_falls_back_to_prev_when_new_empty():
    a = _StubPlugin("A", 100, [PredictStageResponse(predictions=_pd(num_req=1.0, source="base"))])
    b = _StubPlugin("B", 10, [PredictStageResponse(predictions=_pd(isl=2.0))])  # source=""
    out = await chain_augment([a, b], PipelineContext())
    assert out.prediction is not None
    assert out.prediction.source == "base"


# ---------------------------------------------------------------------------
# final break semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_final_breaks_chain_and_subsequent_plugins_never_called():
    # Sort desc: [p100 (100), p50 (50), p10 (10)]; p50 returns final → break.
    p100 = _StubPlugin("p100", 100, [PredictStageResponse(predictions=_pd(num_req=500))])
    p50 = _StubPlugin("p50", 50, [PredictStageResponse(predictions=_pd(isl=2000), final=True)])
    p10 = _StubPlugin("p10", 10, [PredictStageResponse(predictions=_pd(osl=100))])
    out = await chain_augment([p100, p50, p10], PipelineContext())
    assert out.final_from == "p50"
    assert p100.call_count == 1
    assert p50.call_count == 1
    assert p10.call_count == 0
    assert out.prediction is not None
    assert out.prediction.predicted_num_req == 500
    assert out.prediction.predicted_isl == 2000
    assert out.prediction.predicted_osl is None
    # p50 is not lowest priority (10 is) → misuse warning.
    assert len(out.misuse_warnings) == 1
    assert "p50" in out.misuse_warnings[0]


@pytest.mark.asyncio
async def test_final_at_lowest_priority_no_warning():
    # emergency (priority=5) is lowest-priority → runs last → final=true OK.
    low = _StubPlugin("low", 100, [PredictStageResponse(predictions=_pd(num_req=500))])
    emergency = _StubPlugin(
        "emergency",
        5,
        [PredictStageResponse(predictions=_pd(num_req=1000), final=True)],
    )
    out = await chain_augment([low, emergency], PipelineContext())
    assert out.final_from == "emergency"
    assert out.misuse_warnings == []
    assert out.prediction is not None
    assert out.prediction.predicted_num_req == 1000  # emergency overrode low


@pytest.mark.asyncio
async def test_final_at_non_lowest_priority_warns_and_skips_higher_precedence():
    # Misuse: mid-prio plugin returns final → higher-precedence emergency never runs.
    mid = _StubPlugin(
        "mid",
        100,
        [PredictStageResponse(predictions=_pd(num_req=500), final=True)],
    )
    emergency = _StubPlugin(
        "emergency",
        5,
        [PredictStageResponse(predictions=_pd(num_req=9000))],
    )
    out = await chain_augment([mid, emergency], PipelineContext())
    assert out.final_from == "mid"
    assert mid.call_count == 1
    assert emergency.call_count == 0
    assert len(out.misuse_warnings) == 1
    warning = out.misuse_warnings[0]
    assert "mid" in warning
    assert "priority=100" in warning
    assert "lowest_priority=5" in warning


@pytest.mark.asyncio
async def test_multiple_finals_first_in_sorted_order_wins():
    # Both A (100) and B (5) are final=True.
    # Sort desc: [A (100), B (5)] → A runs first, triggers break, B never runs.
    # A.priority (100) != lowest (5) → misuse warning for A.
    a = _StubPlugin(
        "A", 100, [PredictStageResponse(predictions=_pd(num_req=100), final=True)]
    )
    b = _StubPlugin(
        "B", 5, [PredictStageResponse(predictions=_pd(num_req=200), final=True)]
    )
    out = await chain_augment([a, b], PipelineContext())
    assert out.final_from == "A"
    assert a.call_count == 1
    assert b.call_count == 0
    assert len(out.misuse_warnings) == 1
    assert "A" in out.misuse_warnings[0]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_chain_returns_empty_outcome():
    out = await chain_augment([], PipelineContext())
    assert out.prediction is None
    assert out.final_from == ""
    assert out.degraded == []
    assert out.misuse_warnings == []


@pytest.mark.asyncio
async def test_running_prediction_threaded_via_context_predictions():
    # Each plugin should see, via its context, the merged prediction from
    # the plugins that ran before it in the sort order.
    first = _StubPlugin(
        "first", 100, [PredictStageResponse(predictions=_pd(num_req=42.0))]
    )
    second = _StubPlugin("second", 10, [PredictStageResponse()])  # just observes
    await chain_augment([first, second], PipelineContext())
    # first sees predictions=None (chain starts fresh); second sees first's output
    assert first.seen_contexts[0].predictions is None
    assert second.seen_contexts[0].predictions is not None
    assert second.seen_contexts[0].predictions.predicted_num_req == 42.0


@pytest.mark.asyncio
async def test_zero_float_value_preserved_not_treated_as_unset():
    # PredictionData fields are Optional[float]: 0.0 means "I assert 0",
    # None means "no opinion". Partial-merge must distinguish them.
    a = _StubPlugin(
        "A", 100, [PredictStageResponse(predictions=_pd(num_req=1000.0, isl=3000.0, osl=150.0))]
    )
    b = _StubPlugin(
        "B", 10, [PredictStageResponse(predictions=_pd(num_req=0.0))]
    )
    out = await chain_augment([a, b], PipelineContext())
    assert out.prediction is not None
    assert out.prediction.predicted_num_req == 0.0  # B's assertion survives
    assert out.prediction.predicted_isl == 3000.0
    assert out.prediction.predicted_osl == 150.0


@pytest.mark.asyncio
async def test_chain_preserves_initial_context_non_prediction_fields():
    initial = PipelineContext(request_id="req-42", decision_id="dec-7")
    spy = _StubPlugin("spy", 10, [PredictStageResponse()])
    await chain_augment([spy], initial)
    # The plugin's received context should carry the id fields through.
    assert spy.seen_contexts[0].request_id == "req-42"
    assert spy.seen_contexts[0].decision_id == "dec-7"
