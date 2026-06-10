# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for chain_augment.

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

Note: the as-built ``PredictStageResponse`` does not expose a REJECT
mechanism (the proto message has only ``predictions`` / ``reason`` /
``final``). So the ``degraded`` field on ``ChainAugmentOutcome`` is
always empty. A future proto revision can introduce explicit reject;
tests here assert ``degraded == []``.
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


def _pd(num_req=None, isl=None, osl=None, kv=None, accept=None, source=""):
    return PredictionData(
        predicted_num_req=num_req,
        predicted_isl=isl,
        predicted_osl=osl,
        predicted_kv_hit_rate=kv,
        predicted_accept_length=accept,
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
    assert out.chain_break_warnings == []


@pytest.mark.asyncio
async def test_predicted_kv_hit_rate_merges_across_chain():
    # predicted_kv_hit_rate must participate in first-writer-wins partial
    # merge like the other three predicted_* fields. Regression guard: it
    # was missing from _PREDICTION_FIELDS, so any 2+ plugin chain dropped it.
    high = _StubPlugin(
        "high",
        10,
        [PredictStageResponse(predictions=_pd(num_req=1200, kv=0.42))],
    )
    low = _StubPlugin(
        "low",
        100,
        [PredictStageResponse(predictions=_pd(isl=3000, osl=150, kv=0.99))],
    )
    out = await chain_augment([high, low], PipelineContext())
    assert out.prediction is not None
    # high (smaller priority) wrote kv=0.42 first → first-writer-wins
    assert out.prediction.predicted_kv_hit_rate == 0.42
    # and the disjoint fields from low still fill in
    assert out.prediction.predicted_isl == 3000
    assert out.prediction.predicted_osl == 150


@pytest.mark.asyncio
async def test_predicted_kv_hit_rate_fills_from_later_plugin_when_unset():
    # high leaves kv unset (None) → low's kv fills the gap.
    high = _StubPlugin(
        "high", 10, [PredictStageResponse(predictions=_pd(num_req=1200))]
    )
    low = _StubPlugin("low", 100, [PredictStageResponse(predictions=_pd(kv=0.7))])
    out = await chain_augment([high, low], PipelineContext())
    assert out.prediction is not None
    assert out.prediction.predicted_kv_hit_rate == 0.7


@pytest.mark.asyncio
async def test_predicted_accept_length_merges_across_chain():
    high = _StubPlugin(
        "high",
        10,
        [PredictStageResponse(predictions=_pd(num_req=1200, accept=2.0))],
    )
    low = _StubPlugin(
        "low",
        100,
        [PredictStageResponse(predictions=_pd(isl=3000, osl=150, accept=3.0))],
    )
    out = await chain_augment([high, low], PipelineContext())
    assert out.prediction is not None
    assert out.prediction.predicted_accept_length == 2.0
    assert out.prediction.predicted_isl == 3000
    assert out.prediction.predicted_osl == 150


@pytest.mark.asyncio
async def test_predicted_accept_length_fills_from_later_plugin_when_unset():
    high = _StubPlugin(
        "high", 10, [PredictStageResponse(predictions=_pd(num_req=1200))]
    )
    low = _StubPlugin("low", 100, [PredictStageResponse(predictions=_pd(accept=2.5))])
    out = await chain_augment([high, low], PipelineContext())
    assert out.prediction is not None
    assert out.prediction.predicted_accept_length == 2.5


@pytest.mark.asyncio
async def test_patch_high_priority_overrides_single_field():
    # Caller passes arbitrary order; chain_augment sorts priority-ascending.
    # priority=10 (high precedence) runs first and writes num_req=1200;
    # priority=100 (low precedence) runs last — first-writer-wins keeps
    # num_req=1200, and low's isl/osl fill the gaps high left as None.
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
    # Sort asc: B (10) runs first, returns predictions=None (no opinion) so
    # the running prediction stays None. A (100) runs second and emits a full
    # PredictionData. The chain returns A's prediction verbatim — a None
    # response from a higher-precedence plugin doesn't poison later writers.
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
async def test_source_higher_precedence_wins_when_non_empty():
    # Sort asc: B (10) runs first with source="patch"; A (100) runs second
    # with source="base". First-writer-wins for source: B's non-empty value
    # is preserved.
    a = _StubPlugin(
        "A", 100, [PredictStageResponse(predictions=_pd(num_req=1.0, source="base"))]
    )
    b = _StubPlugin(
        "B", 10, [PredictStageResponse(predictions=_pd(isl=2.0, source="patch"))]
    )
    out = await chain_augment([a, b], PipelineContext())
    assert out.prediction is not None
    assert out.prediction.source == "patch"


@pytest.mark.asyncio
async def test_source_falls_back_to_lower_precedence_when_higher_empty():
    # Sort asc: B (10) runs first with source=""; A (100) runs second with
    # source="base". Empty string is treated as "no opinion" for source,
    # so A's value fills in.
    a = _StubPlugin(
        "A", 100, [PredictStageResponse(predictions=_pd(num_req=1.0, source="base"))]
    )
    b = _StubPlugin(
        "B", 10, [PredictStageResponse(predictions=_pd(isl=2.0))]
    )  # source=""
    out = await chain_augment([a, b], PipelineContext())
    assert out.prediction is not None
    assert out.prediction.source == "base"


# ---------------------------------------------------------------------------
# final break semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_final_breaks_chain_and_subsequent_plugins_never_called():
    # Sort asc: [p10 (10), p50 (50), p100 (100)]; p50 returns final → break.
    # p10 ran first (set osl=100), p50 ran second (set isl=2000 + final),
    # p100 never gets to fill predicted_num_req. Misuse warning fires
    # because p50 isn't the lowest-priority plugin in the chain (p10 is).
    p100 = _StubPlugin(
        "p100", 100, [PredictStageResponse(predictions=_pd(num_req=500))]
    )
    p50 = _StubPlugin(
        "p50", 50, [PredictStageResponse(predictions=_pd(isl=2000), final=True)]
    )
    p10 = _StubPlugin("p10", 10, [PredictStageResponse(predictions=_pd(osl=100))])
    out = await chain_augment([p100, p50, p10], PipelineContext())
    assert out.final_from == "p50"
    assert p10.call_count == 1
    assert p50.call_count == 1
    assert p100.call_count == 0
    assert out.prediction is not None
    assert out.prediction.predicted_num_req is None  # p100 never ran
    assert out.prediction.predicted_isl == 2000  # p50 filled this
    assert out.prediction.predicted_osl == 100  # p10 filled this
    # p50 is not lowest priority (p10 is) → misuse warning.
    assert len(out.chain_break_warnings) == 1
    assert "p50" in out.chain_break_warnings[0]


@pytest.mark.asyncio
async def test_final_at_lowest_priority_no_warning():
    # Sort asc: [emergency (5), low (100)]. emergency runs first, sets final →
    # break. emergency.priority equals lowest_priority → no misuse warning.
    # low never runs (chain short-circuited by the authoritative plugin).
    low = _StubPlugin("low", 100, [PredictStageResponse(predictions=_pd(num_req=500))])
    emergency = _StubPlugin(
        "emergency",
        5,
        [PredictStageResponse(predictions=_pd(num_req=1000), final=True)],
    )
    out = await chain_augment([low, emergency], PipelineContext())
    assert out.final_from == "emergency"
    assert emergency.call_count == 1
    assert low.call_count == 0
    assert out.chain_break_warnings == []
    assert out.prediction is not None
    assert out.prediction.predicted_num_req == 1000


@pytest.mark.asyncio
async def test_final_at_non_lowest_priority_warns_and_skips_lower_precedence():
    # Sort asc: [emergency (5), mid (50), low (100)]. mid returns final → break.
    # emergency ran first (no final), mid ran second and short-circuited the
    # chain. low (lower precedence) is correctly skipped. The misuse warning
    # fires because mid is NOT the lowest-priority plugin in the chain — the
    # authoritative emergency had already weighed in, but using final=true from
    # a mid-priority plugin is still a configuration smell.
    emergency = _StubPlugin(
        "emergency",
        5,
        [PredictStageResponse(predictions=_pd(num_req=9000))],
    )
    mid = _StubPlugin(
        "mid",
        50,
        [PredictStageResponse(predictions=_pd(isl=500), final=True)],
    )
    low = _StubPlugin(
        "low",
        100,
        [PredictStageResponse(predictions=_pd(osl=100))],
    )
    out = await chain_augment([mid, emergency, low], PipelineContext())
    assert out.final_from == "mid"
    assert emergency.call_count == 1
    assert mid.call_count == 1
    assert low.call_count == 0
    assert len(out.chain_break_warnings) == 1
    warning = out.chain_break_warnings[0]
    assert "mid" in warning
    assert "priority=50" in warning
    assert "lowest_priority=5" in warning


@pytest.mark.asyncio
async def test_multiple_finals_first_in_sorted_order_wins():
    # Both A (100) and B (5) are final=True.
    # Sort asc: [B (5), A (100)] → B runs first, triggers break, A never runs.
    # B.priority (5) == lowest_priority → no misuse warning.
    a = _StubPlugin(
        "A", 100, [PredictStageResponse(predictions=_pd(num_req=100), final=True)]
    )
    b = _StubPlugin(
        "B", 5, [PredictStageResponse(predictions=_pd(num_req=200), final=True)]
    )
    out = await chain_augment([a, b], PipelineContext())
    assert out.final_from == "B"
    assert b.call_count == 1
    assert a.call_count == 0
    assert out.chain_break_warnings == []
    assert out.prediction is not None
    assert out.prediction.predicted_num_req == 200


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_chain_returns_empty_outcome():
    out = await chain_augment([], PipelineContext())
    assert out.prediction is None
    assert out.final_from == ""
    assert out.degraded == []
    assert out.chain_break_warnings == []


@pytest.mark.asyncio
async def test_running_prediction_threaded_via_context_predictions():
    # Each plugin should see, via its context, the merged prediction from
    # the plugins that ran before it in the sort order. Asc-order: smaller
    # priority runs first, so `first` (priority=10) precedes `second` (100).
    first = _StubPlugin(
        "first", 10, [PredictStageResponse(predictions=_pd(num_req=42.0))]
    )
    second = _StubPlugin("second", 100, [PredictStageResponse()])  # just observes
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
        "A",
        100,
        [PredictStageResponse(predictions=_pd(num_req=1000.0, isl=3000.0, osl=150.0))],
    )
    b = _StubPlugin("B", 10, [PredictStageResponse(predictions=_pd(num_req=0.0))])
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
