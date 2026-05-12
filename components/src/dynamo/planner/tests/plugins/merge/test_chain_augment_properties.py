# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hypothesis property-based tests for chain_augment.

Invariants:

1. **Input order independence** — result is independent of the order the
   caller passes plugins in. ``chain_augment`` sorts internally by priority
   descending; permuting the input yields the same ``prediction`` and
   ``final_from``. (Caveat: when multiple plugins share the same priority,
   Python's ``sorted`` is stable and iteration order within the tie can
   matter; the strategy generates unique priorities to avoid that edge
   case, consistent with v1 semantics that expect distinct priorities.)
2. **Final break** — if any plugin returns ``final=True``, every plugin
   with a strictly smaller priority number (= higher precedence) that
   would follow it in the sorted chain is **never** called.
3. **Passthrough preservation** — if every plugin returns
   ``predictions=None`` (ACCEPT), the resulting ``prediction`` is ``None``
   and ``final_from`` is empty.

Requires ``hypothesis`` (dev extra). Auto-skipped when unavailable.
"""

from __future__ import annotations

import asyncio

import pytest

hypothesis = pytest.importorskip("hypothesis")

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

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
    def __init__(self, plugin_id, priority, response):
        self.plugin_id = plugin_id
        self.priority = priority
        self._response = response
        self.call_count = 0

    async def call(self, method, context):
        self.call_count += 1
        return self._response


def _predict_response(num_req=None, isl=None, osl=None, final=False):
    preds = (
        None
        if num_req is None and isl is None and osl is None
        else PredictionData(
            predicted_num_req=num_req,
            predicted_isl=isl,
            predicted_osl=osl,
        )
    )
    return PredictStageResponse(predictions=preds, final=final)


@st.composite
def plugin_specs(draw, *, min_plugins=1, max_plugins=4):
    """Generate a list of (plugin_id, priority, response) with distinct priorities."""
    n = draw(st.integers(min_value=min_plugins, max_value=max_plugins))
    priorities = draw(
        st.lists(
            st.integers(min_value=1, max_value=200),
            min_size=n,
            max_size=n,
            unique=True,
        )
    )
    specs = []
    for i, prio in enumerate(priorities):
        num_req = draw(st.one_of(st.none(), st.floats(0.0, 1000.0, allow_nan=False)))
        isl = draw(st.one_of(st.none(), st.floats(0.0, 10000.0, allow_nan=False)))
        osl = draw(st.one_of(st.none(), st.floats(0.0, 10000.0, allow_nan=False)))
        final = draw(st.booleans())
        specs.append(
            (
                f"p{i}",
                prio,
                _predict_response(num_req=num_req, isl=isl, osl=osl, final=final),
            )
        )
    return specs


def _build_plugins(specs):
    return [_StubPlugin(pid, prio, resp) for pid, prio, resp in specs]


# ----------------------------------------------------------------------------
# Invariant 1 — input order independence
# ----------------------------------------------------------------------------


@given(specs=plugin_specs(min_plugins=1, max_plugins=4), perm_seed=st.integers(0, 1000))
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
def test_result_independent_of_caller_input_order(specs, perm_seed):
    import random

    rnd = random.Random(perm_seed)
    shuffled = list(specs)
    rnd.shuffle(shuffled)

    out_a = asyncio.run(chain_augment(_build_plugins(specs), PipelineContext()))
    out_b = asyncio.run(chain_augment(_build_plugins(shuffled), PipelineContext()))

    assert out_a.final_from == out_b.final_from
    if out_a.prediction is None:
        assert out_b.prediction is None
    else:
        assert out_b.prediction is not None
        assert out_a.prediction.predicted_num_req == out_b.prediction.predicted_num_req
        assert out_a.prediction.predicted_isl == out_b.prediction.predicted_isl
        assert out_a.prediction.predicted_osl == out_b.prediction.predicted_osl


# ----------------------------------------------------------------------------
# Invariant 2 — final break stops downstream plugins
# ----------------------------------------------------------------------------


@given(specs=plugin_specs(min_plugins=2, max_plugins=4))
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
def test_final_break_stops_subsequent_plugins(specs):
    plugins = _build_plugins(specs)
    asyncio.run(chain_augment(plugins, PipelineContext()))
    # Determine the first plugin in priority-descending order that returned final=True.
    by_priority_desc = sorted(plugins, key=lambda p: -p.priority)
    final_idx = next(
        (i for i, p in enumerate(by_priority_desc) if p._response.final), None
    )
    if final_idx is None:
        # No final; every plugin called exactly once.
        for p in plugins:
            assert p.call_count == 1
    else:
        # First final plugin and every prior plugin called; subsequent plugins not called.
        for i, p in enumerate(by_priority_desc):
            expected = 1 if i <= final_idx else 0
            assert p.call_count == expected, (
                f"plugin {p.plugin_id} (prio={p.priority}): call_count={p.call_count}, "
                f"expected={expected}, sorted_index={i}, final_idx={final_idx}"
            )


# ----------------------------------------------------------------------------
# Invariant 3 — passthrough (all None predictions) preserves prediction=None
# ----------------------------------------------------------------------------


@given(
    n=st.integers(min_value=0, max_value=5),
    priorities=st.lists(
        st.integers(min_value=1, max_value=200), min_size=0, max_size=5, unique=True
    ),
)
@settings(max_examples=40, suppress_health_check=[HealthCheck.too_slow])
def test_all_accept_yields_none_prediction(n, priorities):
    plugins = [
        _StubPlugin(f"p{i}", prio, PredictStageResponse())
        for i, prio in enumerate(priorities[:n])
    ]
    out = asyncio.run(chain_augment(plugins, PipelineContext()))
    assert out.prediction is None
    assert out.final_from == ""
    assert out.misuse_warnings == []
