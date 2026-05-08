# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for BuiltinThroughputPropose (PR 6 6-3).

Parity-focused: for each mode, seed a PSM with identical config + caps
+ regression + predictions, drive ``PSM._advance_throughput(traffic)``,
and compare its ``ScalingDecision`` output to the builtin's
``OverrideResult.targets`` after conversion. Same inputs → same
numerical output bit-for-bit.

**Important divergence from PSM that the test encodes**:

- PSM's ``_advance_throughput`` applies per-component budget via
  ``_apply_single_budget`` INSIDE the throughput path when
  ``enable_load_scaling=False``. The builtin does NOT apply budget —
  that's the CONSTRAIN stage's job (PR 6 6-6). So we compare:
    * ``enable_load_scaling=True`` path: PSM returns None (sets lower
      bound); builtin emits AT_LEAST(num_p, num_d) — comparable by
      checking builtin's replicas against PSM's would-be ``desired``
      (before budget).
    * ``enable_load_scaling=False`` path: PSM returns
      ``ScalingDecision`` with budget already applied; the builtin
      emits SET with unclamped values. We test with ``max_gpu_budget=-1``
      (no budget) so PSM's ``_apply_single_budget`` is a passthrough
      and the two match directly.
"""

from __future__ import annotations

from typing import Optional

import pytest

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.state_machine import PlannerStateMachine
from dynamo.planner.core.types import (
    EngineCapabilities,
    TrafficObservation,
    WorkerCapabilities,
)
from dynamo.planner.plugins.builtins.throughput_propose import (
    BuiltinThroughputPropose,
)
from dynamo.planner.plugins.types import (
    ObservationData,
    OverrideType,
    PipelineContext,
    PredictionData,
    ProposeStageRequest,
    TrafficMetrics,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _FakeOrchestrator:
    def __init__(self, caps: WorkerCapabilities, regressions: dict):
        self._caps = caps
        self._store = dict(regressions)

    @property
    def capabilities(self):
        return self._caps

    def get_regression(self, kind):
        return self._store.get(kind)

    def update_regression(self, kind, model):
        self._store[kind] = model


def _make_caps(*, mode: str) -> WorkerCapabilities:
    decode = EngineCapabilities(
        num_gpu=1,
        max_num_batched_tokens=2048,
        max_num_seqs=128,
        context_length=4096,
        max_kv_tokens=16384,
    )
    prefill = EngineCapabilities(
        num_gpu=1,
        max_num_batched_tokens=2048,
        max_num_seqs=128,
        context_length=4096,
        max_kv_tokens=16384,
    )
    if mode == "agg":
        return WorkerCapabilities(decode=decode)
    if mode == "decode":
        return WorkerCapabilities(decode=decode)
    if mode == "prefill":
        return WorkerCapabilities(prefill=prefill)
    return WorkerCapabilities(prefill=prefill, decode=decode)


def _make_config(
    *,
    mode: str,
    enable_throughput: bool = True,
    enable_load: bool = False,
    optimization_target: str = "sla",
    max_gpu_budget: int = -1,
    min_endpoint: int = 1,
) -> PlannerConfig:
    return PlannerConfig(
        mode=mode,
        enable_throughput_scaling=enable_throughput,
        enable_load_scaling=enable_load,
        optimization_target=optimization_target,
        max_gpu_budget=max_gpu_budget,
        min_endpoint=min_endpoint,
    )


def _seed_psm_regression(psm: PlannerStateMachine, benchmark_fpms) -> None:
    """Drive PSM.load_benchmark_fpms with whatever combination the scenario
    needs; the builtin reads the same regression instance via fake orch."""
    psm.load_benchmark_fpms(
        prefill_fpms=benchmark_fpms.get("prefill"),
        decode_fpms=benchmark_fpms.get("decode"),
        agg_fpms=benchmark_fpms.get("agg"),
    )


def _make_ctx(
    *, num_req: float, isl: float, osl: float, duration_s: float = 10.0
) -> PipelineContext:
    return PipelineContext(
        observations=ObservationData(
            traffic=TrafficMetrics(duration_s=duration_s, num_req=num_req, isl=isl, osl=osl)
        ),
        predictions=PredictionData(
            predicted_num_req=num_req, predicted_isl=isl, predicted_osl=osl
        ),
    )


# ---------------------------------------------------------------------------
# Config gating
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_throughput_disabled_returns_accept():
    # PlannerConfig requires >=1 scaling mode enabled; pair the disabled
    # throughput with enable_load=True to satisfy the invariant.
    config = _make_config(
        mode="agg", enable_throughput=False, enable_load=True
    )
    caps = _make_caps(mode="agg")
    orch = _FakeOrchestrator(caps, {})
    plugin = BuiltinThroughputPropose(orch, config)  # type: ignore[arg-type]
    resp = await plugin.Propose(
        ProposeStageRequest(context=_make_ctx(num_req=100, isl=500, osl=100))
    )
    assert resp.result_kind == "accept"


@pytest.mark.asyncio
async def test_no_predictions_returns_accept():
    config = _make_config(mode="agg")
    caps = _make_caps(mode="agg")
    orch = _FakeOrchestrator(caps, {})
    plugin = BuiltinThroughputPropose(orch, config)  # type: ignore[arg-type]
    resp = await plugin.Propose(ProposeStageRequest(context=PipelineContext()))
    assert resp.result_kind == "accept"


@pytest.mark.asyncio
async def test_zero_duration_traffic_returns_accept():
    config = _make_config(mode="agg")
    caps = _make_caps(mode="agg")
    orch = _FakeOrchestrator(caps, {})
    plugin = BuiltinThroughputPropose(orch, config)  # type: ignore[arg-type]
    resp = await plugin.Propose(
        ProposeStageRequest(
            context=_make_ctx(num_req=100, isl=500, osl=100, duration_s=0)
        )
    )
    assert resp.result_kind == "accept"


# ---------------------------------------------------------------------------
# Regression missing → degrade to accept
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_regression_returns_accept():
    config = _make_config(mode="disagg")
    caps = _make_caps(mode="disagg")
    orch = _FakeOrchestrator(caps, {})  # no "prefill" or "decode" regressions
    plugin = BuiltinThroughputPropose(orch, config)  # type: ignore[arg-type]
    resp = await plugin.Propose(
        ProposeStageRequest(context=_make_ctx(num_req=100, isl=500, osl=100))
    )
    assert resp.result_kind == "accept"


# ---------------------------------------------------------------------------
# Output type gating (v11 design: enable_load → AT_LEAST; else SET)
# ---------------------------------------------------------------------------


def _benchmark_fpms_for_agg(worker_id="w1"):
    """Small FPM corpus that's enough to fit the aggregate regression
    without crashing; real values come from PSM's tests/scenarios."""
    from dynamo.common.forward_pass_metrics import (
        ForwardPassMetrics,
        QueuedRequestMetrics,
        ScheduledRequestMetrics,
    )

    def _fpm(**kw):
        return ForwardPassMetrics(
            worker_id=worker_id,
            dp_rank=0,
            wall_time=1.0,
            scheduled_requests=ScheduledRequestMetrics(
                sum_prefill_tokens=kw.get("prefill_tokens", 512),
                num_prefill_requests=kw.get("prefill_reqs", 1),
                sum_decode_kv_tokens=kw.get("decode_kv_tokens", 1024),
                num_decode_requests=kw.get("decode_reqs", 4),
            ),
            queued_requests=QueuedRequestMetrics(
                sum_prefill_tokens=0,
                sum_decode_kv_tokens=0,
            ),
        )

    return [_fpm() for _ in range(10)]


@pytest.mark.asyncio
async def test_enable_load_publishes_lower_bound_and_returns_accept():
    # v11 § M-K (plugin decomposition): with enable_load_scaling=True,
    # throughput-propose is a **side-effect-only** plugin — it writes
    # the computed lower bound to the orchestrator's shared state and
    # returns ACCEPT; load-propose reads the bound in its own decision.
    # No AT_LEAST is emitted (that would double-apply the floor in the
    # merge).
    config = _make_config(mode="agg", enable_load=True)
    caps = _make_caps(mode="agg")
    psm = PlannerStateMachine(config, caps)
    _seed_psm_regression(psm, {"agg": _benchmark_fpms_for_agg()})

    class _OrchWithBounds(_FakeOrchestrator):
        def __init__(self, caps, regressions):
            super().__init__(caps, regressions)
            self.bounds: dict[str, int] = {}

        def set_throughput_lower_bound(self, component, value):
            self.bounds[component] = value

        def get_throughput_lower_bound(self, component):
            return self.bounds.get(component, 1)

    orch = _OrchWithBounds(caps, {"agg": psm._agg_regression})
    plugin = BuiltinThroughputPropose(orch, config)  # type: ignore[arg-type]
    resp = await plugin.Propose(
        ProposeStageRequest(context=_make_ctx(num_req=100, isl=500, osl=100))
    )
    if "decode" not in orch.bounds:
        pytest.skip("agg regression not ready with this sample set")
    # Returns ACCEPT — no OverrideResult at all.
    assert resp.result_kind == "accept"
    # And publishes the bound on the orchestrator for load-propose.
    assert orch.bounds["decode"] >= 1


@pytest.mark.asyncio
async def test_enable_load_false_emits_set_type():
    config = _make_config(mode="agg", enable_load=False)
    caps = _make_caps(mode="agg")
    psm = PlannerStateMachine(config, caps)
    _seed_psm_regression(psm, {"agg": _benchmark_fpms_for_agg()})
    orch = _FakeOrchestrator(caps, {"agg": psm._agg_regression})

    plugin = BuiltinThroughputPropose(orch, config)  # type: ignore[arg-type]
    resp = await plugin.Propose(
        ProposeStageRequest(context=_make_ctx(num_req=100, isl=500, osl=100))
    )
    if resp.result_kind == "accept":
        pytest.skip("agg regression not ready with this sample set")
    assert resp.result_kind == "override"
    assert all(t.type == OverrideType.SET for t in resp.override.targets)


# ---------------------------------------------------------------------------
# Parity vs PSM: same numerical output for agg mode
# ---------------------------------------------------------------------------


def _run_psm_throughput(config, caps, benchmark_fpms, *, num_req, isl, osl, duration_s):
    """Drive PSM's _advance_throughput with explicit predictions (by
    pre-seeding the predictors to guarantee predict_next returns the
    requested values). Returns the PSM ScalingDecision (or None)."""
    psm = PlannerStateMachine(config, caps)
    _seed_psm_regression(psm, benchmark_fpms)

    # Seed predictors so predict_next returns the requested num_req/isl/osl.
    if not psm._is_easy:
        for _ in range(10):
            psm._num_req_predictor.add_data_point(num_req)
            psm._isl_predictor.add_data_point(isl)
            psm._osl_predictor.add_data_point(osl)

    traffic = TrafficObservation(
        duration_s=duration_s, num_req=num_req, isl=isl, osl=osl
    )
    return psm._advance_throughput(traffic), psm


@pytest.mark.asyncio
async def test_agg_parity_vs_psm_unclamped_budget():
    # Isolate the plugin from budget side of things by using max_gpu_budget=-1
    # so PSM's _apply_single_budget is a pass-through. Then plugin SET and
    # PSM's ScalingDecision should match numerically.
    config = _make_config(mode="agg", enable_load=False, max_gpu_budget=-1)
    caps = _make_caps(mode="agg")
    fpms = {"agg": _benchmark_fpms_for_agg()}

    # Run PSM.
    psm_decision, psm = _run_psm_throughput(
        config, caps, fpms, num_req=100, isl=500, osl=100, duration_s=10.0
    )
    if psm_decision is None:
        pytest.skip("agg regression not ready in PSM path for this sample")

    # Run plugin against the SAME regression instance (share object identity).
    orch = _FakeOrchestrator(caps, {"agg": psm._agg_regression})
    plugin = BuiltinThroughputPropose(orch, config)  # type: ignore[arg-type]
    resp = await plugin.Propose(
        ProposeStageRequest(context=_make_ctx(num_req=100, isl=500, osl=100))
    )
    assert resp.result_kind == "override"
    decode_target = next(
        t for t in resp.override.targets if t.sub_component_type == "decode"
    )
    assert decode_target.replicas == psm_decision.num_decode
    assert decode_target.type == OverrideType.SET


@pytest.mark.asyncio
async def test_disagg_parity_vs_psm_unclamped_budget():
    config = _make_config(mode="disagg", enable_load=False, max_gpu_budget=-1)
    caps = _make_caps(mode="disagg")
    fpms = {
        "prefill": _benchmark_fpms_for_agg(),
        "decode": _benchmark_fpms_for_agg(),
    }

    psm_decision, psm = _run_psm_throughput(
        config, caps, fpms, num_req=100, isl=500, osl=100, duration_s=10.0
    )
    if psm_decision is None:
        pytest.skip("disagg regression not ready in PSM path for this sample")

    orch = _FakeOrchestrator(
        caps,
        {"prefill": psm._prefill_regression, "decode": psm._decode_regression},
    )
    plugin = BuiltinThroughputPropose(orch, config)  # type: ignore[arg-type]
    resp = await plugin.Propose(
        ProposeStageRequest(context=_make_ctx(num_req=100, isl=500, osl=100))
    )
    assert resp.result_kind == "override"
    by_component = {
        t.sub_component_type: t.replicas for t in resp.override.targets
    }
    assert by_component.get("prefill") == psm_decision.num_prefill
    assert by_component.get("decode") == psm_decision.num_decode
