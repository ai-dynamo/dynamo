# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dual-path parity integration test.

For every G3 scenario, drive the same tick sequence through:

1. ``_PSMEngineAdapter`` wrapping ``PlannerStateMachine`` (legacy path)
2. ``OrchestratorEngineAdapter`` wrapping the 5-builtin plugin chain
   (orchestrator path)

Assert that both paths emit the same ``PlannerEffects.scale_to`` **and**
``PlannerEffects.next_tick.at_s`` per tick. Existing parity tests
compare each path against the frozen golden fixture independently; this
test gives a **single regression signal** when the paths diverge —
the thing that protects production cutover.

Scope this file validates:

- ``scale_to``: scaling-decision bit-parity. Already covered per-path in
  ``test_g3_fixture_parity.py`` + ``test_g3_real_parity.py`` but
  cross-asserted here.
- ``next_tick.at_s``: scheduling-cadence parity. Not covered in other
  tests — if the adapter's cadence drifts, main-loop tick timing in
  production drifts with it.

Not covered here (explicitly documented):

- ``diagnostics``: orchestrator path returns empty ``TickDiagnostics``
  (Prometheus migration pending); comparing would fail by design. The
  rollout runbook calls this out as the known observability regression.
- Connector call sequence: ``_apply_effects`` is path-agnostic (reads
  only ``effects.scale_to``), so scale_to parity implies connector
  parity. A separate test could assert this directly; today's scale_to
  parity is sufficient given the one-to-one mapping.
"""

from __future__ import annotations

import pytest

from dynamo.planner.core.engine_protocol import _PSMEngineAdapter
from dynamo.planner.core.state_machine import PlannerStateMachine
from dynamo.planner.plugins.orchestrator.engine_adapter import (
    OrchestratorEngineAdapter,
)
from dynamo.planner.tests.plugins.g3_fixtures.dump_tool import _tick_for
from dynamo.planner.tests.plugins.g3_fixtures.scenarios import ALL_SCENARIOS

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.planner,
]


async def _drive_psm(scenario):
    """Run scenario through the PSM path. Returns list of
    ``(scale_to, next_tick_at_s)`` per tick."""
    config = scenario.make_config()
    caps = scenario.caps_factory()
    psm = PlannerStateMachine(config, caps)
    if scenario.bootstrap_fn is not None:
        scenario.bootstrap_fn(psm)

    engine = _PSMEngineAdapter(psm)
    # ``_PSMEngineAdapter.initial_tick`` forwards to PSM.initial_tick
    # which sets the internal cadence state. Call once so subsequent
    # ticks produce correct next_tick values.
    engine.initial_tick(scenario.initial_tick_at_s)

    decisions = []
    for ti in scenario.ticks:
        scheduled = _tick_for(ti)
        effects = await engine.tick(scheduled, ti)
        decisions.append(
            (effects.scale_to, effects.next_tick.at_s if effects.next_tick else None)
        )

    await engine.shutdown()
    return decisions


async def _drive_orchestrator(scenario):
    """Run scenario through the orchestrator adapter path. Returns
    list of ``(scale_to, next_tick_at_s)`` per tick."""
    config = scenario.make_config()
    caps = scenario.caps_factory()

    # Bootstrap regressions via the adapter's `bootstrap_from_fpms` helper
    # if the scenario's PSM bootstrap ran load_benchmark_fpms. The
    # simplest faithful reproduction: apply the scenario's bootstrap_fn
    # to a throwaway PSM, then hand its regression instances directly.
    throwaway = PlannerStateMachine(config, caps)
    if scenario.bootstrap_fn is not None:
        scenario.bootstrap_fn(throwaway)

    adapter = OrchestratorEngineAdapter(config, caps)
    adapter.install_regressions(
        prefill=getattr(throwaway, "_prefill_regression", None),
        decode=getattr(throwaway, "_decode_regression", None),
        agg=getattr(throwaway, "_agg_regression", None),
    )
    await adapter.bootstrap_plugins()
    adapter.initial_tick(scenario.initial_tick_at_s)

    decisions = []
    for ti in scenario.ticks:
        scheduled = _tick_for(ti)
        effects = await adapter.tick(scheduled, ti)
        decisions.append(
            (effects.scale_to, effects.next_tick.at_s if effects.next_tick else None)
        )

    await adapter.shutdown()
    return decisions


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=lambda s: s.name)
async def test_scale_to_matches_across_paths(scenario):
    psm_decisions = await _drive_psm(scenario)
    orch_decisions = await _drive_orchestrator(scenario)

    assert len(psm_decisions) == len(orch_decisions), (
        f"scenario={scenario.name}: tick-count mismatch — "
        f"psm={len(psm_decisions)} orch={len(orch_decisions)}"
    )
    for i, ((psm_scale, _), (orch_scale, _)) in enumerate(
        zip(psm_decisions, orch_decisions)
    ):
        assert psm_scale == orch_scale, (
            f"scenario={scenario.name} tick={i} scale_to drift:\n"
            f"  psm:  {psm_scale}\n"
            f"  orch: {orch_scale}"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=lambda s: s.name)
async def test_next_tick_at_s_matches_across_paths(scenario):
    """This is the parity assertion the existing per-path tests don't
    check. Main loop cadence depends on ``next_tick.at_s`` — if the
    adapter computes it differently from PSM, ``run()`` will tick at
    different times and the two paths diverge observationally even
    though ``scale_to`` agrees tick-by-tick."""
    psm_decisions = await _drive_psm(scenario)
    orch_decisions = await _drive_orchestrator(scenario)

    for i, ((_, psm_at), (_, orch_at)) in enumerate(
        zip(psm_decisions, orch_decisions)
    ):
        assert psm_at == orch_at, (
            f"scenario={scenario.name} tick={i} next_tick.at_s drift: "
            f"psm={psm_at} orch={orch_at}"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=lambda s: s.name)
async def test_initial_tick_matches_across_paths(scenario):
    """Sanity: ``initial_tick`` must agree too. Covered in
    ``test_engine_protocol.py::test_initial_tick_forwards_to_psm`` for
    PSM adapter and ``test_engine_adapter.py::test_initial_tick_matches_psm``
    for one scenario on the orchestrator side — this extends to every
    G3 scenario."""
    from dynamo.planner.core.engine_protocol import _PSMEngineAdapter

    config = scenario.make_config()
    caps = scenario.caps_factory()

    psm = PlannerStateMachine(config, caps)
    psm_engine = _PSMEngineAdapter(psm)
    psm_initial = psm_engine.initial_tick(scenario.initial_tick_at_s)

    adapter = OrchestratorEngineAdapter(config, caps)
    orch_initial = adapter.initial_tick(scenario.initial_tick_at_s)

    assert psm_initial.at_s == orch_initial.at_s
    assert psm_initial.run_load_scaling == orch_initial.run_load_scaling
    assert psm_initial.run_throughput_scaling == orch_initial.run_throughput_scaling
    assert psm_initial.need_worker_states == orch_initial.need_worker_states
    assert psm_initial.need_worker_fpm == orch_initial.need_worker_fpm
    assert psm_initial.need_traffic_metrics == orch_initial.need_traffic_metrics
    assert (
        psm_initial.traffic_metrics_duration_s
        == orch_initial.traffic_metrics_duration_s
    )

    await adapter.shutdown()
