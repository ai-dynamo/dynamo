# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""G3 behaviour parity through the 5 REAL builtin plugins (PR 6 sub-task 6-7).

The five-builtin chain:

- ``BuiltinLoadPredictor``   at PREDICT   (priority 1)
- ``BuiltinThroughputPropose`` at PROPOSE (priority 10)
- ``BuiltinLoadPropose``     at PROPOSE   (priority 5)
- ``BuiltinReconcile``       at RECONCILE (priority 1)
- ``BuiltinBudgetConstrain`` at CONSTRAIN (priority 1)

Scope of comparison
-------------------

We compare **scale_to only** (ScalingDecision.num_prefill /
num_decode), projected from the orchestrator's ``final_proposal``. We
do NOT compare ``diagnostics`` or ``next_tick`` because:

- v11 § Q2 (PR 6 doc): numeric diagnostic fields are scheduled to move
  from ``TickDiagnostics`` to Prometheus metrics — the golden fixture's
  diagnostic shape will be re-recorded in PR 6 6-8 against the plugin
  chain, not the current PSM shape.
- ``next_tick`` bookkeeping (``_next_load_s`` / ``_next_throughput_s``)
  has not yet been decomposed across plugins; it remains a PSM-shaped
  concern until PR 6 6-9 wires it through the orchestrator.

Until those land, bit-level parity is impossible; the scale_to parity
this file asserts is the most meaningful subset — it validates that
the 5 builtins in combination compute the same scaling decisions as
PSM.on_tick for the current 6-scenario fixture set.

No-change projection
--------------------

PSM returns ``scale_to=None`` when the per-tick decision matches
current replicas; the orchestrator always returns a concrete
``final_proposal.targets``. The projection in ``_outcome_to_effects``
maps ``execute_action=skip_no_targets`` or "targets equal current" back
to ``scale_to=None`` so the two semantic models align.
"""

from __future__ import annotations

import pytest

from dynamo.planner.core.state_machine import PlannerStateMachine
from dynamo.planner.core.types import (
    FpmObservations,
    ScalingDecision,
    WorkerCounts,
)
from dynamo.planner.plugins.builtins.budget_constrain import BuiltinBudgetConstrain
from dynamo.planner.plugins.builtins.load_predictor import BuiltinLoadPredictor
from dynamo.planner.plugins.builtins.load_propose import BuiltinLoadPropose
from dynamo.planner.plugins.builtins.reconcile import BuiltinReconcile
from dynamo.planner.plugins.builtins.throughput_propose import BuiltinThroughputPropose
from dynamo.planner.plugins.clock import WallClock
from dynamo.planner.plugins.merge.types import ComponentKey
from dynamo.planner.plugins.orchestrator.orchestrator import LocalPlannerOrchestrator
from dynamo.planner.plugins.registry.auth import AllowUnauthenticatedAuth
from dynamo.planner.plugins.registry.circuit_breaker import CircuitBreaker
from dynamo.planner.plugins.registry.server import PluginRegistryServer
from dynamo.planner.plugins.scheduler import PluginScheduler
from dynamo.planner.plugins.transport.config import (
    TransportConfig,
    make_transport_for_endpoint,
)
from dynamo.planner.plugins.types import (
    ObservationData,
    PipelineContext,
    PredictionData,
    TrafficMetrics,
)

from dynamo.planner.tests.plugins.g3_fixtures.dump_tool import (
    DEFAULT_OUTPUT_DIR,
    _read_fixture,
    _tick_for,
)
from dynamo.planner.tests.plugins.g3_fixtures.scenarios import ALL_SCENARIOS

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


# ---------------------------------------------------------------------------
# Orchestrator + builtin wiring
# ---------------------------------------------------------------------------


def _build_orchestrator_with_real_builtins(config, caps):
    """Construct a full 5-builtin orchestrator. Regression models are
    installed here but populated by the caller's bootstrap step."""
    clock = WallClock()
    cb = CircuitBreaker(clock)
    transport_config = TransportConfig(request_timeout_seconds=5.0)

    def factory(plugin_id, endpoint, *, in_process_instance=None):
        return make_transport_for_endpoint(
            plugin_id,
            endpoint,
            transport_config,
            in_process_instance=in_process_instance,
        )

    server = PluginRegistryServer(
        clock=clock,
        auth=AllowUnauthenticatedAuth(),
        circuit_breaker=cb,
        transport_factory=factory,
    )
    scheduler = PluginScheduler(server, cb, clock)
    orchestrator = LocalPlannerOrchestrator(
        registry=server,
        scheduler=scheduler,
        circuit_breaker=cb,
        clock=clock,
        tick_max_duration_seconds=30.0,
        capabilities=caps,
    )

    predictor = BuiltinLoadPredictor(orchestrator, config)
    throughput = BuiltinThroughputPropose(orchestrator, config)
    load = BuiltinLoadPropose(orchestrator, config)
    reconcile = BuiltinReconcile(orchestrator, config)
    budget = BuiltinBudgetConstrain(orchestrator, config)

    orchestrator.register_internal(
        plugin_id="builtin_load_predictor",
        plugin_type="predict",
        priority=1,
        instance=predictor,
    )
    # Priority: load (5) < throughput (10) so load wins in type_aware_merge
    # when both emit SET at PROPOSE — mirrors PSM's "load > throughput" rule.
    orchestrator.register_internal(
        plugin_id="builtin_load_propose",
        plugin_type="propose",
        priority=5,
        instance=load,
    )
    orchestrator.register_internal(
        plugin_id="builtin_throughput_propose",
        plugin_type="propose",
        priority=10,
        instance=throughput,
    )
    orchestrator.register_internal(
        plugin_id="builtin_reconcile",
        plugin_type="reconcile",
        priority=1,
        instance=reconcile,
    )
    orchestrator.register_internal(
        plugin_id="builtin_budget_constrain",
        plugin_type="constrain",
        priority=1,
        instance=budget,
    )
    return {
        "orchestrator": orchestrator,
        "predictor": predictor,
        "throughput": throughput,
        "load": load,
        "budget": budget,
    }


# ---------------------------------------------------------------------------
# Regression observation (mimics PSM._observe_fpm)
# ---------------------------------------------------------------------------


def _observe_fpm_into_regressions(orch, obs: FpmObservations, mode: str) -> None:
    """Feed an ``FpmObservations`` dict into whichever regression models
    the orchestrator holds — mirrors PSM's ``_observe_fpm`` side effect,
    which in PR 6 is intended to belong to the FPM-owning plugin but
    currently lives outside the stage chain. Called by the test harness
    between ticks so regression-state tracking is identical to PSM.
    """
    if mode == "agg":
        if obs.decode:
            agg = orch.get_regression("agg")
            if agg is not None:
                for fpm in obs.decode.values():
                    agg.add_observation(fpm)
        return
    if obs.prefill:
        p_reg = orch.get_regression("prefill")
        if p_reg is not None:
            for fpm in obs.prefill.values():
                p_reg.add_observation(fpm)
    if obs.decode:
        d_reg = orch.get_regression("decode")
        if d_reg is not None:
            for fpm in obs.decode.values():
                d_reg.add_observation(fpm)


# ---------------------------------------------------------------------------
# TickInput → PipelineContext
# ---------------------------------------------------------------------------


def _tick_input_to_context(ti) -> PipelineContext:
    traffic = None
    if ti.traffic is not None:
        traffic = TrafficMetrics(
            duration_s=ti.traffic.duration_s,
            num_req=ti.traffic.num_req,
            isl=ti.traffic.isl,
            osl=ti.traffic.osl,
        )
    return PipelineContext(
        request_id=f"tick-{ti.now_s}",
        decision_id=f"d-{ti.now_s}",
        observations=ObservationData(traffic=traffic),
    )


# ---------------------------------------------------------------------------
# Orchestrator outcome → PlannerEffects.scale_to
# ---------------------------------------------------------------------------


def _outcome_scale_to(outcome, worker_counts: WorkerCounts):
    """Project orchestrator output onto a ``ScalingDecision`` matching
    PSM's ``on_tick.effects.scale_to`` semantics:

    - ``execute_action != "apply"`` → ``None``.
    - ``targets`` equal to ``worker_counts`` ready numbers → ``None``
      (PSM returns None on "no change"; orchestrator returns current
      replicas since its model is "what should replicas be now").
    - Else → ``ScalingDecision(num_prefill, num_decode)``.
    """
    if outcome.execute_action != "apply" or outcome.final_proposal is None:
        return None

    by_comp = {t.sub_component_type: t.replicas for t in outcome.final_proposal.targets}
    num_p = by_comp.get("prefill")
    num_d = by_comp.get("decode")

    current_p = worker_counts.ready_num_prefill if worker_counts.ready_num_prefill is not None else None
    current_d = worker_counts.ready_num_decode if worker_counts.ready_num_decode is not None else None

    # "No change" detection — both components match their current count.
    p_unchanged = (num_p is None) or (num_p == current_p)
    d_unchanged = (num_d is None) or (num_d == current_d)
    if p_unchanged and d_unchanged:
        return None

    return ScalingDecision(
        num_prefill=num_p if num_p is not None else None,
        num_decode=num_d if num_d is not None else None,
    )


def _psm_scale_to_from_fixture(record) -> "ScalingDecision | None":
    """Parse the golden fixture record's ``planner_effects.scale_to``
    into a ``ScalingDecision`` (or ``None``)."""
    s = record["planner_effects"]["scale_to"]
    if s is None:
        return None
    return ScalingDecision(num_prefill=s.get("num_prefill"), num_decode=s.get("num_decode"))


# ---------------------------------------------------------------------------
# Scenario driver
# ---------------------------------------------------------------------------


async def _run_scenario_through_real_builtins(scenario):
    """Build orchestrator + real builtins + run scenario ticks. Returns
    list of ``ScalingDecision|None`` for each tick (indexed from 0)."""
    config = scenario.make_config()
    caps = scenario.caps_factory()

    # Bootstrap: (1) build regression models via a throwaway PSM running
    # the scenario's bootstrap_fn (mimicking ``PSM.load_benchmark_fpms``),
    # (2) install them on the orchestrator via ``install_regressions``,
    # (3) fan out Bootstrap RPC to plugins via ``bootstrap_plugins``.
    # The two calls are distinct per PR 6 6-9 split: regression install
    # is orchestrator-owned state; Bootstrap is a plugin lifecycle.
    bootstrap_psm = PlannerStateMachine(config, caps)
    if scenario.bootstrap_fn is not None:
        scenario.bootstrap_fn(bootstrap_psm)

    ctx = _build_orchestrator_with_real_builtins(config, caps)
    orchestrator = ctx["orchestrator"]
    orchestrator.install_regressions(
        prefill=getattr(bootstrap_psm, "_prefill_regression", None),
        decode=getattr(bootstrap_psm, "_decode_regression", None),
        agg=getattr(bootstrap_psm, "_agg_regression", None),
    )
    await orchestrator.bootstrap_plugins()

    decisions: list = []
    mode = config.mode

    for tick_input in scenario.ticks:
        scheduled_tick = _tick_for(tick_input)

        # 1. Observe FPM into regressions (PSM does this inside on_tick
        #    when ``run_load_scaling and not is_easy``). Mirror that here
        #    so regression state tracks PSM exactly.
        run_load = scheduled_tick.run_load_scaling
        is_easy = config.optimization_target != "sla"
        if run_load and not is_easy and tick_input.fpm_observations is not None:
            _observe_fpm_into_regressions(orchestrator, tick_input.fpm_observations, mode)

        # 2. Feed predictor with traffic (PSM does this inside
        #    ``_advance_throughput._predict_load``; our decomposition
        #    has the predictor plugin consume traffic from ctx).
        # 3. Prime the two plugins that take side-channel data.
        ctx["load"].prime_tick(tick_input.fpm_observations, tick_input.worker_counts)
        ctx["budget"].prime_tick(tick_input.worker_counts)

        # 4. Build PipelineContext and drive the orchestrator.
        pc = _tick_input_to_context(tick_input)
        # Pass worker counts as baseline so CONSTRAIN's budget merge has
        # something to clamp (its upstream is the PROPOSE-stage merge
        # output, threaded through RECONCILE passthrough and then
        # re-baselined for CONSTRAIN by pipeline._proposal_to_baseline).
        baseline: dict[ComponentKey, int] = {}
        if tick_input.worker_counts is not None:
            wc = tick_input.worker_counts
            if wc.ready_num_prefill is not None:
                baseline[ComponentKey(sub_component_type="prefill")] = wc.ready_num_prefill
            if wc.ready_num_decode is not None:
                baseline[ComponentKey(sub_component_type="decode")] = wc.ready_num_decode
        outcome = await orchestrator.tick(pc, baseline)

        decisions.append(_outcome_scale_to(outcome, tick_input.worker_counts or WorkerCounts()))

    await orchestrator.shutdown()
    return decisions


# ---------------------------------------------------------------------------
# The parity test (scale_to only)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=lambda s: s.name)
async def test_g3_real_parity_scale_to(scenario):
    fixture_path = DEFAULT_OUTPUT_DIR / f"{scenario.name}.jsonl"
    assert fixture_path.exists(), f"missing fixture: {fixture_path}"

    fixture = _read_fixture(fixture_path)
    expected_scale_to = [_psm_scale_to_from_fixture(rec) for rec in fixture[1:]]

    actual = await _run_scenario_through_real_builtins(scenario)

    mismatches = []
    for i, (exp, act) in enumerate(zip(expected_scale_to, actual)):
        if exp != act:
            mismatches.append(
                f"tick {i}: expected={exp} actual={act}"
            )

    assert not mismatches, (
        f"scale_to parity broken for scenario '{scenario.name}':\n  "
        + "\n  ".join(mismatches)
    )
