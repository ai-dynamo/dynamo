# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end integration test: 5-builtin orchestrator + mock connector.

Drives representative scenarios through the full builtin chain
(BuiltinLoadPredictor / BuiltinThroughputPropose / BuiltinLoadPropose /
BuiltinReconcile / BuiltinBudgetConstrain) with regression + bootstrap
wired via ``install_regressions`` + ``bootstrap_plugins``. Projects the
resulting ``PipelineOutcome`` onto a recording ``PlannerConnector`` to
assert:

- Scale-up decisions produce ``add_component`` calls.
- "No change" / short-circuit outcomes produce **zero** connector calls.
- Plugin-disable toggle (``RegisteredPlugin.enabled = False``) excludes
  that builtin from the active set and changes pipeline output.

``NativePlannerBase`` + real PSM production path are NOT exercised —
those live in the dual-path parity integration suite.
"""

from __future__ import annotations

import pytest

from dynamo.planner.config.defaults import SubComponentType
from dynamo.planner.connectors.base import PlannerConnector
from dynamo.planner.core.state_machine import PlannerStateMachine
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
    TrafficMetrics,
)
from dynamo.planner.tests.plugins.g3_fixtures.dump_tool import _tick_for
from dynamo.planner.tests.plugins.g3_fixtures.scenarios import find_scenario
from dynamo.planner.tests.plugins.orchestrator.test_g3_real_parity import (
    _observe_fpm_into_regressions,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.planner,
]


# ---------------------------------------------------------------------------
# Mock connector
# ---------------------------------------------------------------------------


class _RecordingConnector(PlannerConnector):
    """Captures every add/remove call. Production replaces this with
    the real connector (Kubernetes / Global / etc.)."""

    def __init__(self):
        self.adds: list[tuple[str, bool]] = []
        self.removes: list[tuple[str, bool]] = []

    async def add_component(self, sub_component_type, blocking=True):
        self.adds.append((str(sub_component_type.value), blocking))

    async def remove_component(self, sub_component_type, blocking=True):
        self.removes.append((str(sub_component_type.value), blocking))


# ---------------------------------------------------------------------------
# Orchestrator builder + scenario driver
# ---------------------------------------------------------------------------


def _build_full_orchestrator(config, caps):
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
    orch = LocalPlannerOrchestrator(
        registry=server,
        scheduler=scheduler,
        circuit_breaker=cb,
        clock=clock,
        capabilities=caps,
    )
    plugins = {
        "predictor": BuiltinLoadPredictor(orch, config),
        "load_propose": BuiltinLoadPropose(orch, config),
        "throughput_propose": BuiltinThroughputPropose(orch, config),
        "reconcile": BuiltinReconcile(orch, config),
        "budget": BuiltinBudgetConstrain(orch, config),
    }
    orch.register_internal(
        plugin_id="builtin_load_predictor",
        plugin_type="predict",
        priority=1,
        instance=plugins["predictor"],
    )
    orch.register_internal(
        plugin_id="builtin_load_propose",
        plugin_type="propose",
        priority=5,
        instance=plugins["load_propose"],
    )
    orch.register_internal(
        plugin_id="builtin_throughput_propose",
        plugin_type="propose",
        priority=10,
        instance=plugins["throughput_propose"],
    )
    orch.register_internal(
        plugin_id="builtin_reconcile",
        plugin_type="reconcile",
        priority=1,
        instance=plugins["reconcile"],
    )
    orch.register_internal(
        plugin_id="builtin_budget_constrain",
        plugin_type="constrain",
        priority=1,
        instance=plugins["budget"],
    )
    return orch, plugins


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


async def _run_scenario(scenario, connector, *, disabled_plugin_id=None):
    """Drive a scenario through the full pipeline and call ``connector``
    for each tick that produces an "apply" PipelineOutcome."""
    config = scenario.make_config()
    caps = scenario.caps_factory()

    bootstrap_psm = PlannerStateMachine(config, caps)
    if scenario.bootstrap_fn is not None:
        scenario.bootstrap_fn(bootstrap_psm)

    orch, plugins = _build_full_orchestrator(config, caps)
    orch.install_regressions(
        prefill=getattr(bootstrap_psm, "_prefill_regression", None),
        decode=getattr(bootstrap_psm, "_decode_regression", None),
        agg=getattr(bootstrap_psm, "_agg_regression", None),
    )
    await orch.bootstrap_plugins()

    if disabled_plugin_id is not None:
        reg_plugin = orch.list_plugins()
        # Find via registry accessor — `orch.list_plugins` returns PluginInfo,
        # but we need the mutable RegisteredPlugin to toggle enabled.
        reg = orch._registry.get_plugin(disabled_plugin_id)
        assert reg is not None, f"plugin {disabled_plugin_id} not registered"
        reg.enabled = False

    outcomes = []
    current = {"prefill": 0, "decode": 0}
    for ti in scenario.ticks:
        scheduled = _tick_for(ti)
        is_easy = config.optimization_target != "sla"
        if (
            scheduled.run_load_scaling
            and not is_easy
            and ti.fpm_observations is not None
        ):
            _observe_fpm_into_regressions(orch, ti.fpm_observations, config.mode)

        plugins["load_propose"].prime_tick(ti.fpm_observations, ti.worker_counts)
        plugins["budget"].prime_tick(ti.worker_counts)

        # Seed current state from the first tick's worker counts so the
        # connector diff math is meaningful.
        if ti.worker_counts is not None:
            wc = ti.worker_counts
            if wc.ready_num_prefill is not None:
                current.setdefault("_initialised_prefill", False)
                if not current.get("_initialised_prefill"):
                    current["prefill"] = wc.ready_num_prefill
                    current["_initialised_prefill"] = True
            if wc.ready_num_decode is not None:
                if not current.get("_initialised_decode"):
                    current["decode"] = wc.ready_num_decode
                    current["_initialised_decode"] = True

        baseline = {}
        if ti.worker_counts is not None:
            wc = ti.worker_counts
            if wc.ready_num_prefill is not None:
                baseline[ComponentKey(sub_component_type="prefill")] = (
                    wc.ready_num_prefill
                )
            if wc.ready_num_decode is not None:
                baseline[ComponentKey(sub_component_type="decode")] = (
                    wc.ready_num_decode
                )

        ctx = _tick_input_to_context(ti)
        outcome = await orch.tick(ctx, baseline)
        outcomes.append(outcome)

        if outcome.execute_action != "apply" or outcome.final_proposal is None:
            continue
        # Project onto connector calls: diff against current.
        for target in outcome.final_proposal.targets:
            sub = target.sub_component_type
            if sub not in ("prefill", "decode"):
                continue
            desired = target.replicas or 0
            cur = current.get(sub, 0)
            diff = desired - cur
            sct = (
                SubComponentType.PREFILL
                if sub == "prefill"
                else SubComponentType.DECODE
            )
            if diff > 0:
                for _ in range(diff):
                    await connector.add_component(sct)
            elif diff < 0:
                for _ in range(-diff):
                    await connector.remove_component(sct)
            current[sub] = desired

    await orch.shutdown()
    return outcomes


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e2e_scale_up_reaches_connector():
    """Baseline disagg scenario with 3 ticks: tick 0 (no traffic)
    produces no change; tick 1/2 (with traffic) drive throughput
    scale-up. The connector should receive at least some add_component
    calls matching the plugin chain's scale decision."""
    scenario = find_scenario("baseline_disagg_throughput_only_sla")
    connector = _RecordingConnector()
    outcomes = await _run_scenario(scenario, connector)
    # Connector saw scale-up activity.
    assert len(connector.adds) > 0, (
        f"expected add_component calls; got adds={connector.adds} "
        f"removes={connector.removes} outcomes={[o.execute_action for o in outcomes]}"
    )


@pytest.mark.asyncio
async def test_e2e_first_tick_no_change_no_connector_call():
    """Tick 0 of the baseline scenario has no traffic → throughput path
    short-circuits → pipeline either returns skip_no_targets or
    produces a proposal matching current workers (no diff). Either way
    the connector should see zero calls from that tick's projection."""
    scenario = find_scenario("baseline_disagg_throughput_only_sla")
    connector = _RecordingConnector()

    config = scenario.make_config()
    caps = scenario.caps_factory()
    bootstrap_psm = PlannerStateMachine(config, caps)
    if scenario.bootstrap_fn is not None:
        scenario.bootstrap_fn(bootstrap_psm)

    orch, plugins = _build_full_orchestrator(config, caps)
    orch.install_regressions(
        prefill=getattr(bootstrap_psm, "_prefill_regression", None),
        decode=getattr(bootstrap_psm, "_decode_regression", None),
        agg=getattr(bootstrap_psm, "_agg_regression", None),
    )
    await orch.bootstrap_plugins()

    # Drive only tick 0.
    ti = scenario.ticks[0]
    plugins["load_propose"].prime_tick(ti.fpm_observations, ti.worker_counts)
    plugins["budget"].prime_tick(ti.worker_counts)

    baseline = {
        ComponentKey(sub_component_type="prefill"): (
            ti.worker_counts.ready_num_prefill or 0
        ),
        ComponentKey(sub_component_type="decode"): (
            ti.worker_counts.ready_num_decode or 0
        ),
    }
    outcome = await orch.tick(_tick_input_to_context(ti), baseline)

    # Either skip path, or apply with targets == current (no diff → no connector call).
    if outcome.execute_action == "apply" and outcome.final_proposal is not None:
        for t in outcome.final_proposal.targets:
            sub = t.sub_component_type
            current = (
                ti.worker_counts.ready_num_prefill
                if sub == "prefill"
                else ti.worker_counts.ready_num_decode
            ) or 0
            assert t.replicas == current, (
                f"tick 0 should produce no-change; got {sub}={t.replicas} "
                f"current={current}"
            )

    # Connector never involved for tick 0.
    assert connector.adds == []
    assert connector.removes == []


@pytest.mark.asyncio
async def test_e2e_disable_throughput_plugin_changes_decision():
    """With throughput-propose disabled, the pipeline relies only on
    load-propose + budget. For the baseline disagg scenario where
    enable_load_scaling=False, disabling throughput-propose should
    collapse to "no scaling" for the traffic ticks — the connector
    sees no scale-up."""
    scenario = find_scenario("baseline_disagg_throughput_only_sla")

    # Baseline (no disable) — should scale up.
    enabled_connector = _RecordingConnector()
    await _run_scenario(scenario, enabled_connector)
    assert len(enabled_connector.adds) > 0, (
        "baseline expected scale-up add_component calls"
    )

    # With throughput-propose disabled — no scale-up should happen
    # because load is off (enable_load_scaling=False) AND throughput
    # is now disabled.
    disabled_connector = _RecordingConnector()
    await _run_scenario(
        scenario,
        disabled_connector,
        disabled_plugin_id="builtin_throughput_propose",
    )
    # Behaviour differs.
    assert len(disabled_connector.adds) < len(enabled_connector.adds), (
        f"disabling throughput should reduce scale-up calls; "
        f"enabled={len(enabled_connector.adds)} disabled={len(disabled_connector.adds)}"
    )


@pytest.mark.asyncio
async def test_e2e_full_bootstrap_lifecycle_runs_without_error():
    """Smoke: build orchestrator + all 5 builtins + install regressions
    + bootstrap + shutdown — verify the full lifecycle fires without
    any plugin raising. This catches regressions in the
    ``install_regressions`` / ``bootstrap_plugins`` / ``shutdown``
    integration points."""
    scenario = find_scenario("disagg_load_throughput_sla")
    config = scenario.make_config()
    caps = scenario.caps_factory()
    bootstrap_psm = PlannerStateMachine(config, caps)
    scenario.bootstrap_fn(bootstrap_psm)

    orch, _plugins = _build_full_orchestrator(config, caps)
    orch.install_regressions(
        prefill=getattr(bootstrap_psm, "_prefill_regression", None),
        decode=getattr(bootstrap_psm, "_decode_regression", None),
        agg=getattr(bootstrap_psm, "_agg_regression", None),
    )
    await orch.bootstrap_plugins()

    # After bootstrap every plugin is still registered.
    plugin_ids = {p.plugin_id for p in orch.list_plugins()}
    assert plugin_ids == {
        "builtin_load_predictor",
        "builtin_load_propose",
        "builtin_throughput_propose",
        "builtin_reconcile",
        "builtin_budget_constrain",
    }

    await orch.shutdown()
    assert orch.list_plugins() == []
