# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``NativePlannerBase._wire_predicted_load_if_supported``
hook (PR 7 sub-task 7-7).

The hook runs once per tick between ``engine.tick`` and
``_apply_effects``. It's a no-op unless:

1. ``scheduling.use_orchestrator`` is True (PSM path is intentionally
   left alone — it never called ``set_predicted_load`` pre-PR-7, and
   fixing that is out of scope).
2. The connector exposes a ``set_predicted_load`` callable (currently
   only ``GlobalPlannerConnector``).
3. At least one of the prediction fields in ``effects.diagnostics`` is
   populated — the orchestrator adapter fills these from
   ``ChainAugmentOutcome.prediction`` so this gate fires on any tick
   where the PREDICT stage produced output.
"""

from __future__ import annotations

import pytest

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.base import NativePlannerBase
from dynamo.planner.core.types import PlannerEffects, TickDiagnostics

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


class _RecordingConnector:
    """Mimics ``GlobalPlannerConnector.set_predicted_load`` — records
    calls so the test can assert the wire fired with the right args."""

    def __init__(self):
        self.calls: list[dict] = []

    def set_predicted_load(self, *, num_requests, isl, osl):
        self.calls.append(
            {"num_requests": num_requests, "isl": isl, "osl": osl}
        )


class _ConnectorWithoutHook:
    """Mimics ``KubernetesConnector`` / ``VirtualConnector`` — no
    ``set_predicted_load`` method."""

    pass


class _MinimalPlanner(NativePlannerBase):
    def __init__(self, config: PlannerConfig, connector):
        self.config = config
        self.runtime = None
        self.namespace = config.namespace
        self.model_name = None
        self.connector = connector
        self._state_machine = None
        self._engine = None
        self._last_worker_counts = None
        self.prometheus_port = 0
        self.prometheus_metrics = None


def _config(use_orchestrator: bool = True) -> PlannerConfig:
    return PlannerConfig(
        environment="kubernetes",
        mode="agg",
        enable_load_scaling=True,
        enable_throughput_scaling=True,
        optimization_target="sla",
        scheduling={"use_orchestrator": use_orchestrator},
    )


def _effects_with_predictions(nr=100.0, isl=500.0, osl=150.0):
    return PlannerEffects(
        scale_to=None,
        next_tick=None,
        diagnostics=TickDiagnostics(
            predicted_num_req=nr,
            predicted_isl=isl,
            predicted_osl=osl,
        ),
    )


# ---------------------------------------------------------------------------
# Fires when all conditions met
# ---------------------------------------------------------------------------


def test_wires_predicted_load_when_orchestrator_and_connector_supports():
    connector = _RecordingConnector()
    planner = _MinimalPlanner(_config(use_orchestrator=True), connector)
    planner._wire_predicted_load_if_supported(_effects_with_predictions())

    assert len(connector.calls) == 1
    call = connector.calls[0]
    assert call["num_requests"] == 100.0
    assert call["isl"] == 500.0
    assert call["osl"] == 150.0


# ---------------------------------------------------------------------------
# Suppressed when any guard fails
# ---------------------------------------------------------------------------


def test_skipped_on_psm_path_even_if_connector_supports():
    """PSM path historically doesn't call ``set_predicted_load`` —
    keep it that way (fixing the pre-existing gap is out of scope)."""
    connector = _RecordingConnector()
    planner = _MinimalPlanner(_config(use_orchestrator=False), connector)
    planner._wire_predicted_load_if_supported(_effects_with_predictions())
    assert connector.calls == []


def test_skipped_when_connector_lacks_set_predicted_load():
    """KubernetesConnector / VirtualConnector don't expose the method.
    ``hasattr`` + ``callable`` guards mean we silently skip."""
    connector = _ConnectorWithoutHook()
    planner = _MinimalPlanner(_config(use_orchestrator=True), connector)
    # Must not raise AttributeError.
    planner._wire_predicted_load_if_supported(_effects_with_predictions())


def test_skipped_when_all_prediction_fields_none():
    connector = _RecordingConnector()
    planner = _MinimalPlanner(_config(use_orchestrator=True), connector)
    planner._wire_predicted_load_if_supported(
        PlannerEffects(
            scale_to=None,
            next_tick=None,
            diagnostics=TickDiagnostics(),  # all fields None
        )
    )
    assert connector.calls == []


def test_fires_when_only_one_prediction_field_set():
    """A partial prediction (e.g. layered-predictor pattern from PR 4)
    should still wire — GlobalPlanner accepts ``None`` for individual
    fields."""
    connector = _RecordingConnector()
    planner = _MinimalPlanner(_config(use_orchestrator=True), connector)
    planner._wire_predicted_load_if_supported(
        PlannerEffects(
            scale_to=None,
            next_tick=None,
            diagnostics=TickDiagnostics(predicted_num_req=50.0),
        )
    )
    assert len(connector.calls) == 1
    assert connector.calls[0]["num_requests"] == 50.0
    assert connector.calls[0]["isl"] is None
    assert connector.calls[0]["osl"] is None


# ---------------------------------------------------------------------------
# Orchestrator adapter populates diagnostics.predicted_* from prediction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adapter_populates_diagnostics_from_predictions():
    """Integration-level: end-to-end confirms the adapter surfaces
    ``ChainAugmentOutcome.prediction`` into ``TickDiagnostics``, which
    is the pipe the wire in ``run()`` reads from."""
    from dynamo.planner.core.state_machine import PlannerStateMachine
    from dynamo.planner.plugins.orchestrator.engine_adapter import (
        OrchestratorEngineAdapter,
    )
    from dynamo.planner.tests.plugins.g3_fixtures.dump_tool import _tick_for
    from dynamo.planner.tests.plugins.g3_fixtures.scenarios import find_scenario

    scenario = find_scenario("baseline_disagg_throughput_only_sla")
    config = scenario.make_config()
    caps = scenario.caps_factory()
    bootstrap_psm = PlannerStateMachine(config, caps)
    scenario.bootstrap_fn(bootstrap_psm)

    adapter = OrchestratorEngineAdapter(config, caps)
    adapter.install_regressions(
        prefill=bootstrap_psm._prefill_regression,
        decode=bootstrap_psm._decode_regression,
    )
    await adapter.bootstrap_plugins()
    adapter.initial_tick(scenario.initial_tick_at_s)

    # Find a tick with traffic so the predictor produces output.
    traffic_tick = next(t for t in scenario.ticks if t.traffic is not None)
    effects = await adapter.tick(_tick_for(traffic_tick), traffic_tick)

    # At least one prediction field should be populated (ConstantPredictor
    # yields the last observed value after traffic feed).
    has_prediction = (
        effects.diagnostics.predicted_num_req is not None
        or effects.diagnostics.predicted_isl is not None
        or effects.diagnostics.predicted_osl is not None
    )
    assert has_prediction, (
        f"adapter didn't populate prediction fields; got diagnostics={effects.diagnostics}"
    )

    await adapter.shutdown()
