# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``OrchestratorEngineAdapter`` hosting of the planner retune controller.

The controller itself is replaced by a recording stub, so these tests cover
only the engine contract: construction, ``start``/step 6/``close`` wiring,
and the quiet ticks of a retune-only planner.
"""

from __future__ import annotations

import logging

import pytest

from dynamo.planner.core.types import (
    RetuneDecision,
    RetuneParams,
    TickInput,
    WorkerCounts,
)
from dynamo.planner.offline.replay_adapter import ReplayPlannerAdapter
from dynamo.planner.plugins.clock import VirtualClock
from dynamo.planner.plugins.orchestrator.engine_adapter import OrchestratorEngineAdapter
from dynamo.planner.tests.retune_stubs import (
    ROUTER_MAP,
    StubTickResult,
    caps,
    install_stub_controller,
    retune_inputs,
    retune_only_config,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


@pytest.fixture
def stub_controller(monkeypatch):
    return install_stub_controller(monkeypatch)


def _engine(**overrides) -> OrchestratorEngineAdapter:
    return OrchestratorEngineAdapter(
        retune_only_config(**overrides),
        caps(),
        clock=VirtualClock(),
        retune_inputs=retune_inputs(),
    )


def _decision() -> RetuneDecision:
    point = RetuneParams(router={"router_temperature": 0.5})
    return RetuneDecision(
        decision_id="retune-0-w0",
        outcome="kept",
        window_index=0,
        window_start_s=0.0,
        window_end_s=60.0,
        as_of_s=120.0,
        launched_at_s=120.0,
        decided_at_s=125.0,
        num_requests=3,
        num_candidates=2,
        previous=point,
        chosen=point,
        baseline_score=1.0,
        best_score=1.0,
    )


def _counts() -> WorkerCounts:
    return WorkerCounts(ready_num_decode=2, expected_num_decode=2)


def test_retune_without_inputs_fails_at_construction(stub_controller):
    with pytest.raises(ValueError, match="requires retune inputs"):
        OrchestratorEngineAdapter(retune_only_config(), caps())

    assert stub_controller.instances == []


def test_controller_receives_config_inputs_capabilities_clock_and_cadence(
    stub_controller,
):
    config = retune_only_config()
    inputs = retune_inputs()
    capabilities = caps()
    clock = VirtualClock()

    engine = OrchestratorEngineAdapter(
        config, capabilities, clock=clock, retune_inputs=inputs
    )

    (controller,) = stub_controller.instances
    assert engine._retune is controller
    assert controller.args == (
        config,
        inputs,
        capabilities,
        clock,
        float(config.scheduling.scale_interval_seconds),
    )
    assert controller.started == []
    engine.initial_tick(0.0)
    assert controller.started == [0.0]


@pytest.mark.asyncio
async def test_tick_runs_retune_after_the_pipeline(stub_controller, monkeypatch):
    engine = _engine()
    controller = stub_controller.instances[0]
    decision = _decision()
    controller.results.append(
        StubTickResult(
            decisions=[decision],
            router_parameters=ROUTER_MAP,
            audit_events=["retune:decided w=0", "retune:launched w=1"],
        )
    )
    original_tick = engine._orchestrator.tick

    async def tick_with_audit(*args, **kwargs):
        outcome = await original_tick(*args, **kwargs)
        outcome.audit_events.append("pipeline:event")
        return outcome

    monkeypatch.setattr(engine._orchestrator, "tick", tick_with_audit)
    scheduled = engine.initial_tick(0.0)

    effects = await engine.tick(
        scheduled, TickInput(now_s=scheduled.at_s, worker_counts=_counts())
    )

    assert controller.ticks == [(scheduled.at_s, _counts())]
    assert effects.router_parameters == ROUTER_MAP
    assert effects.retune_decisions == [decision]
    assert effects.diagnostics.audit_events == [
        "pipeline:event",
        "retune:decided w=0",
        "retune:launched w=1",
    ]
    # Retune never scales: a retune-only planner keeps its replicas.
    assert effects.scale_to is None
    assert effects.next_tick is not None


@pytest.mark.asyncio
async def test_shutdown_continues_when_controller_close_fails(
    stub_controller, monkeypatch, caplog
):
    engine = _engine()
    stub_controller.instances[0].close_error = RuntimeError("worker stuck")
    orchestrator_shutdowns = []

    async def orchestrator_shutdown():
        orchestrator_shutdowns.append(True)

    monkeypatch.setattr(engine._orchestrator, "shutdown", orchestrator_shutdown)

    with caplog.at_level(logging.WARNING):
        await engine.shutdown()

    assert stub_controller.instances[0].closed == 1
    assert orchestrator_shutdowns == [True]
    assert "retune controller close raised RuntimeError" in caplog.text


@pytest.mark.parametrize(
    "ready, overrides",
    [(16, {}), (1, {"min_endpoint": 2})],
    ids=["above-default-max-gpu-budget", "below-min-endpoint"],
)
@pytest.mark.asyncio
async def test_retune_only_planner_never_reconciles_replicas(
    stub_controller, ready, overrides
):
    # 16 one-GPU workers exceed the default max_gpu_budget (8) and one worker
    # is below min_endpoint 2. A scaling planner repairs both; a retune-only
    # planner keeps the deployment's replicas. Its ticks are also quiet:
    # no load loop, no FPM pull (which the SLA target would otherwise make),
    # and nothing that forces the hosts to emit tick diagnostics.
    engine = _engine(optimization_target="sla", **overrides)
    counts = WorkerCounts(ready_num_decode=ready, expected_num_decode=ready)

    scheduled = engine.initial_tick(0.0)
    for _ in range(30):
        assert not scheduled.run_load_scaling
        assert not scheduled.run_throughput_scaling
        assert not scheduled.need_worker_fpm
        assert not scheduled.need_traffic_metrics
        effects = await engine.tick(
            scheduled, TickInput(now_s=scheduled.at_s, worker_counts=counts)
        )
        assert effects.scale_to is None
        assert effects.diagnostics.audit_events == []
        assert not effects.diagnostics.short_circuit_reason
        assert effects.diagnostics.load_decision_reason is None
        assert not ReplayPlannerAdapter._should_emit_tick_diagnostics(
            scheduled, effects
        )
        assert effects.next_tick is not None
        scheduled = effects.next_tick
