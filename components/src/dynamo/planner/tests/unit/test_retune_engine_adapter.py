# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``OrchestratorEngineAdapter`` hosting of the planner retune controller.

The controller itself is replaced by a recording stub, so these tests cover
only the engine contract: construction, ``start``/step 6/``close`` wiring,
and the quiet ticks of a retune-only planner (R11).
"""

from __future__ import annotations

import logging
import sys
import types
from dataclasses import dataclass, field
from typing import Any

import pytest

from dynamo.planner.config.planner_config import PlannerConfig, RetuneEngineSpec
from dynamo.planner.core.types import (
    EngineCapabilities,
    RetuneDecision,
    RetuneFeedback,
    RetuneInputs,
    RetuneParams,
    TickInput,
    WorkerCapabilities,
    WorkerCounts,
)
from dynamo.planner.offline.replay_adapter import ReplayPlannerAdapter
from dynamo.planner.plugins.clock import VirtualClock
from dynamo.planner.plugins.orchestrator.engine_adapter import OrchestratorEngineAdapter

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]

_CONTROLLER_MODULE = "dynamo.planner.retune.controller"
_ROUTER_MAP = {
    "aggregated": {
        "decode_active_request_weight": 0.0,
        "disk_cache_hit_weight": 0.25,
        "host_cache_hit_weight": 0.75,
        "overlap_score_credit": 1.0,
        "overlap_score_credit_decay": 0.0,
        "prefill_load_scale": 1.0,
        "router_temperature": 0.5,
        "shared_cache_multiplier": 0.0,
    }
}


@dataclass(frozen=True)
class _TickResult:
    decisions: list[RetuneDecision] = field(default_factory=list)
    router_parameters: dict[str, dict[str, float]] | None = None
    audit_events: list[str] = field(default_factory=list)


class _StubController:
    """Records the calls the engine makes; returns queued tick results."""

    instances: list[_StubController] = []

    def __init__(self, config, inputs, capabilities, clock, scale_interval_s):
        self.args = (config, inputs, capabilities, clock, scale_interval_s)
        self.started: list[float] = []
        self.ticks: list[tuple[float, Any, list[RetuneFeedback]]] = []
        self.results: list[_TickResult] = []
        self.closed = 0
        self.close_error: Exception | None = None
        _StubController.instances.append(self)

    def start(self, start_s: float) -> None:
        self.started.append(start_s)

    async def on_tick(self, now_s, worker_counts, feedback):
        self.ticks.append((now_s, worker_counts, list(feedback)))
        if self.results:
            return self.results.pop(0)
        return _TickResult(router_parameters=_ROUTER_MAP)

    def close(self) -> None:
        self.closed += 1
        if self.close_error is not None:
            raise self.close_error


@pytest.fixture
def stub_controller(monkeypatch):
    """Install the stub as ``dynamo.planner.retune.controller``."""

    module = types.ModuleType(_CONTROLLER_MODULE)
    module.RetuneController = _StubController  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _CONTROLLER_MODULE, module)
    _StubController.instances = []
    return _StubController


def _caps() -> WorkerCapabilities:
    return WorkerCapabilities(
        decode=EngineCapabilities(
            num_gpu=1, max_num_batched_tokens=2048, max_kv_tokens=16384
        )
    )


def _retune_only_config(**overrides) -> PlannerConfig:
    return PlannerConfig(
        mode="agg",
        enable_load_scaling=False,
        enable_throughput_scaling=False,
        retune={"interval_seconds": 60, "router": {"router_temperature": [0.0, 0.5]}},
        **overrides,
    )


def _inputs() -> RetuneInputs:
    return RetuneInputs(engine=RetuneEngineSpec(), trace_paths=("trace.jsonl",))


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


def test_controller_is_built_only_when_retune_is_configured(stub_controller):
    engine = OrchestratorEngineAdapter(
        PlannerConfig(mode="agg", optimization_target="throughput"), _caps()
    )

    assert engine._retune is None
    assert stub_controller.instances == []


def test_retune_without_inputs_fails_at_construction(stub_controller):
    with pytest.raises(ValueError, match="requires retune inputs"):
        OrchestratorEngineAdapter(_retune_only_config(), _caps())

    assert stub_controller.instances == []


def test_controller_receives_config_inputs_capabilities_clock_and_cadence(
    stub_controller,
):
    config = _retune_only_config()
    inputs = _inputs()
    capabilities = _caps()
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


def test_initial_tick_starts_the_controller(stub_controller):
    engine = OrchestratorEngineAdapter(
        _retune_only_config(), _caps(), clock=VirtualClock(), retune_inputs=_inputs()
    )

    engine.initial_tick(0.0)

    assert stub_controller.instances[0].started == [0.0]


@pytest.mark.asyncio
async def test_tick_runs_retune_after_the_pipeline(stub_controller):
    engine = OrchestratorEngineAdapter(
        _retune_only_config(), _caps(), clock=VirtualClock(), retune_inputs=_inputs()
    )
    controller = stub_controller.instances[0]
    decision = _decision()
    controller.results.append(
        _TickResult(
            decisions=[decision],
            router_parameters=_ROUTER_MAP,
            audit_events=["retune:decided w=0", "retune:launched w=1"],
        )
    )
    feedback = [RetuneFeedback(decision_id="retune-0-w0", target="router", error="422")]
    scheduled = engine.initial_tick(0.0)

    effects = await engine.tick(
        scheduled,
        TickInput(
            now_s=scheduled.at_s, worker_counts=_counts(), retune_feedback=feedback
        ),
    )

    assert controller.ticks == [(scheduled.at_s, _counts(), feedback)]
    assert effects.router_parameters == _ROUTER_MAP
    assert effects.retune_decisions == [decision]
    assert effects.diagnostics.audit_events == [
        "retune:decided w=0",
        "retune:launched w=1",
    ]
    # Retune never scales: a retune-only planner keeps its replicas.
    assert effects.scale_to is None
    assert effects.next_tick is not None
    # The retune audit strings alone make the replay host emit diagnostics.
    assert ReplayPlannerAdapter._should_emit_tick_diagnostics(scheduled, effects)


@pytest.mark.asyncio
async def test_retune_audit_events_follow_pipeline_audit_events(
    stub_controller, monkeypatch
):
    engine = OrchestratorEngineAdapter(
        _retune_only_config(), _caps(), clock=VirtualClock(), retune_inputs=_inputs()
    )
    controller = stub_controller.instances[0]
    controller.results.append(_TickResult(audit_events=["retune:busy w=3"]))
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

    assert effects.diagnostics.audit_events == ["pipeline:event", "retune:busy w=3"]
    # The empty-router-domain contract (None) passes through unchanged.
    assert effects.router_parameters is None


@pytest.mark.asyncio
async def test_shutdown_closes_the_controller(stub_controller):
    engine = OrchestratorEngineAdapter(
        _retune_only_config(), _caps(), clock=VirtualClock(), retune_inputs=_inputs()
    )

    await engine.shutdown()

    assert stub_controller.instances[0].closed == 1


@pytest.mark.asyncio
async def test_shutdown_continues_when_controller_close_fails(
    stub_controller, monkeypatch, caplog
):
    engine = OrchestratorEngineAdapter(
        _retune_only_config(), _caps(), clock=VirtualClock(), retune_inputs=_inputs()
    )
    stub_controller.instances[0].close_error = RuntimeError("worker stuck")
    orchestrator_shutdowns = []

    async def orchestrator_shutdown():
        orchestrator_shutdowns.append(True)

    monkeypatch.setattr(engine._orchestrator, "shutdown", orchestrator_shutdown)

    with caplog.at_level(logging.WARNING):
        await engine.shutdown()

    assert orchestrator_shutdowns == [True]
    assert "retune controller close raised RuntimeError" in caplog.text


@pytest.mark.parametrize("optimization_target", ["throughput", "sla"])
@pytest.mark.asyncio
async def test_retune_only_ticks_run_no_load_loop_and_pull_no_fpm(
    stub_controller, optimization_target
):
    engine = OrchestratorEngineAdapter(
        _retune_only_config(optimization_target=optimization_target),
        _caps(),
        clock=VirtualClock(),
        retune_inputs=_inputs(),
    )
    stub_controller.instances[0].results.extend(_TickResult() for _ in range(30))

    scheduled = engine.initial_tick(0.0)
    for _ in range(30):
        assert not scheduled.run_load_scaling
        assert not scheduled.run_throughput_scaling
        assert not scheduled.need_worker_fpm
        assert not scheduled.need_traffic_metrics
        effects = await engine.tick(
            scheduled, TickInput(now_s=scheduled.at_s, worker_counts=_counts())
        )
        # No retune audit strings and no load loop: nothing forces the
        # hosts to emit tick diagnostics.
        assert effects.diagnostics.audit_events == []
        assert not effects.diagnostics.short_circuit_reason
        assert effects.diagnostics.load_decision_reason is None
        assert not ReplayPlannerAdapter._should_emit_tick_diagnostics(
            scheduled, effects
        )
        assert effects.scale_to is None
        assert effects.next_tick is not None
        scheduled = effects.next_tick


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
    # planner keeps the deployment's replicas.
    engine = OrchestratorEngineAdapter(
        _retune_only_config(**overrides),
        _caps(),
        clock=VirtualClock(),
        retune_inputs=_inputs(),
    )
    stub_controller.instances[0].results.extend(_TickResult() for _ in range(5))
    counts = WorkerCounts(ready_num_decode=ready, expected_num_decode=ready)

    scheduled = engine.initial_tick(0.0)
    for _ in range(5):
        effects = await engine.tick(
            scheduled, TickInput(now_s=scheduled.at_s, worker_counts=counts)
        )
        assert effects.scale_to is None
        scheduled = effects.next_tick


def test_scaling_planners_keep_their_load_loop_cadence():
    config = PlannerConfig(
        mode="agg",
        optimization_target="sla",
        enable_load_scaling=True,
        enable_throughput_scaling=False,
    )
    engine = OrchestratorEngineAdapter(config, _caps(), clock=VirtualClock())

    scheduled = engine.initial_tick(0.0)

    assert scheduled.run_load_scaling
    assert scheduled.need_worker_fpm
