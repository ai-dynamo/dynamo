# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration-style coverage for one native planner tick."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dynamo.planner.config.defaults import SubComponentType
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.adapters import AggPlanner
from dynamo.planner.core.types import (
    BatchDrainLimitDecision,
    PlannerEffects,
    ScalingDecision,
    ScheduledTick,
    TickInput,
)
from dynamo.planner.environment.state import DeploymentState
from dynamo.planner.monitoring.traffic_metrics import Metrics
from dynamo.planner.monitoring.worker_info import WorkerInfo
from dynamo.planner.plugins.builtins.observe import (
    EnvironmentObservePlugin,
    ObserveStageRequest,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


@pytest.mark.asyncio
@pytest.mark.parametrize("advisory", [False, True])
async def test_complete_tick_applies_scaling_only_when_not_advisory(advisory):
    events = []
    state = DeploymentState()
    state.decode.info = WorkerInfo(k8s_name="decode-worker")
    state.decode.replicas.active = 2

    environment = MagicMock()
    environment.deployment_state.return_value = state
    environment.metrics_state.return_value = Metrics()

    async def refresh():
        events.append("environment.refresh")
        return state

    applied_targets = []

    async def apply_scaling(targets, blocking=False):
        del blocking
        events.append("environment.apply_scaling")
        applied_targets.extend(targets)

    environment.refresh = AsyncMock(side_effect=refresh)
    environment.apply_scaling = AsyncMock(side_effect=apply_scaling)

    observer = EnvironmentObservePlugin(
        environment,
        require_prefill=False,
        require_decode=True,
    )
    next_tick = ScheduledTick(at_s=20.0)

    class RecordingEngine:
        async def observe(self, scheduled_tick, now_s):
            events.append("observe")
            response = await observer.Observe(
                ObserveStageRequest(
                    scheduled_tick=scheduled_tick,
                    now_s=now_s,
                )
            )
            return response.tick_input

        async def tick(self, scheduled_tick, tick_input):
            del scheduled_tick
            events.append("engine.tick")
            assert planner._config_lock.locked()
            assert tick_input.worker_counts.ready_num_decode == 2
            return PlannerEffects(
                scale_to=ScalingDecision(num_decode=3),
                next_tick=next_tick,
            )

    class RecordingAggPlanner(AggPlanner):
        async def _apply_effects(self, effects):
            events.append("apply effects")
            assert not self._config_lock.locked()
            await super()._apply_effects(effects)

    config = PlannerConfig(
        mode="agg",
        advisory=advisory,
        namespace="test-namespace",
        metric_reporting_prometheus_port=0,
        live_dashboard_port=0,
        report_interval_hours=None,
    )
    with patch(
        "dynamo.planner.core.base.PlannerPrometheusMetrics",
        return_value=MagicMock(),
    ):
        planner = RecordingAggPlanner(None, config, environment)

    engine = RecordingEngine()
    planner._engine = engine
    result = await planner._run_one_tick(
        engine,
        ScheduledTick(at_s=10.0, need_worker_states=True),
    )

    assert result is next_tick
    expected_events = [
        "environment.refresh",
        "observe",
        "engine.tick",
        "apply effects",
    ]
    if not advisory:
        expected_events.append("environment.apply_scaling")
        assert len(applied_targets) == 1
        assert applied_targets[0].sub_component_type == SubComponentType.DECODE
        assert applied_targets[0].component_name == "decode-worker"
        assert applied_targets[0].desired_replicas == 3
    else:
        assert applied_targets == []
    assert events == expected_events


@pytest.mark.asyncio
@pytest.mark.parametrize("advisory", [False, True])
async def test_batch_drain_is_applied_before_scaling_and_respects_advisory(advisory):
    events = []
    state = DeploymentState()
    state.decode.info = WorkerInfo(k8s_name="decode-worker")
    state.decode.replicas.active = 1

    environment = MagicMock()
    environment.deployment_state.return_value = state
    environment.metrics_state.return_value = Metrics()
    environment.refresh = AsyncMock(return_value=state)
    environment.apply_batch_drain_limits = AsyncMock(
        side_effect=lambda _decisions: events.append("drain")
    )
    environment.apply_scaling = AsyncMock(
        side_effect=lambda _targets, blocking=False: events.append("scale")
    )

    next_tick = ScheduledTick(at_s=20.0)
    drain = BatchDrainLimitDecision(
        pool_id="pool",
        max_admission_rps=5.0,
        valid_until_s=100.0,
        decision_id="decision",
    )

    class Engine:
        async def tick(self, _scheduled_tick, _tick_input):
            return PlannerEffects(
                scale_to=ScalingDecision(num_decode=2),
                next_tick=next_tick,
                batch_drain_limits=[drain],
            )

    # Keep this test focused on effect ordering; observation composition is
    # covered by the EnvironmentObservePlugin tests.
    engine = Engine()

    async def observe(_scheduled_tick, now_s):
        return TickInput(
            now_s=now_s,
            worker_counts=EnvironmentObservePlugin(
                environment,
                require_prefill=False,
                require_decode=True,
            )._collect_worker_counts(),
        )

    engine.observe = observe

    config = PlannerConfig(
        mode="agg",
        advisory=advisory,
        namespace="test-namespace",
        metric_reporting_prometheus_port=0,
        live_dashboard_port=0,
        report_interval_hours=None,
    )
    with patch(
        "dynamo.planner.core.base.PlannerPrometheusMetrics",
        return_value=MagicMock(),
    ):
        planner = AggPlanner(None, config, environment)

    await planner._run_one_tick(
        engine,
        ScheduledTick(at_s=10.0, need_worker_states=True),
    )

    if advisory:
        assert events == []
    else:
        assert events == ["drain", "scale"]


@pytest.mark.asyncio
async def test_batch_actuation_failure_skips_scaling_without_stopping_tick():
    state = DeploymentState()
    state.decode.info = WorkerInfo(k8s_name="decode-worker")
    state.decode.replicas.active = 1
    environment = MagicMock()
    environment.deployment_state.return_value = state
    environment.metrics_state.return_value = Metrics()
    environment.refresh = AsyncMock(return_value=state)
    environment.apply_batch_drain_limits = AsyncMock(
        side_effect=RuntimeError("redis unavailable")
    )
    environment.apply_scaling = AsyncMock()

    next_tick = ScheduledTick(at_s=20.0)

    class Engine:
        async def observe(self, _scheduled_tick, now_s):
            return TickInput(
                now_s=now_s,
                worker_counts=EnvironmentObservePlugin(
                    environment,
                    require_prefill=False,
                    require_decode=True,
                )._collect_worker_counts(),
            )

        async def tick(self, _scheduled_tick, _tick_input):
            return PlannerEffects(
                scale_to=ScalingDecision(num_decode=2),
                next_tick=next_tick,
                batch_drain_limits=[
                    BatchDrainLimitDecision(
                        pool_id="pool",
                        max_admission_rps=5.0,
                        valid_until_s=100.0,
                        decision_id="decision",
                    )
                ],
            )

    config = PlannerConfig(
        mode="agg",
        namespace="test-namespace",
        metric_reporting_prometheus_port=0,
        live_dashboard_port=0,
        report_interval_hours=None,
    )
    with patch(
        "dynamo.planner.core.base.PlannerPrometheusMetrics",
        return_value=MagicMock(),
    ):
        planner = AggPlanner(None, config, environment)

    engine = Engine()
    result = await planner._run_one_tick(
        engine,
        ScheduledTick(at_s=10.0, need_worker_states=True),
    )

    assert result is next_tick
    environment.apply_scaling.assert_not_awaited()
