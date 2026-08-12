# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration-style coverage for one native planner tick."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dynamo.planner.config.defaults import SubComponentType
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.adapters import AggPlanner
from dynamo.planner.core.types import PlannerEffects, ScalingDecision, ScheduledTick
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
            assert tick_input.worker_counts.ready_num_decode == 2
            return PlannerEffects(
                scale_to=ScalingDecision(num_decode=3),
                next_tick=next_tick,
            )

    class RecordingAggPlanner(AggPlanner):
        async def _apply_effects(self, effects):
            events.append("apply effects")
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
