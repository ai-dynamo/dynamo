# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration-style coverage for one native planner tick."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from dynamo.planner.config.defaults import SubComponentType
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.adapters import AggPlanner
from dynamo.planner.core.types import (
    PlannerEffects,
    ScalingDecision,
    ScheduledTick,
    TickInput,
    WorkerCounts,
)
from dynamo.planner.environment.state import DeploymentState
from dynamo.planner.errors import GPUShapeUnavailableError
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
@pytest.mark.timeout(5)
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
@pytest.mark.timeout(5)
async def test_runtime_patch_queued_during_decision_discards_stale_effects():
    state = DeploymentState()
    state.decode.info = WorkerInfo(k8s_name="decode-worker")
    state.decode.num_gpus = 1
    environment = MagicMock()
    environment.deployment_state.return_value = state
    environment.metrics_state.return_value = Metrics()

    submitted_effects = []

    class RecordingAggPlanner(AggPlanner):
        async def _apply_effects(self, effects):
            submitted_effects.append(effects)

    config = PlannerConfig(
        mode="agg",
        namespace="test-namespace",
        max_gpu_budget=8,
        metric_reporting_prometheus_port=0,
        live_dashboard_port=0,
        report_interval_hours=None,
    )
    with patch(
        "dynamo.planner.core.base.PlannerPrometheusMetrics",
        return_value=MagicMock(),
    ):
        planner = RecordingAggPlanner(None, config, environment)

    planner._refresh_and_update_capabilities = AsyncMock()
    planner._observe_tick = AsyncMock(
        return_value=TickInput(
            now_s=10.0,
            worker_counts=WorkerCounts(
                ready_num_decode=2,
                expected_num_decode=2,
            ),
        )
    )
    runtime_patch_tasks = []

    class RacingEngine:
        async def tick(self, scheduled_tick, tick_input):
            del scheduled_tick, tick_input
            patch_task = asyncio.create_task(
                planner.patch_min_endpoints({"max_gpu_budget": 1})
            )
            runtime_patch_tasks.append(patch_task)
            await asyncio.sleep(0)
            assert not patch_task.done()
            return PlannerEffects(
                scale_to=ScalingDecision(num_decode=8),
                next_tick=next_tick,
            )

    next_tick = ScheduledTick(at_s=20.0)
    completed_tick = await planner._run_one_tick(
        RacingEngine(), ScheduledTick(at_s=10.0)
    )

    assert (await runtime_patch_tasks[0])["max_gpu_budget"] == 1
    assert completed_tick is next_tick
    assert submitted_effects == []


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_effect_admission_linearizes_before_runtime_patch():
    state = DeploymentState()
    state.decode.info = WorkerInfo(k8s_name="decode-worker")
    state.decode.num_gpus = 1
    environment = MagicMock()
    environment.deployment_state.return_value = state
    submitted_effects = []

    class RecordingAggPlanner(AggPlanner):
        async def _apply_effects(self, effects):
            submitted_effects.append(effects)

    config = PlannerConfig(
        mode="agg",
        namespace="test-namespace",
        max_gpu_budget=8,
        metric_reporting_prometheus_port=0,
        live_dashboard_port=0,
        report_interval_hours=None,
    )
    with patch(
        "dynamo.planner.core.base.PlannerPrometheusMetrics",
        return_value=MagicMock(),
    ):
        planner = RecordingAggPlanner(None, config, environment)

    effects = PlannerEffects(scale_to=ScalingDecision(num_decode=8))
    await planner._effect_admission_lock.acquire()
    submission_task = asyncio.create_task(planner._submit_effects(effects, 0))
    await asyncio.sleep(0)
    patch_task = asyncio.create_task(planner.patch_min_endpoints({"max_gpu_budget": 1}))
    await asyncio.sleep(0)
    planner._effect_admission_lock.release()

    await submission_task
    patch_response = await patch_task

    assert submitted_effects == [effects]
    assert patch_response["max_gpu_budget"] == 1


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_stalled_effect_ack_keeps_runtime_api_available_without_duplicate():
    state = DeploymentState()
    state.decode.info = WorkerInfo(k8s_name="decode-worker")
    state.decode.num_gpus = 1
    environment = MagicMock()
    environment.deployment_state.return_value = state
    environment.metrics_state.return_value = Metrics()

    submission_started = asyncio.Event()
    acknowledge_submission = asyncio.Event()
    submitted_effects = []

    class StalledAggPlanner(AggPlanner):
        async def _apply_effects(self, effects):
            submitted_effects.append(effects)
            submission_started.set()
            await acknowledge_submission.wait()

    config = PlannerConfig(
        mode="agg",
        namespace="test-namespace",
        max_gpu_budget=8,
        metric_reporting_prometheus_port=0,
        live_dashboard_port=0,
        report_interval_hours=None,
    )
    with patch(
        "dynamo.planner.core.base.PlannerPrometheusMetrics",
        return_value=MagicMock(),
    ):
        planner = StalledAggPlanner(None, config, environment)

    planner._refresh_and_update_capabilities = AsyncMock()
    planner._observe_tick = AsyncMock(
        return_value=TickInput(
            now_s=10.0,
            worker_counts=WorkerCounts(
                ready_num_decode=2,
                expected_num_decode=2,
            ),
        )
    )
    first_next_tick = ScheduledTick(at_s=20.0)
    second_next_tick = ScheduledTick(at_s=30.0)
    engine = MagicMock()
    engine.tick = AsyncMock(
        side_effect=[
            PlannerEffects(
                scale_to=ScalingDecision(num_decode=8),
                next_tick=first_next_tick,
            ),
            PlannerEffects(
                scale_to=ScalingDecision(num_decode=1),
                next_tick=second_next_tick,
            ),
        ]
    )

    with patch("dynamo.planner.core.base._EFFECT_SUBMISSION_WAIT_SECONDS", 0.01):
        completed_first_tick = await planner._run_one_tick(
            engine, ScheduledTick(at_s=10.0)
        )
        await submission_started.wait()
        patch_response = await planner.patch_min_endpoints({"max_gpu_budget": 1})
        completed_second_tick = await planner._run_one_tick(engine, first_next_tick)

    pending_submission = planner._effect_submission_task
    assert pending_submission is not None
    assert completed_first_tick is first_next_tick
    assert completed_second_tick is second_next_tick
    assert patch_response["max_gpu_budget"] == 1
    assert (await planner.get_min_endpoints())["max_gpu_budget"] == 1
    assert len(submitted_effects) == 1

    acknowledge_submission.set()
    await pending_submission
    await asyncio.sleep(0)
    assert planner._effect_submission_task is None
    assert len(submitted_effects) == 1


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_late_submission_error_after_timeout_fails_closed_without_retry(caplog):
    state = DeploymentState()
    state.decode.info = WorkerInfo(k8s_name="decode-worker")
    state.decode.num_gpus = 1
    environment = MagicMock()
    environment.deployment_state.return_value = state
    environment.metrics_state.return_value = Metrics()

    fail_submission = asyncio.Event()
    submission_attempts = []

    class AmbiguousAggPlanner(AggPlanner):
        async def _apply_effects(self, effects):
            submission_attempts.append(effects)
            await fail_submission.wait()
            raise RuntimeError("response lost after request submission")

    config = PlannerConfig(
        mode="agg",
        namespace="test-namespace",
        max_gpu_budget=8,
        metric_reporting_prometheus_port=0,
        live_dashboard_port=0,
        report_interval_hours=None,
    )
    with patch(
        "dynamo.planner.core.base.PlannerPrometheusMetrics",
        return_value=MagicMock(),
    ):
        planner = AmbiguousAggPlanner(None, config, environment)

    planner._refresh_and_update_capabilities = AsyncMock()
    planner._observe_tick = AsyncMock(
        return_value=TickInput(
            now_s=10.0,
            worker_counts=WorkerCounts(
                ready_num_decode=2,
                expected_num_decode=2,
            ),
        )
    )
    first_next_tick = ScheduledTick(at_s=20.0)
    second_next_tick = ScheduledTick(at_s=30.0)
    engine = MagicMock()
    engine.tick = AsyncMock(
        side_effect=[
            PlannerEffects(
                scale_to=ScalingDecision(num_decode=8),
                next_tick=first_next_tick,
            ),
            PlannerEffects(
                scale_to=ScalingDecision(num_decode=1),
                next_tick=second_next_tick,
            ),
        ]
    )

    with patch("dynamo.planner.core.base._EFFECT_SUBMISSION_WAIT_SECONDS", 0.01):
        completed_first_tick = await planner._run_one_tick(
            engine, ScheduledTick(at_s=10.0)
        )

    pending_submission = planner._effect_submission_task
    assert pending_submission is not None
    assert completed_first_tick is first_next_tick
    assert len(submission_attempts) == 1

    fail_submission.set()
    with pytest.raises(RuntimeError, match="response lost"):
        await pending_submission
    await asyncio.sleep(0)

    patch_response = await planner.patch_min_endpoints({"max_gpu_budget": 1})
    assert patch_response["max_gpu_budget"] == 1
    assert (await planner.get_min_endpoints())["max_gpu_budget"] == 1

    completed_second_tick = await planner._run_one_tick(engine, first_next_tick)

    assert completed_second_tick is second_next_tick
    assert len(submission_attempts) == 1
    assert planner._effect_submission_outcome_unknown is not None
    assert "requires a Planner restart" in caplog.text


@pytest.mark.asyncio
async def test_run_retries_authoritative_zero_shape_without_shutting_down():
    environment = MagicMock()
    config = PlannerConfig(
        mode="agg",
        namespace="test-namespace",
        load_adjustment_interval_seconds=5,
        metric_reporting_prometheus_port=0,
        live_dashboard_port=0,
        report_interval_hours=None,
    )
    with patch(
        "dynamo.planner.core.base.PlannerPrometheusMetrics",
        return_value=MagicMock(),
    ):
        planner = AggPlanner(None, config, environment)

    engine = MagicMock()
    current_tick = ScheduledTick(at_s=0.0)
    next_tick = ScheduledTick(at_s=0.0)
    engine.initial_tick.return_value = current_tick
    planner._engine = engine
    planner._run_one_tick = AsyncMock(
        side_effect=[
            GPUShapeUnavailableError(
                "decode", "operator published an authoritative zero-GPU shape"
            ),
            next_tick,
            asyncio.CancelledError(),
        ]
    )
    planner._shutdown_runtime = AsyncMock()

    with (
        patch("dynamo.planner.core.base.time.time", return_value=100.0),
        patch(
            "dynamo.planner.core.base.asyncio.sleep", new_callable=AsyncMock
        ) as sleep,
        pytest.raises(asyncio.CancelledError),
    ):
        await planner.run()

    assert planner._run_one_tick.await_args_list == [
        call(engine, current_tick),
        call(engine, current_tick),
        call(engine, next_tick),
    ]
    sleep.assert_awaited_once_with(0.5)
    planner._shutdown_runtime.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_repeated_gpu_shape_errors_wait_and_cancel_cleanly():
    environment = MagicMock()
    config = PlannerConfig(
        mode="agg",
        namespace="test-namespace",
        load_adjustment_interval_seconds=100,
        metric_reporting_prometheus_port=0,
        live_dashboard_port=0,
        report_interval_hours=None,
    )
    with patch(
        "dynamo.planner.core.base.PlannerPrometheusMetrics",
        return_value=MagicMock(),
    ):
        planner = AggPlanner(None, config, environment)

    engine = MagicMock()
    current_tick = ScheduledTick(at_s=0.0)
    engine.initial_tick.return_value = current_tick
    planner._engine = engine
    planner._run_one_tick = AsyncMock(
        side_effect=[
            GPUShapeUnavailableError("decode", "still stale"),
            GPUShapeUnavailableError("decode", "still stale"),
            asyncio.CancelledError(),
        ]
    )
    planner._shutdown_runtime = AsyncMock()

    with (
        patch("dynamo.planner.core.base.time.time", return_value=100.0),
        patch(
            "dynamo.planner.core.base.asyncio.sleep", new_callable=AsyncMock
        ) as sleep,
        pytest.raises(asyncio.CancelledError),
    ):
        await planner.run()

    assert sleep.await_args_list == [call(5.0), call(5.0)]
    planner._shutdown_runtime.assert_awaited_once()
