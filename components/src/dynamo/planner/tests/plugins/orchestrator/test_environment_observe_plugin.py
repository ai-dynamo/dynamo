# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.planner.core.types import (
    BatchSchedulingObservation,
    FpmObservations,
    ScheduledTick,
    TrafficObservation,
)
from dynamo.planner.environment.state import DeploymentState
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


class _FakeEnvironment:
    def __init__(self) -> None:
        self.state = DeploymentState()
        self.full_traffic_calls = 0
        self.kv_hit_calls: list[float] = []
        self.fpm_calls = 0
        self.batch_calls = 0
        self.batch_error: Exception | None = None
        self.replica_refresh_calls = 0
        self.replica_refresh_error: Exception | None = None

    def deployment_state(self) -> DeploymentState:
        return self.state

    async def collect_traffic(self):
        self.full_traffic_calls += 1
        return TrafficObservation(duration_s=60, num_req=10, isl=20, osl=30)

    async def collect_kv_hit_rate_observation(self, duration_s: float):
        self.kv_hit_calls.append(duration_s)
        return TrafficObservation(
            duration_s=duration_s,
            num_req=0,
            isl=0,
            osl=0,
            kv_hit_rate=0.75,
        )

    def collect_fpm(self):
        self.fpm_calls += 1
        return FpmObservations(prefill={}, decode={})

    async def collect_batch_scheduling(self):
        self.batch_calls += 1
        if self.batch_error is not None:
            raise self.batch_error
        return BatchSchedulingObservation()

    async def refresh_replica_state(self):
        self.replica_refresh_calls += 1
        if self.replica_refresh_error is not None:
            raise self.replica_refresh_error
        return self.state


async def test_environment_observe_plugin_collects_requested_observations():
    env = _FakeEnvironment()
    env.state.prefill.replicas.active = 2
    env.state.prefill.replicas.expected = 3
    env.state.prefill.replicas.scaling = True
    env.state.decode.replicas.active = 4
    env.state.decode.replicas.expected = 4
    env.state.decode.replicas.scaling = False

    plugin = EnvironmentObservePlugin(env, require_prefill=True, require_decode=True)
    response = await plugin.Observe(
        ObserveStageRequest(
            scheduled_tick=ScheduledTick(
                at_s=123,
                need_traffic_metrics=True,
                use_full_traffic_metrics=True,
                need_worker_states=True,
                need_worker_fpm=True,
            ),
            now_s=456,
        )
    )

    tick_input = response.tick_input
    assert tick_input.now_s == 456
    assert tick_input.traffic is not None
    assert tick_input.traffic.num_req == 10
    assert tick_input.worker_counts is not None
    assert tick_input.worker_counts.ready_num_prefill == 2
    assert tick_input.worker_counts.expected_num_prefill == 3
    assert tick_input.worker_counts.prefill_scaling_in_progress is True
    assert tick_input.worker_counts.ready_num_decode == 4
    assert tick_input.worker_counts.decode_scaling_in_progress is False
    assert tick_input.fpm_observations is not None
    assert env.full_traffic_calls == 1
    assert env.fpm_calls == 1


async def test_environment_observe_plugin_collects_load_only_traffic_window():
    env = _FakeEnvironment()
    plugin = EnvironmentObservePlugin(env, require_prefill=False, require_decode=True)

    response = await plugin.Observe(
        ObserveStageRequest(
            scheduled_tick=ScheduledTick(
                at_s=123,
                need_traffic_metrics=True,
                use_full_traffic_metrics=False,
                traffic_metrics_duration_s=5,
            ),
            now_s=456,
        )
    )

    assert response.tick_input.traffic is not None
    assert response.tick_input.traffic.kv_hit_rate == 0.75
    assert env.full_traffic_calls == 0
    assert env.kv_hit_calls == [5]
    assert env.fpm_calls == 0


async def test_environment_observe_plugin_collects_batch_and_anchors_after_io():
    env = _FakeEnvironment()
    plugin = EnvironmentObservePlugin(
        env,
        require_prefill=False,
        require_decode=True,
        clock=lambda: 500.0,
    )

    response = await plugin.Observe(
        ObserveStageRequest(
            scheduled_tick=ScheduledTick(
                at_s=123,
                need_batch_scheduling=True,
            ),
            now_s=456,
        )
    )

    assert response.tick_input.now_s == 500.0
    assert response.tick_input.batch == BatchSchedulingObservation()
    assert env.batch_calls == 1


async def test_environment_observe_plugin_snapshots_capacity_after_batch_io():
    class _CapacityChangesDuringBatch(_FakeEnvironment):
        external_active = 8

        async def collect_batch_scheduling(self):
            self.batch_calls += 1
            self.external_active = 1
            return BatchSchedulingObservation()

        async def refresh_replica_state(self):
            self.replica_refresh_calls += 1
            self.state.decode.replicas.active = self.external_active
            self.state.decode.replicas.expected = self.external_active
            return self.state

    env = _CapacityChangesDuringBatch()
    env.state.decode.replicas.active = 8
    env.state.decode.replicas.expected = 8
    plugin = EnvironmentObservePlugin(
        env,
        require_prefill=False,
        require_decode=True,
        clock=lambda: 500.0,
    )

    response = await plugin.Observe(
        ObserveStageRequest(
            scheduled_tick=ScheduledTick(
                at_s=123,
                need_worker_states=True,
                need_batch_scheduling=True,
            ),
            now_s=456,
        )
    )

    assert response.tick_input.worker_counts is not None
    assert response.tick_input.worker_counts.ready_num_decode == 1
    assert response.tick_input.now_s == 500.0
    assert env.replica_refresh_calls == 1


async def test_environment_observe_plugin_replica_refresh_failure_omits_capacity():
    env = _FakeEnvironment()
    env.state.decode.replicas.active = 8
    env.replica_refresh_error = RuntimeError("Kubernetes unavailable")
    plugin = EnvironmentObservePlugin(
        env,
        require_prefill=False,
        require_decode=True,
        clock=lambda: 500.0,
    )

    response = await plugin.Observe(
        ObserveStageRequest(
            scheduled_tick=ScheduledTick(
                at_s=123,
                need_worker_states=True,
                need_batch_scheduling=True,
            ),
            now_s=456,
        )
    )

    assert response.tick_input.batch == BatchSchedulingObservation()
    assert response.tick_input.worker_counts is None
    assert env.replica_refresh_calls == 1


async def test_environment_observe_plugin_batch_failure_is_fail_closed():
    env = _FakeEnvironment()
    env.batch_error = RuntimeError("telemetry unavailable")
    plugin = EnvironmentObservePlugin(
        env,
        require_prefill=False,
        require_decode=True,
        clock=lambda: 500.0,
    )

    response = await plugin.Observe(
        ObserveStageRequest(
            scheduled_tick=ScheduledTick(
                at_s=123,
                need_worker_states=True,
                need_batch_scheduling=True,
            ),
            now_s=456,
        )
    )

    assert response.tick_input.now_s == 500.0
    assert response.tick_input.batch is None
    assert response.tick_input.worker_counts is not None
    assert env.batch_calls == 1
