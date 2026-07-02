# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dynamo.planner.core.types import FpmObservations, ScheduledTick, TrafficObservation
from dynamo.planner.environment.state import DeploymentState
from dynamo.planner.plugins.builtins.observe import (
    EnvironmentObservePlugin,
    ObserveStageRequest,
)


class _FakeEnvironment:
    def __init__(self) -> None:
        self.state = DeploymentState()
        self.full_traffic_calls = 0
        self.kv_hit_calls: list[float] = []
        self.fpm_calls = 0

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
