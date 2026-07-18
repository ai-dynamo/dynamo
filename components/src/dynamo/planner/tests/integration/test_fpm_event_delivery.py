# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from contextlib import AsyncExitStack

import pytest

from dynamo.llm import (
    FpmDirectPublisher,
    ModelInput,
    ModelType,
    WorkerType,
    register_model,
)
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.base import NativePlannerBase
from dynamo.planner.core.state_machine import PlannerScalingState
from dynamo.planner.core.types import (
    EngineCapabilities,
    ScheduledTick,
    WorkerCapabilities,
    WorkerCounts,
)
from dynamo.planner.environment.metrics_provider.runtime_provider import (
    RuntimeFpmProvider,
)
from dynamo.planner.environment.state import DeploymentState
from dynamo.planner.monitoring.worker_info import WorkerInfo
from dynamo.runtime import DistributedRuntime

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.planner,
    pytest.mark.forked,
]


class _PlannerSources:
    def __init__(self, namespace: str, state: DeploymentState) -> None:
        self._namespace = namespace
        self._state = state

    def deployment_state(self) -> DeploymentState:
        return self._state

    def runtime_namespace(self) -> str:
        return self._namespace


class _PlannerEnvironment(_PlannerSources):
    def __init__(self, namespace, state, provider):
        super().__init__(namespace, state)
        self._provider = provider

    def collect_fpm(self):
        return self._provider.collect_fpm()


class _DecodePlanner(NativePlannerBase):
    require_decode = True


@pytest.mark.asyncio
async def test_endpoint_fpm_reaches_planner_and_drives_scale_up(tmp_path, monkeypatch):
    monkeypatch.setenv("DYN_FILE_KV", str(tmp_path))
    namespace = "fpm-test"
    endpoint_path = f"{namespace}.worker.generate"
    async with AsyncExitStack() as stack:
        worker_runtime = DistributedRuntime(
            asyncio.get_running_loop(), "file", "tcp", event_plane="zmq"
        )
        stack.callback(worker_runtime.shutdown)
        planner_runtime = DistributedRuntime(
            asyncio.get_running_loop(), "file", "tcp", event_plane="zmq"
        )
        stack.callback(planner_runtime.shutdown)
        decoy_runtime = DistributedRuntime(
            asyncio.get_running_loop(), "file", "tcp", event_plane="zmq"
        )
        stack.callback(decoy_runtime.shutdown)
        worker_endpoint = worker_runtime.endpoint(endpoint_path)
        worker_id = str(worker_endpoint.connection_id())
        decoy_endpoint = decoy_runtime.endpoint(f"{namespace}.worker.generate-decoy")
        # Deliberately reuse the serving worker ID. If endpoint scoping regresses, identity-based
        # assertions cannot distinguish this decoy from the configured serving pool.
        decoy_publisher = FpmDirectPublisher(decoy_endpoint, worker_id, dp_size=1)
        stack.callback(decoy_publisher.shutdown)
        await register_model(
            ModelInput.Tensor,
            ModelType.TensorBased,
            worker_endpoint,
            "test-model",
            worker_type=WorkerType.Aggregated,
        )
        state = DeploymentState()
        state.decode.info = WorkerInfo(
            component_name="worker",
            endpoint="generate",
            model_name="test-model",
        )
        state.decode.replicas.active = 1
        state.decode.replicas.expected = 1
        sources = _PlannerSources(namespace, state)
        config = PlannerConfig(
            mode="agg",
            enable_load_scaling=True,
            enable_throughput_scaling=False,
            optimization_target="load",
            served_model_name="test-model",
            decode_scale_up_kv_rate=90.0,
            decode_scale_down_kv_rate=50.0,
            min_endpoint=1,
            max_gpu_budget=-1,
            metric_reporting_prometheus_port=0,
        )
        # Establish a decoy-only quiet phase before the serving publisher emits anything. A
        # component-scoped subscription would observe this as the real worker because the IDs are
        # intentionally identical; the endpoint-scoped subscriber must remain empty.
        async with AsyncExitStack() as isolation_stack:
            isolation_provider = RuntimeFpmProvider(
                require_prefill=False,
                require_decode=True,
                backend="vllm",
                model_name="test-model",
                runtime=planner_runtime,
                state_source=sources,
                namespace_source=sources,
            )
            await isolation_provider.async_init()
            isolation_stack.push_async_callback(isolation_provider.shutdown)
            for _ in range(20):
                decoy_publisher.publish(
                    dp_rank=0,
                    scheduled_num_prefill_requests=0,
                    scheduled_sum_prefill_tokens=0,
                    scheduled_sum_prefill_kv_tokens=0,
                    scheduled_num_decode_requests=1,
                    scheduled_sum_decode_kv_tokens=999,
                    queued_num_prefill_requests=0,
                    queued_sum_prefill_tokens=0,
                    queued_num_decode_requests=1,
                    queued_sum_decode_kv_tokens=999,
                    wall_time_secs=0.01,
                )
                await asyncio.sleep(0.05)
                assert not isolation_provider.collect_fpm().decode

        publisher = FpmDirectPublisher(worker_endpoint, worker_id, dp_size=1)
        stack.callback(publisher.shutdown)
        provider = RuntimeFpmProvider(
            require_prefill=False,
            require_decode=True,
            backend="vllm",
            model_name="test-model",
            runtime=planner_runtime,
            state_source=sources,
            namespace_source=sources,
        )
        await provider.async_init()
        stack.push_async_callback(provider.shutdown)
        planner = _DecodePlanner(
            None,
            config,
            _PlannerEnvironment(namespace, state, provider),
        )

        tick_input = None
        for _ in range(100):
            publisher.publish(
                dp_rank=0,
                scheduled_num_prefill_requests=0,
                scheduled_sum_prefill_tokens=0,
                scheduled_sum_prefill_kv_tokens=0,
                scheduled_num_decode_requests=1,
                scheduled_sum_decode_kv_tokens=200,
                queued_num_prefill_requests=0,
                queued_sum_prefill_tokens=0,
                queued_num_decode_requests=1,
                queued_sum_decode_kv_tokens=100,
                wall_time_secs=0.01,
            )
            await asyncio.sleep(0.05)
            tick_input = await planner._gather_tick_input(
                ScheduledTick(
                    at_s=0.0,
                    need_worker_states=True,
                    need_worker_fpm=True,
                    run_load_scaling=True,
                )
            )
            if tick_input.fpm_observations.decode:
                break

        assert tick_input is not None
        observations = tick_input.fpm_observations
        assert observations.decode is not None
        assert (worker_id, 0) in observations.decode

        scaling = PlannerScalingState(
            config,
            WorkerCapabilities(decode=EngineCapabilities(num_gpu=1, max_kv_tokens=100)),
        )
        assert tick_input.worker_counts == WorkerCounts(
            ready_num_decode=1,
            expected_num_decode=1,
        )
        scaling.observe_worker_counts(tick_input.worker_counts)

        decision = scaling.advance_load(observations)

        assert decision is not None
        assert decision.num_decode is not None
        assert decision.num_decode > 1
