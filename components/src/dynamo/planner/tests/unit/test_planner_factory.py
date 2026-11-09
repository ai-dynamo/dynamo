# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.adapters import AggPlanner
from dynamo.planner.core.planner_factory import (
    construct_connector,
    construct_environment,
)
from dynamo.planner.core.types import (
    BatchDrainLimitDecision,
    PlannerEffects,
    ScalingDecision,
    ScheduledTick,
    TickInput,
    WorkerCounts,
)
from dynamo.planner.environment.batch_runtime import NativeBatchSchedulingProvider
from dynamo.planner.environment.runtime import RuntimeNamespaceBinding
from dynamo.planner.monitoring.worker_info import WorkerInfo

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _config(**overrides) -> PlannerConfig:
    values = {
        "namespace": "base-ns",
        "backend": "vllm",
        "mode": "disagg",
        "environment": "kubernetes",
    }
    values.update(overrides)
    return PlannerConfig.model_construct(**values)


def test_construct_environment_shares_runtime_namespace_binding():
    connector = MagicMock()
    runtime = MagicMock()
    with (
        patch(
            "dynamo.planner.core.planner_factory.construct_connector",
            return_value=connector,
        ),
        patch(
            "dynamo.planner.core.planner_factory.RuntimeFpmProvider"
        ) as fpm_provider_class,
        patch(
            "dynamo.planner.core.planner_factory.PrometheusTrafficProvider"
        ) as traffic_provider_class,
    ):
        environment = construct_environment(
            config=_config(),
            runtime=runtime,
            require_prefill=True,
            require_decode=True,
        )

    namespace_source = traffic_provider_class.call_args.kwargs["namespace_source"]
    assert namespace_source is environment.runtime_namespace_source
    assert namespace_source is fpm_provider_class.call_args.kwargs["namespace_source"]
    assert namespace_source.runtime_namespace() == "base-ns"


@pytest.mark.asyncio
async def test_runtime_namespace_binding_resolves_without_distributed_runtime():
    resolver = MagicMock()
    resolver.get_worker_runtime_namespace.return_value = "base-ns-workerhash"
    namespace_binding = RuntimeNamespaceBinding(
        namespace="base-ns",
        resolver=resolver,
    )

    assert namespace_binding.runtime_namespace() == "base-ns"
    assert await namespace_binding.refresh_runtime_namespace() is True
    assert namespace_binding.runtime_namespace() == "base-ns-workerhash"
    resolver.get_worker_runtime_namespace.assert_called_once_with("base-ns")


def test_construct_environment_binds_namespace_without_runtime():
    connector = MagicMock()
    with (
        patch(
            "dynamo.planner.core.planner_factory.construct_connector",
            return_value=connector,
        ),
        patch(
            "dynamo.planner.core.planner_factory.RuntimeFpmProvider"
        ) as fpm_provider_class,
        patch(
            "dynamo.planner.core.planner_factory.PrometheusTrafficProvider"
        ) as traffic_provider_class,
    ):
        environment = construct_environment(
            config=_config(),
            runtime=None,
            require_prefill=True,
            require_decode=True,
        )

    namespace_source = traffic_provider_class.call_args.kwargs["namespace_source"]
    assert namespace_source is environment.runtime_namespace_source
    assert namespace_source.runtime_namespace() == "base-ns"
    fpm_provider_class.assert_not_called()


@pytest.mark.asyncio
async def test_enabled_batch_config_reaches_redis_actuator_on_native_tick():
    config = PlannerConfig.model_validate(
        {
            "namespace": "base-ns",
            "environment": "kubernetes",
            "mode": "agg",
            "metric_reporting_prometheus_port": 0,
            "live_dashboard_port": 0,
            "report_interval_hours": None,
            "batch_scheduling": {
                "enabled": True,
                "gateway": {
                    "base_url": "http://batch-gateway:8000",
                    "tenant": "planner-poc",
                },
                "metrics": {
                    "frontend_metrics_url": "http://frontend:8000/metrics",
                    "dispatcher_metrics_url": "http://llm-d-async:9090/metrics",
                    "online_match_labels": {
                        "model": "test-model",
                        "request_type": "stream",
                    },
                },
                "redis": {
                    "url": "redis://batch-gateway-valkey:6379/0",
                    "control_key": "llm-d-async:drain-limit:dynamo-batch",
                },
                "pool": {
                    "pool_id": "dynamo-batch",
                    "work_class": "gsm8k-128",
                    "safe_rps_per_ready_replica": 10.0,
                    "drain_lease_duration_seconds": 120.0,
                    "max_replicas": 8,
                },
            },
        }
    )
    connector = MagicMock()
    connector.set_component_replicas = AsyncMock()
    with (
        patch(
            "dynamo.planner.core.planner_factory.construct_connector",
            return_value=connector,
        ),
        patch("dynamo.planner.core.planner_factory.PrometheusTrafficProvider"),
    ):
        environment = construct_environment(
            config=config,
            runtime=None,
            require_prefill=False,
            require_decode=True,
        )

    provider = environment.batch_provider
    assert isinstance(provider, NativeBatchSchedulingProvider)
    assert provider._actuation_enabled is True
    actuator = MagicMock()
    actuator.apply_drain_limit = AsyncMock()
    provider._actuator = actuator
    provider._initialized = True
    environment.deployment_state().decode.info = WorkerInfo(k8s_name="decode-worker")

    drain = BatchDrainLimitDecision(
        pool_id="dynamo-batch",
        max_admission_rps=4.0,
        valid_until_s=100.0,
        decision_id="decision",
    )
    engine = MagicMock()
    engine.tick = AsyncMock(
        return_value=PlannerEffects(
            scale_to=ScalingDecision(num_decode=2),
            next_tick=ScheduledTick(at_s=20.0),
            batch_drain_limits=[drain],
        )
    )
    with patch(
        "dynamo.planner.core.base.PlannerPrometheusMetrics",
        return_value=MagicMock(),
    ):
        planner = AggPlanner(None, config, environment)
    planner._refresh_and_update_capabilities = AsyncMock()
    planner._observe_tick = AsyncMock(
        return_value=TickInput(
            now_s=10.0,
            worker_counts=WorkerCounts(ready_num_decode=1),
        )
    )

    await planner._run_one_tick(engine, ScheduledTick(at_s=10.0))

    actuator.apply_drain_limit.assert_awaited_once_with(drain)
    connector.set_component_replicas.assert_awaited_once()


@pytest.mark.parametrize("global_planner_namespace", [None, ""])
def test_global_planner_namespace_uses_explicit_runtime_validation(
    global_planner_namespace,
):
    config = _config(
        environment="global-planner",
        global_planner_namespace=global_planner_namespace,
    )

    with pytest.raises(ValueError, match="global_planner_namespace is required"):
        construct_connector(config, runtime=MagicMock())
