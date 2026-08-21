# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, cast

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.connectors.base import PlannerConnector, WorkerInfoProvider
from dynamo.planner.connectors.global_planner import GlobalPlannerConnector
from dynamo.planner.connectors.kubernetes import KubernetesConnector
from dynamo.planner.connectors.virtual import VirtualConnector
from dynamo.planner.environment.base import PlannerEnvironmentImpl
from dynamo.planner.environment.interface import PlannerEnvironment
from dynamo.planner.environment.metrics_provider.prometheus_traffic_provider import (
    PrometheusTrafficProvider,
)
from dynamo.planner.environment.metrics_provider.runtime_provider import (
    RuntimeFpmProvider,
)
from dynamo.planner.environment.runtime import (
    RuntimeNamespaceBinding,
    RuntimeNamespaceResolver,
)
from dynamo.runtime import DistributedRuntime

if TYPE_CHECKING:
    from dynamo.planner.core.base import NativePlannerBase


def construct_connector(
    config: PlannerConfig,
    runtime: Optional[DistributedRuntime] = None,
    worker_info_provider: Optional[WorkerInfoProvider] = None,
) -> PlannerConnector:
    if config.environment == "global-planner":
        if runtime is None:
            raise ValueError("runtime is required for environment='global-planner'")
        if not config.global_planner_namespace:
            raise ValueError(
                "global_planner_namespace is required for environment='global-planner'"
            )
        return GlobalPlannerConnector(
            runtime=runtime,
            dynamo_namespace=config.namespace,
            global_planner_namespace=config.global_planner_namespace,
            global_planner_component="GlobalPlanner",
            model_name=config.model_name,
        )
    if config.environment == "kubernetes":
        return KubernetesConnector(
            dynamo_namespace=config.namespace,
            model_name=config.model_name,
        )
    if config.environment == "virtual":
        if runtime is None:
            raise ValueError("runtime is required for environment='virtual'")
        if worker_info_provider is None:
            raise ValueError(
                "worker_info_provider is required for environment='virtual'"
            )
        return VirtualConnector(
            runtime=runtime,
            dynamo_namespace=config.namespace,
            worker_info_provider=worker_info_provider,
            model_name=config.model_name,
        )
    raise ValueError(f"Invalid environment: {config.environment}")


def construct_environment(
    *,
    config: PlannerConfig,
    require_prefill: bool,
    require_decode: bool,
    runtime: Optional[DistributedRuntime] = None,
) -> PlannerEnvironment:
    fpm_provider: Optional[RuntimeFpmProvider] = None
    if config.environment == "virtual":
        if runtime is None:
            raise ValueError("runtime is required for environment='virtual'")
        fpm_provider = RuntimeFpmProvider(
            require_prefill=require_prefill,
            require_decode=require_decode,
            backend=config.backend,
            model_name=config.model_name,
            runtime=runtime,
        )
        connector = construct_connector(
            config,
            runtime,
            worker_info_provider=fpm_provider,
        )
    else:
        connector = construct_connector(config, runtime)

    environment = PlannerEnvironmentImpl(
        config=config,
        controller=connector,
        require_prefill=require_prefill,
        require_decode=require_decode,
    )

    namespace_binding: Optional[RuntimeNamespaceBinding] = None
    if callable(getattr(connector, "get_worker_runtime_namespace", None)):
        namespace_binding = RuntimeNamespaceBinding(
            namespace=config.namespace,
            resolver=cast(RuntimeNamespaceResolver, connector),
        )
        environment.runtime_namespace_source = namespace_binding

    if runtime is not None and namespace_binding is not None:
        if fpm_provider is None:
            fpm_provider = RuntimeFpmProvider(
                require_prefill=require_prefill,
                require_decode=require_decode,
                backend=config.backend,
                model_name=config.model_name,
                runtime=runtime,
                state_source=environment,
                namespace_source=namespace_binding,
            )
        else:
            fpm_provider.bind_sources(
                state_source=environment,
                namespace_source=namespace_binding,
            )
        environment.fpm_provider = fpm_provider

    environment.traffic_provider = PrometheusTrafficProvider(
        config=config,
        state_source=environment,
        metrics_state=environment.metrics_state(),
        namespace_source=namespace_binding,
    )

    return environment


def construct_planner(
    *,
    runtime: DistributedRuntime,
    config: PlannerConfig,
) -> NativePlannerBase:
    # Connector/environment-only callers should not import the planner/plugin graph.
    from dynamo.planner.core.adapters import (
        AggPlanner,
        DecodePlanner,
        DisaggPlanner,
        PrefillPlanner,
    )

    planner_cls: type[NativePlannerBase]
    if config.mode == "disagg":
        planner_cls = DisaggPlanner
    elif config.mode == "prefill":
        planner_cls = PrefillPlanner
    elif config.mode == "decode":
        planner_cls = DecodePlanner
    elif config.mode == "agg":
        planner_cls = AggPlanner
    else:
        raise ValueError(f"Invalid planner mode: {config.mode}")

    environment = construct_environment(
        config=config,
        runtime=runtime,
        require_prefill=planner_cls.require_prefill,
        require_decode=planner_cls.require_decode,
    )
    return planner_cls(runtime, config, environment)
