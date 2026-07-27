# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Planner connector interface.

A connector is the deployment-control peripheral consumed by
``PlannerEnvironment``.  It owns scaling, deployment validation, worker
capability discovery, and replica-state introspection for one deployment mode.
"""

from __future__ import annotations

from typing import Optional, Protocol

from dynamo.planner.config.defaults import SubComponentType, TargetReplica
from dynamo.planner.monitoring.worker_info import WorkerInfo


class WorkerInfoProvider(Protocol):
    def get_worker_info(
        self,
        sub_component_type: SubComponentType,
        backend: str = "vllm",
    ) -> WorkerInfo:
        pass


class PlannerConnector(WorkerInfoProvider, Protocol):
    """Deployment-control interface the planner uses to inspect and scale one deployment.

    ``construct_connector`` selects one implementation per
    ``PlannerConfig.environment``: ``KubernetesConnector`` patches DGD replica
    counts directly, ``VirtualConnector`` publishes decisions through the runtime
    coordinator for the deployment environment to apply, and
    ``GlobalPlannerConnector`` forwards them to a centralized GlobalPlanner.
    ``PlannerEnvironmentImpl.initialize`` drives ``async_init``, then
    ``validate_deployment``, then ``wait_for_deployment_ready``; ``async_init``
    has to run first, because ``GlobalPlannerConnector.set_component_replicas``
    raises ``RuntimeError`` until it holds a remote client.

    A clean return is not proof of the outcome. ``validate_deployment`` inspects
    the deployment only under Kubernetes and is a no-op in the other two modes.
    ``get_gpu_counts`` yields ``(None, None)`` when GPU shape is unavailable,
    which is always the case for virtual deployments, and ``get_model_name`` can
    return the placeholder ``"managed-remotely"`` under a global planner.
    ``set_component_replicas`` may log and return without scaling when the
    deployment is not ready or the global planner rejects the request, though all
    three implementations do raise ``EmptyTargetReplicasError`` on an empty target
    list. ``get_actual_worker_counts`` reports ``0`` for a component whose name
    argument is ``None`` rather than a deployment-wide total.

    Being a ``Protocol`` rather than an ABC, every method body here is ``pass``.
    All three implementations subclass it explicitly, so an override that is
    missing or misnamed returns ``None`` at runtime instead of raising. The
    surface callers rely on is also wider than what is declared here:
    ``construct_environment`` feature-detects ``get_worker_runtime_namespace``,
    which all three connectors provide.
    """

    async def async_init(self) -> None:
        pass

    async def validate_deployment(
        self,
        prefill_component_name: Optional[str] = None,
        decode_component_name: Optional[str] = None,
        require_prefill: bool = True,
        require_decode: bool = True,
    ) -> None:
        pass

    async def wait_for_deployment_ready(self, include_planner: bool = True) -> None:
        pass

    def get_model_name(
        self,
        require_prefill: bool = True,
        require_decode: bool = True,
    ) -> str:
        pass

    def get_gpu_counts(
        self,
        require_prefill: bool = True,
        require_decode: bool = True,
    ) -> tuple[Optional[int], Optional[int]]:
        pass

    async def get_actual_worker_counts(
        self,
        prefill_component_name: Optional[str] = None,
        decode_component_name: Optional[str] = None,
    ) -> tuple[int, int, bool]:
        pass

    async def set_component_replicas(
        self, target_replicas: list[TargetReplica], blocking: bool = True
    ) -> None:
        pass


__all__ = ["PlannerConnector", "WorkerInfoProvider"]
