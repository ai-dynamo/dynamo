# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Planner connector interface.

A connector is the deployment-control peripheral consumed by
``PlannerEnvironment``.  It owns scaling, deployment validation, worker
capability discovery, and replica-state introspection for one deployment mode.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Optional, Protocol, TypeGuard, runtime_checkable

from dynamo.planner.config.defaults import SubComponentType, TargetReplica
from dynamo.planner.monitoring.worker_info import WorkerInfo

if TYPE_CHECKING:
    from dynamo.planner.monitoring.dgd_services import ComponentPowerConfig


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
    ``PlannerConfig.environment``: ``KubernetesConnector`` scales through the DGD
    scaling adapter's ``Scale`` subresource and falls back to patching DGD replica
    counts directly when no adapter exists, ``VirtualConnector`` publishes
    decisions through the runtime coordinator for the deployment environment to
    apply, and ``GlobalPlannerConnector`` forwards them to a centralized
    GlobalPlanner.
    ``PlannerEnvironmentImpl.initialize`` drives ``async_init``, then
    ``validate_deployment``, then ``wait_for_deployment_ready``; ``async_init``
    has to run first, because ``GlobalPlannerConnector.set_component_replicas``
    raises ``RuntimeError`` until it holds a remote client.

    A clean return is not proof of the outcome. ``validate_deployment`` inspects
    the deployment only under Kubernetes and is a no-op in the other two modes.
    ``get_gpu_counts`` yields ``(None, None)`` from ``VirtualConnector`` always
    and from ``GlobalPlannerConnector`` when it holds no pool-local Kubernetes
    connector, while the Kubernetes implementation narrows the return to
    ``tuple[int, int]`` and raises ``DeploymentValidationError`` rather than
    reporting an unknown shape. ``get_model_name`` can return the placeholder
    ``"managed-remotely"`` under a global planner.
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


@runtime_checkable
class PowerAwareConnector(Protocol):
    """Narrow read-only power capability — Kubernetes-specific, not part of PlannerConnector.

    The environment calls ``is_power_aware_connector(controller)`` at each
    power-aware boundary when ``enable_power_awareness=True``. All four methods
    must be present and callable; a typo or absent method is caught at that check
    rather than silently falling back to a no-op.
    """

    def get_graph_deployment(self) -> dict:
        ...

    def get_component_power_configs(
        self,
        require_prefill: bool = True,
        require_decode: bool = True,
        prefill_component_name: Optional[str] = None,
        decode_component_name: Optional[str] = None,
        deployment: Optional[dict] = None,
    ) -> tuple[Optional[ComponentPowerConfig], Optional[ComponentPowerConfig]]:
        ...

    async def wait_for_settled_graph_deployment(
        self,
        include_planner: bool = False,
        *,
        require_prefill: bool = True,
        require_decode: bool = True,
        prefill_component_name: Optional[str] = None,
        decode_component_name: Optional[str] = None,
    ) -> dict:
        ...

    async def get_power_aware_worker_counts(
        self,
        prefill_component_name: Optional[str] = None,
        decode_component_name: Optional[str] = None,
    ) -> tuple[int, int, bool]:
        ...


_POWER_AWARE_REQUIRED: tuple[str, ...] = (
    "get_graph_deployment",
    "get_component_power_configs",
    "wait_for_settled_graph_deployment",
    "get_power_aware_worker_counts",
)


def is_power_aware_connector(obj: object) -> TypeGuard[PowerAwareConnector]:
    """Return True if ``obj`` is a ``PowerAwareConnector``.

    Uses ``inspect.getattr_static`` (so the check is correct on Python < 3.12,
    where ``isinstance(..., Protocol)`` may return True for objects such as an
    unrestricted ``MagicMock`` that satisfy the protocol only via
    ``__getattr__``). Each attribute must also be callable to exclude
    non-callable class-level sentinels.
    """
    return all(
        callable(inspect.getattr_static(obj, name, None))
        for name in _POWER_AWARE_REQUIRED
    )


__all__ = [
    "PlannerConnector",
    "PowerAwareConnector",
    "WorkerInfoProvider",
    "is_power_aware_connector",
]
