# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Planner environment interfaces.

The planner core consumes a ``PlannerEnvironment`` instead of talking directly
to Kubernetes, the virtual connector, Prometheus, or Dynamo runtime.  Concrete
environments compose a deployment controller with optional metrics providers.
"""

from __future__ import annotations

from typing import Optional, Protocol

from dynamo.planner.config.defaults import TargetReplica
from dynamo.planner.core.types import FpmObservations, TrafficObservation
from dynamo.planner.environment.state import DeploymentState
from dynamo.planner.monitoring.traffic_metrics import Metrics


class DeploymentStateSource(Protocol):
    def deployment_state(self) -> DeploymentState:
        pass


class RuntimeNamespaceSource(Protocol):
    def runtime_namespace(self) -> str:
        pass

    async def refresh_runtime_namespace(self) -> bool:
        """Return True when runtime-backed providers should rebind."""
        pass


class PlannerEnvironment(Protocol):
    async def initialize(self) -> None:
        pass

    async def refresh(self) -> DeploymentState:
        pass

    def deployment_state(self) -> DeploymentState:
        pass

    def runtime_namespace(self) -> str:
        pass

    def metrics_state(self) -> Metrics:
        pass

    async def collect_traffic(self) -> Optional[TrafficObservation]:
        pass

    def collect_accept_length(self, interval_str: str) -> Optional[float]:
        pass

    async def collect_kv_hit_rate_observation(
        self, duration_s: float
    ) -> Optional[TrafficObservation]:
        pass

    def collect_fpm(self) -> FpmObservations:
        pass

    async def apply_scaling(
        self, targets: list[TargetReplica], blocking: bool = False
    ) -> None:
        pass

    async def shutdown(self) -> None:
        pass
