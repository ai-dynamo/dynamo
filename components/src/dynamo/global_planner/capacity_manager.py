# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capacity backend contract for the GlobalPlanner — reads and writes scaling state.

:class:`CapacityManager` is the infrastructure-facing base class the
:class:`~dynamo.global_planner.orchestrator.Orchestrator` drives to **observe**
current pool state and **scale** replicas. It defines a neutral,
infrastructure-agnostic surface (``observe`` / ``scale`` / ``discover`` /
``ensure_participant`` / ``participant_exists`` / ``remember_roles`` /
``current_replicas``) keyed by opaque ``participant_id``; the orchestrator never
sees Kubernetes concepts.

The concrete Kubernetes backend is
:class:`~dynamo.global_planner.kubernetes_capacity_manager.KubernetesCapacityManager`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from dynamo.planner import TargetReplica


@dataclass
class PoolSpec:
    """Snapshot of one pool's state read from a backend.

    Infrastructure-agnostic: the budget math consumes only ``current_replicas``
    and ``gpu_per_replica``. ``component_name`` is an opaque backend handle for
    the pool (empty for backends that don't need one); it round-trips through the
    orchestrator so a paired scale can target the right component. Pools with
    ``gpu_per_replica == 0`` contribute 0 to budget totals.
    """

    sub_type: str
    current_replicas: int
    gpu_per_replica: int
    component_name: str = ""


# participant_id -> (sub_type -> PoolSpec)
PoolSnapshot = dict[str, dict[str, PoolSpec]]


class CapacityManager:
    """Base capacity backend: observe pool state and scale replicas.

    Neutral contract keyed by opaque ``participant_id``. Concrete backends
    override the observe/scale surface; ``discover`` defaults to a no-op for
    backends that have nothing to pre-populate.
    """

    def discover(self, managed_deployments: Optional[set[str]]) -> list[str]:
        """Pre-populate the participant set at startup (no-op by default).

        ``managed_deployments`` scopes discovery in explicit mode, or ``None``
        for implicit mode. Returns the discovered participant identities.
        """
        return []

    def ensure_participant(
        self,
        participant_id: str,
        caller_name: str,
        namespace: str,
        deployment_name: str,
    ) -> None:
        """Register a participant (idempotent) so it counts toward the budget and
        can be scaled."""
        raise NotImplementedError

    def participant_exists(self, participant_id: str) -> bool:
        """Whether ``participant_id`` has been registered/discovered."""
        raise NotImplementedError

    def remember_roles(self, participant_id: str, targets: list[TargetReplica]) -> None:
        """Record any backend-specific hints carried by a request's targets
        (e.g. ``TargetReplica.component_name``) before the next ``observe``.

        No-op by default; backends that need request context to read pool state
        override it. The orchestrator forwards ``targets`` opaquely.
        """
        return None

    def observe(self, require_complete: bool = False) -> PoolSnapshot:
        """Read current pool state for every known participant.

        When ``require_complete`` is true, a failure to read any participant
        raises instead of being skipped, so budget enforcement cannot
        under-count usage from a partially-read snapshot.
        """
        raise NotImplementedError

    async def scale(
        self,
        participant_id: str,
        targets: list[TargetReplica],
        blocking: bool,
    ) -> None:
        """Apply desired replica targets to one participant."""
        raise NotImplementedError

    def current_replicas(self, participant_id: str) -> dict[str, int]:
        """Read back current replica counts as ``{sub_type: replicas}``."""
        raise NotImplementedError
