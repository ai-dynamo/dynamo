# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-memory Global Planner capacity backend for offline Replay.

The production Global Planner observes and patches Kubernetes through a
``CapacityManager``. Offline Replay uses the same orchestrator with this backend:
configured participants are pre-registered, desired replica counts are mirrored
in memory, and approved scale calls are staged as ordered actions for the Rust
simulation world to apply after the Python callback returns.

One orchestrator submission may scale several participants. ``begin_batch`` /
``finish_batch`` make those calls visible to the orchestrator as one overlay while
keeping the committed mirror unchanged until the controller accepts the complete
batch. ``abort_batch`` discards both the overlay and its actions.
"""

from __future__ import annotations

import operator
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional

from dynamo.global_planner.capacity_manager import (
    CapacityManager,
    PoolSnapshot,
    PoolSpec,
)
from dynamo.planner import SubComponentType, TargetReplica

__all__ = [
    "ReplayCapacityManager",
    "ReplayParticipantSpec",
    "ReplayScaleAction",
    "ReplayScaleTarget",
]


@dataclass(frozen=True)
class ReplayParticipantSpec:
    """Pre-registered Global Planner participant and its capacity pools."""

    participant_id: str
    pools: tuple[PoolSpec, ...]


@dataclass(frozen=True)
class ReplayScaleTarget:
    """One immutable pool target staged for the simulation world."""

    sub_type: str
    desired_replicas: int
    component_name: Optional[str] = None


@dataclass(frozen=True)
class ReplayScaleAction:
    """One ordered backend ``scale`` call staged during an active batch."""

    participant_id: str
    targets: tuple[ReplayScaleTarget, ...]
    blocking: bool


_PoolKey = tuple[str, str]
_VALID_ROLES = frozenset(role.value for role in SubComponentType)


def _non_negative_int(value: object, *, field: str) -> int:
    """Normalize an integer-like value and reject bools and negative values."""
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a non-negative integer, got {value!r}")
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise ValueError(
            f"{field} must be a non-negative integer, got {value!r}"
        ) from exc
    if normalized < 0:
        raise ValueError(f"{field} must be non-negative, got {normalized}")
    return normalized


def _normalize_role(value: object, *, field: str) -> str:
    """Normalize a planner role to its string value."""
    raw = value.value if isinstance(value, SubComponentType) else value
    if not isinstance(raw, str) or raw not in _VALID_ROLES:
        valid = ", ".join(sorted(_VALID_ROLES))
        raise ValueError(f"{field} must be one of [{valid}], got {raw!r}")
    return raw


class ReplayCapacityManager(CapacityManager):
    """Transactional in-memory capacity backend for a Replay world.

    ``scale`` never mutates a Rust runtime. It updates the active batch overlay
    and records an immutable action. The world controller obtains those actions
    from :meth:`finish_batch` and applies them after the planner callback returns.
    """

    def __init__(self, participants: Iterable[ReplayParticipantSpec]):
        self._participant_ids: tuple[str, ...]
        self._pool_specs: dict[_PoolKey, PoolSpec] = {}
        self._roles_by_participant: dict[str, tuple[str, ...]] = {}
        self._desired_replicas: dict[_PoolKey, int] = {}
        self._batch_overlay: Optional[dict[_PoolKey, int]] = None
        self._batch_actions: Optional[list[ReplayScaleAction]] = None

        participant_ids: set[str] = set()
        for participant in participants:
            participant_id = participant.participant_id
            if not isinstance(participant_id, str) or not participant_id:
                raise ValueError("participant_id must be a non-empty string")
            if participant_id in participant_ids:
                raise ValueError(f"duplicate replay participant {participant_id!r}")
            participant_ids.add(participant_id)

            pools = tuple(participant.pools)
            if not pools:
                raise ValueError(
                    f"replay participant {participant_id!r} must define at least "
                    "one pool"
                )

            roles: set[str] = set()
            for pool in pools:
                role = _normalize_role(
                    pool.sub_type,
                    field=f"pool role for participant {participant_id!r}",
                )
                if role in roles:
                    raise ValueError(
                        f"duplicate pool role {role!r} for participant "
                        f"{participant_id!r}"
                    )
                roles.add(role)

                replicas = _non_negative_int(
                    pool.current_replicas,
                    field=f"initial replicas for {participant_id!r}/{role}",
                )
                gpu_per_replica = _non_negative_int(
                    pool.gpu_per_replica,
                    field=f"gpu_per_replica for {participant_id!r}/{role}",
                )
                key = (participant_id, role)
                self._pool_specs[key] = PoolSpec(
                    sub_type=role,
                    current_replicas=replicas,
                    gpu_per_replica=gpu_per_replica,
                    component_name=pool.component_name,
                )
                self._desired_replicas[key] = replicas

            self._roles_by_participant[participant_id] = tuple(sorted(roles))

        self._participant_ids = tuple(sorted(participant_ids))

    @property
    def batch_active(self) -> bool:
        """Whether a scale-action batch is currently being staged."""
        return self._batch_overlay is not None

    def begin_batch(self) -> None:
        """Start a new desired-state overlay and ordered action batch."""
        if self.batch_active:
            raise RuntimeError("a replay capacity batch is already active")
        self._batch_overlay = {}
        self._batch_actions = []

    def finish_batch(self) -> tuple[ReplayScaleAction, ...]:
        """Commit the active overlay and return its actions in apply order."""
        overlay, actions = self._require_batch()
        self._desired_replicas.update(overlay)
        result = tuple(actions)
        self._batch_overlay = None
        self._batch_actions = None
        return result

    def abort_batch(self) -> None:
        """Discard the active overlay and every action staged in it."""
        self._require_batch()
        self._batch_overlay = None
        self._batch_actions = None

    def discover(self, managed_deployments: Optional[set[str]]) -> list[str]:
        """Return pre-registered participant IDs in deterministic order."""
        if managed_deployments is None:
            return list(self._participant_ids)
        return [
            participant_id
            for participant_id in self._participant_ids
            if participant_id in managed_deployments
        ]

    def ensure_participant(
        self,
        participant_id: str,
        caller_name: str,
        namespace: str,
        deployment_name: str,
    ) -> None:
        """Validate that a participant was declared by the Replay scenario."""
        del caller_name, namespace, deployment_name
        self._require_participant(participant_id)

    def participant_exists(self, participant_id: str) -> bool:
        return participant_id in self._roles_by_participant

    def remember_roles(self, participant_id: str, targets: list[TargetReplica]) -> None:
        """Replay pool roles are explicit; validate identity and otherwise no-op."""
        del targets
        self._require_participant(participant_id)

    def observe(self, require_complete: bool = False) -> PoolSnapshot:
        """Return a complete, deterministic snapshot of the current overlay."""
        del require_complete
        snapshot: PoolSnapshot = {}
        for participant_id in self._participant_ids:
            pools: dict[str, PoolSpec] = {}
            for role in self._roles_by_participant[participant_id]:
                key = (participant_id, role)
                registered = self._pool_specs[key]
                pools[role] = PoolSpec(
                    sub_type=role,
                    current_replicas=self._desired_for(key),
                    gpu_per_replica=registered.gpu_per_replica,
                    component_name=registered.component_name,
                )
            snapshot[participant_id] = pools
        return snapshot

    async def observe_async(self, require_complete: bool = False) -> PoolSnapshot:
        """Return in-memory state directly; no thread or infrastructure I/O."""
        return self.observe(require_complete)

    async def scale(
        self,
        participant_id: str,
        targets: list[TargetReplica],
        blocking: bool,
    ) -> None:
        """Stage one ordered participant action in the active batch.

        Every target is validated before the overlay or action list is mutated, so
        a malformed call cannot leave a partially staged participant update.
        """
        self._require_participant(participant_id)
        overlay, actions = self._require_batch()
        if not targets:
            raise ValueError("replay scale action must include at least one target")

        seen_roles: set[str] = set()
        normalized_targets: list[ReplayScaleTarget] = []
        updates: list[tuple[_PoolKey, int]] = []
        for target in targets:
            role = _normalize_role(
                target.sub_component_type,
                field=f"target role for participant {participant_id!r}",
            )
            if role in seen_roles:
                raise ValueError(
                    f"duplicate target role {role!r} for participant {participant_id!r}"
                )
            seen_roles.add(role)

            key = (participant_id, role)
            if key not in self._pool_specs:
                raise ValueError(
                    f"unknown pool role {role!r} for participant {participant_id!r}"
                )
            desired = _non_negative_int(
                target.desired_replicas,
                field=f"desired replicas for {participant_id!r}/{role}",
            )
            normalized_targets.append(
                ReplayScaleTarget(
                    sub_type=role,
                    desired_replicas=desired,
                    component_name=target.component_name,
                )
            )
            updates.append((key, desired))

        for key, desired in updates:
            overlay[key] = desired
        actions.append(
            ReplayScaleAction(
                participant_id=participant_id,
                targets=tuple(normalized_targets),
                blocking=bool(blocking),
            )
        )

    def current_replicas(self, participant_id: str) -> dict[str, int]:
        """Read desired counts, including changes in the active batch overlay."""
        self._require_participant(participant_id)
        return {
            role: self._desired_for((participant_id, role))
            for role in self._roles_by_participant[participant_id]
        }

    def _desired_for(self, key: _PoolKey) -> int:
        overlay = self._batch_overlay
        if overlay is not None and key in overlay:
            return overlay[key]
        return self._desired_replicas[key]

    def _require_participant(self, participant_id: str) -> None:
        if not self.participant_exists(participant_id):
            raise KeyError(
                f"replay participant {participant_id!r} was not pre-registered"
            )

    def _require_batch(
        self,
    ) -> tuple[dict[_PoolKey, int], list[ReplayScaleAction]]:
        if self._batch_overlay is None or self._batch_actions is None:
            raise RuntimeError("no replay capacity batch is active")
        return self._batch_overlay, self._batch_actions
