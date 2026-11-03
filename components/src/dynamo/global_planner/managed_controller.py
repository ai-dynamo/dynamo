# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serialized, fail-closed execution of fixed-shape allocation plans.

Only the reconciliation coroutine touches capacity. Request handlers update
retained demand without awaiting it, including while a donor is draining.
Reservations and transfer intent are process-local: restart must fence the old
writer and reconstruct an entirely settled, valid allocation before proceeding.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

from dynamo.global_planner.allocation import AllocationPlan, plan_allocation
from dynamo.global_planner.allocation_policy import AllocationPolicy
from dynamo.global_planner.managed_capacity import (
    ManagedCapacity,
    ManagedCapacityError,
    ManagedPoolSnapshot,
)
from dynamo.planner.connectors.protocol import (
    ManagedAllocationStatus,
    ScaleResponse,
    ScaleStatus,
)

logger = logging.getLogger(__name__)


@dataclass
class _Transfer:
    transfer_id: str
    requester: str
    plan: AllocationPlan
    before: dict[str, int]
    expected: dict[str, int]
    started: float
    phase: str = "reducing"


class ManagedAllocationController:
    """One controller, one execution task, one capacity-changing plan at a time.

    Request IDs deduplicate the most recent 256 updates per pool. The supported
    Local Planner transport does not retry sent requests; clients must not replay
    requests beyond this window or across controller restarts.
    """

    def __init__(
        self,
        policy: AllocationPolicy,
        capacity: ManagedCapacity,
        *,
        poll_interval_seconds: float = 2.0,
        operation_timeout_seconds: float = 1800.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.policy = policy
        self.capacity = capacity
        self.poll_interval_seconds = poll_interval_seconds
        self.operation_timeout_seconds = operation_timeout_seconds
        self.clock = clock
        self.controller_id = str(uuid.uuid4())
        self.pools = {pool.pool_id: pool for pool in policy.pools}
        self.desired: dict[str, int] = {}
        self.request_ids: dict[str, str] = {}
        self._seen: dict[str, OrderedDict[str, int]] = {
            pool_id: OrderedDict() for pool_id in self.pools
        }
        self.reasons: dict[str, str] = {}
        self.observed: dict[str, ManagedPoolSnapshot] = {}
        self.granted: dict[str, int] = {}
        self._identities: dict[str, str] = {}
        self.active: _Transfer | None = None
        self.phase = "recovering"
        self.reason = "waiting for a fenced, fully settled allocation"
        self._acquired = False
        self._started = clock()
        self._running = False

    def submit(self, pool_id: str, desired: int, request_id: str) -> ScaleResponse:
        """Retain the newest demand; never wait for Kubernetes or a transfer."""
        self.pools[pool_id].normalize_replicas(desired)
        if not request_id or len(request_id) > 128:
            raise ValueError(
                "managed requests require a request_id of 1..128 characters"
            )
        seen = self._seen[pool_id]
        if request_id in seen:
            if seen[request_id] != desired:
                raise ValueError("request_id was already used with different demand")
            return self.status(pool_id)
        seen[request_id] = desired
        if len(seen) > 256:
            seen.popitem(last=False)
        self.desired[pool_id] = desired
        self.request_ids[pool_id] = request_id
        self.reasons[pool_id] = "demand queued for reconciliation"
        return self.status(pool_id)

    def status(self, pool_id: str) -> ScaleResponse:
        pool = self.pools[pool_id]
        observation = self.observed.get(pool_id)
        grant = self.granted.get(pool_id)
        desired = self.desired.get(pool_id)
        phase = self.phase
        reason = self.reason
        if self.phase == "ready":
            reason = self.reasons.get(pool_id, "settled; awaiting demand")
            if self.active is not None:
                phase = self.active.phase
            elif desired is not None and pool.normalize_replicas(desired) != grant:
                phase = "pending"
            else:
                phase = "settled"
        allocation = ManagedAllocationStatus(
            pool_id=pool_id,
            controller_id=self.controller_id,
            phase=phase,
            reason=reason,
            desired_replicas=desired,
            granted_replicas=grant,
            observed_replicas=observation.observed_replicas if observation else None,
            ready_replicas=observation.ready_replicas if observation else None,
            terminating_replicas=observation.terminating_replicas
            if observation
            else None,
            request_id=self.request_ids.get(pool_id),
            transfer_id=self.active.transfer_id if self.active else None,
        )
        status = ScaleStatus.SCALING
        if phase == "recovery_required":
            status = ScaleStatus.ERROR
        elif phase == "settled":
            status = ScaleStatus.SUCCESS
        return ScaleResponse(
            status=status,
            message=reason,
            current_replicas={pool.role: observation.observed_replicas}
            if observation
            else {},
            allocation=allocation,
        )

    def _freeze(self, reason: str) -> None:
        self.phase = "recovery_required"
        self.reason = reason
        # Preserve grants/reservations, but never present stale observations as
        # proof of present capacity. There is deliberately no reset endpoint.
        self.observed = {}
        logger.error("Managed allocation requires manual recovery: %s", reason)

    def _validate_snapshot(self, snapshots: dict[str, ManagedPoolSnapshot]) -> None:
        if snapshots.keys() != self.pools.keys():
            raise ManagedCapacityError("incomplete pool observation")
        for pool_id, snap in snapshots.items():
            counts = (
                snap.target_replicas,
                snap.observed_replicas,
                snap.ready_replicas,
                snap.terminating_replicas,
            )
            if (
                snap.pool_id != pool_id
                or not snap.identity
                or any(type(count) is not int or count < 0 for count in counts)
            ):
                raise ManagedCapacityError(f"invalid observation for {pool_id}")
            if (
                snap.ready_replicas > snap.observed_replicas
                or snap.terminating_replicas > snap.observed_replicas
            ):
                raise ManagedCapacityError(f"inconsistent observation for {pool_id}")
            if snap.settled and (
                snap.observed_replicas != snap.target_replicas
                or snap.ready_replicas != snap.target_replicas
                or snap.terminating_replicas
            ):
                raise ManagedCapacityError(f"false settled observation for {pool_id}")
            if self._identities and self._identities[pool_id] != snap.identity:
                raise ManagedCapacityError(f"enrollment identity changed for {pool_id}")
        total = sum(
            s.observed_replicas * self.pools[p].gpu_per_replica
            for p, s in snapshots.items()
        )
        if total > self.policy.max_total_gpus:
            raise ManagedCapacityError("observed capacity exceeds the shared ceiling")

    def _recover(self, snapshots: dict[str, ManagedPoolSnapshot]) -> None:
        if not all(s.settled for s in snapshots.values()):
            if self.clock() - self._started >= self.operation_timeout_seconds:
                raise ManagedCapacityError("startup settlement timed out")
            return
        targets = {p: s.target_replicas for p, s in snapshots.items()}
        if any(
            not self.pools[p].min_replicas <= n <= self.pools[p].max_replicas
            for p, n in targets.items()
        ):
            raise ManagedCapacityError("startup allocation violates pool bounds")
        total = sum(n * self.pools[p].gpu_per_replica for p, n in targets.items())
        if not self.policy.min_total_gpus <= total <= self.policy.max_total_gpus:
            raise ManagedCapacityError("startup allocation violates shared bounds")
        self.granted = targets
        self._identities = {p: s.identity for p, s in snapshots.items()}
        self.phase = "ready"
        self.reason = "allocation settled"

    def _check_expected(self, snapshots: dict[str, ManagedPoolSnapshot]) -> None:
        expected = self.active.expected if self.active else self.granted
        for pool_id, snap in snapshots.items():
            if snap.target_replicas != expected[pool_id]:
                raise ManagedCapacityError(f"unexpected target drift for {pool_id}")
            maximum = (
                max(self.active.before[pool_id], expected[pool_id])
                if self.active
                else expected[pool_id]
            )
            if snap.observed_replicas > maximum:
                raise ManagedCapacityError(f"unexpected extra workers for {pool_id}")
            if self.active is None and not snap.settled:
                raise ManagedCapacityError(
                    f"unsettled pool outside a transfer: {pool_id}"
                )

    def _choose_plan(self) -> None:
        current = {
            p: n * self.pools[p].gpu_per_replica for p, n in self.granted.items()
        }
        for pool_id in sorted(self.desired, key=lambda p: (-self.pools[p].priority, p)):
            decision = plan_allocation(
                self.policy,
                current,
                pool_id,
                self.desired[pool_id],
                desired_replicas_by_pool=self.desired,
            )
            self.reasons[pool_id] = decision.reason.value
            if decision.plan is None or decision.plan.target_replicas == self.granted:
                continue
            self.active = _Transfer(
                transfer_id=str(uuid.uuid4()),
                requester=pool_id,
                plan=decision.plan,
                before=dict(self.granted),
                expected=dict(self.granted),
                started=self.clock(),
            )
            # This is the committed allocation, including capacity reserved for
            # the requester. Observed workers remain charged until release.
            self.granted = dict(decision.plan.target_replicas)
            return

    async def _scale(self, pool_id: str, replicas: int) -> None:
        assert self.active is not None
        await self.capacity.scale(
            pool_id,
            replicas,
            expected_identity=self._identities[pool_id],
            expected_target=self.active.expected[pool_id],
        )
        self.active.expected[pool_id] = replicas

    async def _advance(self, snapshots: dict[str, ManagedPoolSnapshot]) -> None:
        transfer = self.active
        assert transfer is not None
        if self.clock() - transfer.started >= self.operation_timeout_seconds:
            raise ManagedCapacityError(
                "transfer settlement timed out; reservations retained"
            )
        if transfer.phase == "reducing":
            for pool_id, before in transfer.before.items():
                target = self.granted[pool_id]
                if target < before:
                    await self._scale(pool_id, target)
            transfer.phase = "waiting_release"
            return
        if not all(s.settled for s in snapshots.values()):
            return
        if transfer.phase == "waiting_release":
            # Every reduction has settled and every other pool is unchanged.
            # Count all present pods, then reserve all not-yet-started replicas.
            held = sum(
                s.observed_replicas * self.pools[p].gpu_per_replica
                for p, s in snapshots.items()
            )
            growth = sum(
                max(0, self.granted[p] - s.target_replicas)
                * self.pools[p].gpu_per_replica
                for p, s in snapshots.items()
            )
            if held + growth > self.policy.max_total_gpus:
                raise ManagedCapacityError(
                    "release did not establish requester headroom"
                )
            transfer.phase = "starting"
            for pool_id, target in self.granted.items():
                if target > transfer.expected[pool_id]:
                    await self._scale(pool_id, target)
            transfer.phase = "settling"
            return
        if transfer.phase == "settling":
            self.active = None
            self.reason = "allocation settled"

    async def reconcile_once(self) -> None:
        """One bounded observation/execution step; owned solely by run()."""
        if self.phase == "recovery_required":
            return
        try:
            if not self._acquired:
                await self.capacity.acquire(self.controller_id)
                self._acquired = True
            snapshots = await self.capacity.observe()
            self._validate_snapshot(snapshots)
            self.observed = snapshots
            if self.phase == "recovering":
                self._recover(snapshots)
                if self.phase != "ready":
                    return
            self._check_expected(snapshots)
            if self.active is None:
                self._choose_plan()
            if self.active is not None:
                await self._advance(snapshots)
        except (ManagedCapacityError, TimeoutError) as exc:
            self._freeze(str(exc))

    async def run(self) -> None:
        if self._running:
            raise RuntimeError("managed controller is already running")
        self._running = True
        try:
            while True:
                await self.reconcile_once()
                await asyncio.sleep(self.poll_interval_seconds)
        finally:
            self._freeze("controller stopped; restart requires settled reconstruction")
