# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serialized, fail-closed execution of fixed-shape allocation plans.

Only the reconciliation coroutine touches capacity. Request handlers update
retained demand without awaiting it, including while a donor is draining.
Reservations and transfer intent are process-local: restart must fence the old
writer and reconstruct an entirely settled, valid allocation before proceeding.

A transfer moves through ``REDUCING -> WAITING_RELEASE -> STARTING ->
SETTLING``: donors shrink first, their GPUs must be observed released, and
only then does the requester grow.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import OrderedDict
from collections.abc import Mapping
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
    AllocationPhase,
    ManagedAllocationStatus,
    ScaleResponse,
    ScaleStatus,
)

logger = logging.getLogger(__name__)

_MAX_REQUEST_ID_LENGTH = 128
_REQUEST_ID_WINDOW = 256


@dataclass
class _Transfer:
    transfer_id: str
    requester: str
    plan: AllocationPlan
    before: dict[str, int]
    expected: dict[str, int]
    started: float
    phase: AllocationPhase = AllocationPhase.REDUCING

    def involves(self, pool_id: str) -> bool:
        return self.plan.target_replicas[pool_id] != self.before[pool_id]


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
        # Demand, grant and observation are tracked independently: a grant is
        # a committed target, never evidence of ready capacity.
        self.desired: dict[str, int] = {}
        self.request_ids: dict[str, str] = {}
        self.reasons: dict[str, str] = {}
        self.granted: dict[str, int] = {}
        self.observed: dict[str, ManagedPoolSnapshot] = {}
        self.active: _Transfer | None = None
        self.phase = AllocationPhase.RECOVERING
        self.reason = "waiting for a fenced, fully settled allocation"
        self._seen: dict[str, OrderedDict[str, int]] = {
            pool_id: OrderedDict() for pool_id in self.pools
        }
        self._identities: dict[str, str] = {}
        self._acquired = False
        self._started = clock()
        self._running = False

    # -- Request handling (never awaits capacity) ---------------------------

    def submit(self, pool_id: str, desired: int, request_id: str) -> ScaleResponse:
        """Retain the newest demand; never wait for Kubernetes or a transfer."""
        self.pools[pool_id].normalize_replicas(desired)
        if not request_id or len(request_id) > _MAX_REQUEST_ID_LENGTH:
            raise ValueError(
                "managed requests require a request_id of "
                f"1..{_MAX_REQUEST_ID_LENGTH} characters"
            )
        seen = self._seen[pool_id]
        if request_id in seen:
            if seen[request_id] != desired:
                raise ValueError("request_id was already used with different demand")
            return self.status(pool_id)
        seen[request_id] = desired
        if len(seen) > _REQUEST_ID_WINDOW:
            seen.popitem(last=False)
        self.desired[pool_id] = desired
        self.request_ids[pool_id] = request_id
        self.reasons[pool_id] = "demand queued for reconciliation"
        return self.status(pool_id)

    def status(self, pool_id: str) -> ScaleResponse:
        pool = self.pools[pool_id]
        phase, reason = self._pool_phase(pool_id)
        observation = self.observed.get(pool_id)
        allocation = ManagedAllocationStatus(
            pool_id=pool_id,
            controller_id=self.controller_id,
            phase=phase,
            reason=reason,
            desired_replicas=self.desired.get(pool_id),
            granted_replicas=self.granted.get(pool_id),
            observed_replicas=observation.observed_replicas if observation else None,
            ready_replicas=observation.ready_replicas if observation else None,
            terminating_replicas=(
                observation.terminating_replicas if observation else None
            ),
            request_id=self.request_ids.get(pool_id),
            transfer_id=self.active.transfer_id if self.active else None,
        )
        if phase == AllocationPhase.RECOVERY_REQUIRED:
            status = ScaleStatus.ERROR
        elif phase == AllocationPhase.SETTLED:
            status = ScaleStatus.SUCCESS
        else:
            status = ScaleStatus.SCALING
        return ScaleResponse(
            status=status,
            message=reason,
            current_replicas=(
                {pool.role: observation.observed_replicas} if observation else {}
            ),
            allocation=allocation,
        )

    def _pool_phase(self, pool_id: str) -> tuple[AllocationPhase, str]:
        if self.phase != AllocationPhase.READY:
            return self.phase, self.reason
        reason = self.reasons.get(pool_id, "settled; awaiting demand")
        if self.active is not None and self.active.involves(pool_id):
            return self.active.phase, reason
        desired = self.desired.get(pool_id)
        if (
            desired is not None
            and self.pools[pool_id].normalize_replicas(desired) != self.granted[pool_id]
        ):
            return AllocationPhase.PENDING, reason
        return AllocationPhase.SETTLED, reason

    # -- Reconciliation ------------------------------------------------------

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

    async def reconcile_once(self) -> None:
        """One bounded observation/execution step; owned solely by run()."""
        if self.phase == AllocationPhase.RECOVERY_REQUIRED:
            return
        try:
            if not self._acquired:
                await self.capacity.acquire(self.controller_id)
                self._acquired = True
            snapshots = await self.capacity.observe()
            self._validate_snapshot(snapshots)
            self.observed = snapshots
            if self.phase == AllocationPhase.RECOVERING:
                self._recover(snapshots)
                if self.phase != AllocationPhase.READY:
                    return
            self._check_expected(snapshots)
            if self.active is None and _all_settled(snapshots):
                # A worker restart outside a transfer only defers planning:
                # targets and holdings are still bounded by the checks above.
                self._choose_plan()
            if self.active is not None:
                await self._advance(snapshots)
        except (ManagedCapacityError, TimeoutError) as exc:
            self._freeze(str(exc))

    def _freeze(self, reason: str) -> None:
        self.phase = AllocationPhase.RECOVERY_REQUIRED
        self.reason = reason
        # Preserve grants/reservations, but never present stale observations as
        # proof of present capacity. There is deliberately no reset endpoint.
        self.observed = {}
        logger.error("Managed allocation requires manual recovery: %s", reason)

    def _validate_snapshot(self, snapshots: Mapping[str, ManagedPoolSnapshot]) -> None:
        """Reject observations the capacity backend should never produce."""
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
        observed = {p: s.observed_replicas for p, s in snapshots.items()}
        if self._gpus(observed) > self.policy.max_total_gpus:
            raise ManagedCapacityError("observed capacity exceeds the shared ceiling")

    def _recover(self, snapshots: Mapping[str, ManagedPoolSnapshot]) -> None:
        """Adopt a settled, in-bounds allocation as the baseline, or wait."""
        if not _all_settled(snapshots):
            if self.clock() - self._started >= self.operation_timeout_seconds:
                raise ManagedCapacityError("startup settlement timed out")
            return
        targets = {p: s.target_replicas for p, s in snapshots.items()}
        if any(
            not self.pools[p].min_replicas <= n <= self.pools[p].max_replicas
            for p, n in targets.items()
        ):
            raise ManagedCapacityError("startup allocation violates pool bounds")
        total = self._gpus(targets)
        if not self.policy.min_total_gpus <= total <= self.policy.max_total_gpus:
            raise ManagedCapacityError("startup allocation violates shared bounds")
        self.granted = targets
        self._identities = {p: s.identity for p, s in snapshots.items()}
        self.phase = AllocationPhase.READY
        self.reason = "allocation settled"

    def _check_expected(self, snapshots: Mapping[str, ManagedPoolSnapshot]) -> None:
        """Reject target drift and workers beyond what this controller wrote."""
        transfer = self.active
        expected = transfer.expected if transfer else self.granted
        for pool_id, snap in snapshots.items():
            if snap.target_replicas != expected[pool_id]:
                raise ManagedCapacityError(f"unexpected target drift for {pool_id}")
            # During a transfer a donor may still hold its pre-reduction workers.
            maximum = expected[pool_id]
            if transfer is not None:
                maximum = max(maximum, transfer.before[pool_id])
            if snap.observed_replicas > maximum:
                raise ManagedCapacityError(f"unexpected extra workers for {pool_id}")

    def _choose_plan(self) -> None:
        """Commit the first admissible plan, serving demand in priority order."""
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

    async def _advance(self, snapshots: Mapping[str, ManagedPoolSnapshot]) -> None:
        """Move the active transfer forward by at most one phase."""
        transfer = self.active
        assert transfer is not None
        if self.clock() - transfer.started >= self.operation_timeout_seconds:
            raise ManagedCapacityError(
                "transfer settlement timed out; reservations retained"
            )
        if transfer.phase == AllocationPhase.REDUCING:
            for pool_id, before in transfer.before.items():
                if self.granted[pool_id] < before:
                    await self._scale(pool_id, self.granted[pool_id])
            transfer.phase = AllocationPhase.WAITING_RELEASE
            return
        if not _all_settled(snapshots):
            return
        if transfer.phase == AllocationPhase.WAITING_RELEASE:
            # Every reduction has settled and every other pool is unchanged.
            # Count all present pods, then reserve all not-yet-started replicas.
            held = self._gpus({p: s.observed_replicas for p, s in snapshots.items()})
            growth = self._gpus(
                {
                    p: max(0, self.granted[p] - s.target_replicas)
                    for p, s in snapshots.items()
                }
            )
            if held + growth > self.policy.max_total_gpus:
                raise ManagedCapacityError(
                    "release did not establish requester headroom"
                )
            transfer.phase = AllocationPhase.STARTING
            for pool_id, target in self.granted.items():
                if target > transfer.expected[pool_id]:
                    await self._scale(pool_id, target)
            transfer.phase = AllocationPhase.SETTLING
            return
        if transfer.phase == AllocationPhase.SETTLING:
            self.active = None
            self.reason = "allocation settled"

    async def _scale(self, pool_id: str, replicas: int) -> None:
        assert self.active is not None
        await self.capacity.scale(
            pool_id,
            replicas,
            expected_identity=self._identities[pool_id],
            expected_target=self.active.expected[pool_id],
        )
        self.active.expected[pool_id] = replicas

    def _gpus(self, replicas_by_pool: Mapping[str, int]) -> int:
        return sum(
            replicas * self.pools[pool_id].gpu_per_replica
            for pool_id, replicas in replicas_by_pool.items()
        )


def _all_settled(snapshots: Mapping[str, ManagedPoolSnapshot]) -> bool:
    return all(snapshot.settled for snapshot in snapshots.values())
