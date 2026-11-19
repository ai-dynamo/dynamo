# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure, bounded allocation decisions for fixed-size, single-role pools.

These are committed-allocation decisions, not permission to start workers.
Execution must separately reserve capacity and confirm donor GPU release.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dynamo.global_planner.allocation_policy import (
        AllocationPolicy,
        PoolAllocationPolicy,
    )


DEFAULT_MAX_SEARCH_STEPS = 100_000


class AllocationSearchLimit(ValueError):
    """The bounded search did not prove feasibility or infeasibility."""


@dataclass
class _SearchBudget:
    limit: int
    steps: int = 0

    def __post_init__(self) -> None:
        if type(self.limit) is not int or self.limit <= 0:
            raise ValueError("max_search_steps must be a positive integer")

    def consume(self, work: int = 1) -> None:
        if work > self.limit - self.steps:
            raise AllocationSearchLimit(
                f"allocation search exceeded {self.limit} work units; "
                "feasibility or the preferred plan is unknown"
            )
        self.steps += work


def bounded_feasible_total(
    ranges: Sequence[tuple[int, int, int]],
    minimum: int,
    maximum: int,
    *,
    max_search_steps: int = DEFAULT_MAX_SEARCH_STEPS,
) -> bool:
    """Prove whether a whole-replica vector fits the exact shared GPU band.

    Each range contains ``(minimum replicas, maximum replicas, GPU cost)``.
    Exhaustion raises :class:`AllocationSearchLimit`, never an infeasibility
    result. A transition consumes one work unit; retained reachable totals
    cannot outnumber that bound. No array is sized by the GPU budget.
    """
    budget = _SearchBudget(max_search_steps)
    if any(type(bound) is not int or bound < 0 for bound in (minimum, maximum)):
        raise ValueError("shared GPU bounds must be nonnegative integers")
    if minimum > maximum:
        raise ValueError("minimum exceeds maximum")

    baseline = 0
    largest = 0
    for lower, upper, cost in ranges:
        budget.consume()
        if any(type(value) is not int for value in (lower, upper, cost)):
            raise ValueError("replica bounds and GPU costs must be integers")
        if lower < 0 or upper < lower or cost <= 0:
            raise ValueError("invalid replica interval or GPU cost")
        baseline += lower * cost
        largest += upper * cost

    if baseline > maximum or largest < minimum:
        return False
    if baseline >= minimum:
        return True

    needed, ceiling = minimum - baseline, maximum - baseline
    # Insertion order makes the bounded search deterministic. Remaining pools
    # retain their baseline, so reaching the band is already a complete witness.
    totals = {0: None}
    for lower, upper, cost in ranges:
        following: dict[int, None] = {}
        for total in totals:
            largest_extra = min((upper - lower) * cost, ceiling - total)
            for extra in range(0, largest_extra + 1, cost):
                budget.consume()
                candidate = total + extra
                if candidate >= needed:
                    return True
                following[candidate] = None
        totals = following
    return False


class AllocationReason(str, Enum):
    ADMITTED = "admitted"
    POOL_MINIMUM = "pool_minimum"
    POOL_MAXIMUM = "pool_maximum"
    GLOBAL_MINIMUM = "global_minimum"
    REQUESTER_BUSY = "requester_busy"
    INSUFFICIENT_CAPACITY = "insufficient_capacity"
    GRANULARITY = "granularity"
    SEARCH_LIMIT = "search_limit"


@dataclass(frozen=True)
class AllocationPlan:
    """A complete proposed allocation vector and its exact replica targets."""

    allocations_gpus: dict[str, int]
    target_replicas: dict[str, int]
    reclaimed_gpus: dict[str, int]


@dataclass(frozen=True)
class AllocationDecision:
    """An effective request with either a complete plan or a pending reason."""

    requester_id: str
    desired_replicas: int
    effective_replicas: int
    effective_gpus: int
    reason: AllocationReason
    plan: AllocationPlan | None
    search_steps: int

    @property
    def admitted(self) -> bool:
        return self.plan is not None


def _validate_allocations(
    policy: AllocationPolicy, allocations_gpus: Mapping[str, int]
) -> None:
    if not isinstance(allocations_gpus, Mapping):
        raise ValueError("allocations must be a mapping of pool IDs to GPU counts")
    if set(allocations_gpus) != {pool.pool_id for pool in policy.pools}:
        raise ValueError("allocations must contain exactly the configured pool IDs")
    for pool in policy.pools:
        allocated = allocations_gpus[pool.pool_id]
        if type(allocated) is not int:
            raise ValueError(f"allocation for {pool.pool_id} must be an integer")
        if not pool.effective_min_gpus <= allocated <= pool.effective_max_gpus:
            raise ValueError(f"allocation for {pool.pool_id} violates its pool bounds")
        if allocated % pool.gpu_per_replica:
            raise ValueError(f"allocation for {pool.pool_id} is not whole replicas")
    if (
        not policy.min_total_gpus
        <= sum(allocations_gpus.values())
        <= policy.max_total_gpus
    ):
        raise ValueError("allocations violate the shared GPU bounds")


def plan_allocation(
    policy: AllocationPolicy,
    current_allocations_gpus: Mapping[str, int],
    requester_id: str,
    desired_replicas: int,
    *,
    unavailable_pool_ids: Collection[str] = (),
    desired_replicas_by_pool: Mapping[str, int] | None = None,
    max_search_steps: int = DEFAULT_MAX_SEARCH_STEPS,
) -> AllocationDecision:
    """Plan a full growth target, or a voluntary reduction retaining warm GPUs.

    Inputs are a validated policy and a complete, valid committed GPU vector.
    Desired replicas are clamped to the pool's effective interval. Voluntary
    reductions retain the least whole-replica allocation preserving the shared
    floor. Growth is never partially granted. Busy pools cannot be requesters
    or donors. No scale-down intent is needed from a lower-priority donor.

    Optional retained demand authorizes voluntary releases above each donor's
    normalized demand, including equal/higher-priority donors. Omitted pools
    supply no voluntary release; callers must supply only current demand.
    This authorization never changes allocations without a complete plan.
    The explicit request remains authoritative for the requester.

    Donor plans minimize forced GPU reclamation by descending donor priority
    class, then excess reclamation, then the number of affected pools. Thus
    voluntary releases are preferred over forced reductions. Final ties prefer
    reclaiming from lexicographically earlier pool IDs. Dynamic
    programming retains only the best prefix for each reclaimed total: future
    choices add the same objective terms to either prefix, preserving order.

    Search work, including score-vector construction, is bounded by
    ``max_search_steps``; input validation and sorting scale with the pool
    count separately. Exhaustion returns pending without an unproven incumbent.
    Returned dictionaries are fresh and inputs are never mutated.
    """
    budget = _SearchBudget(max_search_steps)
    _validate_allocations(policy, current_allocations_gpus)
    pools = {pool.pool_id: pool for pool in policy.pools}
    if not isinstance(requester_id, str) or requester_id not in pools:
        raise ValueError(f"unknown requester pool: {requester_id}")
    unavailable = set(unavailable_pool_ids)
    if unavailable - pools.keys():
        raise ValueError("unavailable_pool_ids contains an unknown pool")
    voluntary_releases = _voluntary_releases(
        pools, current_allocations_gpus, desired_replicas_by_pool
    )
    requester = pools[requester_id]
    effective_replicas = requester.normalize_replicas(desired_replicas)
    effective_gpus = effective_replicas * requester.gpu_per_replica
    reason = AllocationReason.ADMITTED
    if effective_replicas > desired_replicas:
        reason = AllocationReason.POOL_MINIMUM
    elif effective_replicas < desired_replicas:
        reason = AllocationReason.POOL_MAXIMUM

    def decision(
        result_reason: AllocationReason, reclaim: Mapping[str, int] | None = None
    ) -> AllocationDecision:
        plan = None
        if reclaim is not None:
            updated = dict(current_allocations_gpus)
            updated[requester_id] = effective_gpus
            for pool_id, amount in reclaim.items():
                updated[pool_id] -= amount
            _validate_allocations(policy, updated)
            plan = AllocationPlan(
                allocations_gpus=dict(sorted(updated.items())),
                target_replicas={
                    pool_id: amount // pools[pool_id].gpu_per_replica
                    for pool_id, amount in sorted(updated.items())
                },
                reclaimed_gpus=dict(sorted(reclaim.items())),
            )
        return AllocationDecision(
            requester_id=requester_id,
            desired_replicas=desired_replicas,
            effective_replicas=effective_replicas,
            effective_gpus=effective_gpus,
            reason=result_reason,
            plan=plan,
            search_steps=budget.steps,
        )

    if requester_id in unavailable:
        return decision(AllocationReason.REQUESTER_BUSY)

    other_total = (
        sum(current_allocations_gpus.values()) - current_allocations_gpus[requester_id]
    )
    if other_total + effective_gpus < policy.min_total_gpus:
        warm_gpus = policy.min_total_gpus - other_total
        effective_replicas = (
            warm_gpus + requester.gpu_per_replica - 1
        ) // requester.gpu_per_replica
        effective_gpus = effective_replicas * requester.gpu_per_replica
        reason = AllocationReason.GLOBAL_MINIMUM

    proposed_total = other_total + effective_gpus
    required = max(0, proposed_total - policy.max_total_gpus)
    if not required:
        return decision(reason, {})

    reclaim_ceiling = proposed_total - policy.min_total_gpus
    available_releases = {
        pool.pool_id: (
            current_allocations_gpus[pool.pool_id] - pool.effective_min_gpus
            if pool.priority < requester.priority
            else voluntary_releases.get(pool.pool_id, 0)
        )
        for pool in policy.pools
        if pool.pool_id != requester_id and pool.pool_id not in unavailable
    }
    donors = sorted(
        (pool for pool in policy.pools if available_releases.get(pool.pool_id, 0) > 0),
        key=lambda pool: pool.pool_id,
    )
    if sum(available_releases.values()) < required:
        return decision(AllocationReason.INSUFFICIENT_CAPACITY)

    # Above this bound, returning any reclaimed replica still meets the needed
    # reduction without worsening its priority objective, and reduces excess.
    # Excluding those totals preserves the optimum without needless reclaim.
    reclaim_ceiling = min(
        reclaim_ceiling,
        required + max(pool.gpu_per_replica for pool in donors) - 1,
    )

    try:
        reclaim = _cheapest_reclaim(
            donors,
            available_releases,
            voluntary_releases,
            requester.priority,
            required,
            reclaim_ceiling,
            budget,
        )
    except AllocationSearchLimit:
        return decision(AllocationReason.SEARCH_LIMIT)
    if reclaim is None:
        return decision(AllocationReason.GRANULARITY)
    return decision(reason, reclaim)


def _voluntary_releases(
    pools: Mapping[str, PoolAllocationPolicy],
    current_allocations_gpus: Mapping[str, int],
    desired_replicas_by_pool: Mapping[str, int] | None,
) -> dict[str, int]:
    """GPUs each pool would give up to reach its own normalized demand."""
    if desired_replicas_by_pool is None:
        return {}
    if not isinstance(desired_replicas_by_pool, Mapping):
        raise ValueError("desired_replicas_by_pool must be a mapping")
    if set(desired_replicas_by_pool) - pools.keys():
        raise ValueError("desired_replicas_by_pool contains an unknown pool")
    releases = {}
    for pool_id, desired in desired_replicas_by_pool.items():
        pool = pools[pool_id]
        normalized = pool.normalize_replicas(desired) * pool.gpu_per_replica
        releases[pool_id] = max(0, current_allocations_gpus[pool_id] - normalized)
    return releases


def _cheapest_reclaim(
    donors: Sequence[PoolAllocationPolicy],
    available_releases: Mapping[str, int],
    voluntary_releases: Mapping[str, int],
    requester_priority: int,
    required: int,
    reclaim_ceiling: int,
    budget: _SearchBudget,
) -> dict[str, int] | None:
    """Return the preferred GPU reduction per donor reclaiming ``required``.

    Returns ``None`` when whole-replica granularity admits no reduction in
    ``[required, reclaim_ceiling]``; raises :class:`AllocationSearchLimit` if
    the search budget is exhausted first.
    """
    # A score contains forced priority-class totals, affected pools, and negated
    # reductions in pool-ID order. Negative values implement the stable tie.
    priorities = sorted(
        {pool.priority for pool in donors if pool.priority < requester_priority},
        reverse=True,
    )
    priority_indexes = {priority: index for index, priority in enumerate(priorities)}
    budget.consume(len(donors) + len(priorities))
    states: dict[int, tuple[tuple[int, ...], int, tuple[int, ...]]] = {
        0: ((0,) * len(priorities), 0, ())
    }
    for index, pool in enumerate(donors):
        following: dict[int, tuple[tuple[int, ...], int, tuple[int, ...]]] = {}
        available = available_releases[pool.pool_id]
        for total, (class_totals, affected, reductions) in states.items():
            largest = min(available, reclaim_ceiling - total)
            for amount in range(0, largest + 1, pool.gpu_per_replica):
                # Charge tuple construction and comparison by their sizes,
                # so a very wide policy cannot bypass the search-work bound.
                budget.consume(1 + len(priorities) + index + 1)
                updated_classes = list(class_totals)
                forced = max(0, amount - voluntary_releases.get(pool.pool_id, 0))
                if forced:
                    updated_classes[priority_indexes[pool.priority]] += forced
                score = (
                    tuple(updated_classes),
                    affected + (amount > 0),
                    (*reductions, -amount),
                )
                candidate = total + amount
                incumbent = following.get(candidate)
                if incumbent is None or score < incumbent:
                    following[candidate] = score
        states = following

    candidates = (
        (class_totals, total - required, affected, reductions)
        for total, (class_totals, affected, reductions) in states.items()
        if total >= required
    )
    best = min(candidates, default=None)
    if best is None:
        return None
    return {
        pool.pool_id: -amount
        for pool, amount in zip(donors, best[3], strict=True)
        if amount
    }
