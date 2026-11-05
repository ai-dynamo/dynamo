# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact allocation policy checks, including an independent exhaustive oracle."""

from dataclasses import asdict
from itertools import product

import pytest

from dynamo.global_planner.allocation import (
    AllocationReason,
    AllocationSearchLimit,
    bounded_feasible_total,
    plan_allocation,
)
from dynamo.global_planner.allocation_policy import (
    AllocationPolicy,
    PoolAllocationPolicy,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _pool(
    pool_id: str,
    *,
    cost: int = 4,
    minimum: int = 4,
    maximum: int = 16,
    priority: int = 0,
) -> PoolAllocationPolicy:
    return PoolAllocationPolicy(
        pool_id=pool_id,
        participant_id=f"test/{pool_id}",
        component_name="worker",
        role="decode",
        gpu_per_replica=cost,
        min_gpus=minimum,
        max_gpus=maximum,
        priority=priority,
    )


def _policy(*pools, floor=0, ceiling=16) -> AllocationPolicy:
    return AllocationPolicy(min_total_gpus=floor, max_total_gpus=ceiling, pools=pools)


def test_reclaim_without_a_voluntary_donor_intent():
    policy = _policy(_pool("interactive", priority=100), _pool("batch", priority=10))
    current = {"interactive": 8, "batch": 8}

    decision = plan_allocation(policy, current, "interactive", 3)

    assert decision.admitted
    assert decision.reason is AllocationReason.ADMITTED
    assert decision.plan.allocations_gpus == {"interactive": 12, "batch": 4}
    assert decision.plan.target_replicas == {"interactive": 3, "batch": 1}
    assert decision.plan.reclaimed_gpus == {"batch": 4}
    assert current == {"interactive": 8, "batch": 8}


def test_pending_growth_never_partially_reclaims():
    policy = _policy(_pool("interactive", priority=100), _pool("batch", priority=10))
    current = {"interactive": 8, "batch": 8}

    decision = plan_allocation(policy, current, "interactive", 4)

    assert decision.reason is AllocationReason.INSUFFICIENT_CAPACITY
    assert not decision.admitted
    assert decision.plan is None
    assert decision.effective_replicas == 4
    assert current == {"interactive": 8, "batch": 8}


@pytest.mark.parametrize("donor_priority", [100, 101])
def test_equal_and_higher_priority_pools_cannot_be_preempted(donor_priority):
    policy = _policy(
        _pool("requester", priority=100), _pool("protected", priority=donor_priority)
    )
    decision = plan_allocation(policy, {"requester": 8, "protected": 8}, "requester", 3)
    assert decision.reason is AllocationReason.INSUFFICIENT_CAPACITY
    assert decision.plan is None


def test_donor_minimum_is_protected_and_later_eligible_pool_is_used():
    policy = _policy(
        _pool("requester", priority=100),
        _pool("at-minimum", priority=0),
        _pool("donor", priority=10),
        ceiling=20,
    )
    decision = plan_allocation(
        policy, {"requester": 8, "at-minimum": 4, "donor": 8}, "requester", 3
    )
    assert decision.plan.reclaimed_gpus == {"donor": 4}


def test_free_capacity_needs_no_donor_search():
    policy = _policy(_pool("requester", priority=100), _pool("other"), ceiling=24)
    decision = plan_allocation(
        policy, {"requester": 8, "other": 8}, "requester", 3, max_search_steps=1
    )
    assert decision.plan.allocations_gpus == {"requester": 12, "other": 8}
    assert decision.plan.reclaimed_gpus == {}
    assert decision.search_steps == 0


@pytest.mark.parametrize(
    "desired,expected,reason",
    [(0, 1, AllocationReason.POOL_MINIMUM), (99, 3, AllocationReason.POOL_MAXIMUM)],
)
def test_desired_and_effective_targets_are_separate(desired, expected, reason):
    policy = _policy(_pool("requester", maximum=14))
    decision = plan_allocation(policy, {"requester": 8}, "requester", desired)
    assert decision.desired_replicas == desired
    assert decision.effective_replicas == expected
    assert decision.effective_gpus == expected * 4
    assert decision.reason is reason
    assert decision.plan.target_replicas == {"requester": expected}


@pytest.mark.parametrize("floor,expected", [(8, 4), (9, 8), (12, 8)])
def test_voluntary_downscale_retains_the_smallest_warm_allocation(floor, expected):
    policy = _policy(_pool("requester", minimum=0), _pool("other"), floor=floor)
    decision = plan_allocation(policy, {"requester": 8, "other": 4}, "requester", 0)
    assert decision.reason is AllocationReason.GLOBAL_MINIMUM
    assert decision.plan.allocations_gpus == {"requester": expected, "other": 4}
    assert decision.effective_replicas == expected // 4


def test_equal_global_bounds_allow_only_exact_committed_transfer():
    policy = _policy(
        _pool("requester", priority=100), _pool("donor"), floor=16, ceiling=16
    )
    decision = plan_allocation(policy, {"requester": 8, "donor": 8}, "requester", 3)
    assert decision.plan.allocations_gpus == {"requester": 12, "donor": 4}
    # The pure plan does not assert that a release-before-start execution could
    # maintain 16 ready GPUs throughout the physical transfer.
    downscale = plan_allocation(policy, {"requester": 8, "donor": 8}, "donor", 1)
    assert downscale.reason is AllocationReason.GLOBAL_MINIMUM
    assert downscale.plan.allocations_gpus == {"requester": 8, "donor": 8}


def test_retained_voluntary_demand_releases_warm_capacity_at_exact_floor():
    policy = _policy(
        _pool("interactive", priority=100),
        _pool("batch", priority=10),
        floor=16,
        ceiling=16,
    )
    current = {"interactive": 12, "batch": 4}
    desired = {"interactive": 2, "batch": 2}

    # A standalone reduction cannot cross the shared floor. The later growth
    # request can use that retained voluntary demand in one complete transfer.
    standalone = plan_allocation(policy, current, "interactive", 2)
    assert standalone.reason is AllocationReason.GLOBAL_MINIMUM
    assert standalone.plan.allocations_gpus == current
    transfer = plan_allocation(
        policy, current, "batch", 2, desired_replicas_by_pool=desired
    )
    assert transfer.plan.allocations_gpus == {"interactive": 8, "batch": 8}
    assert transfer.plan.reclaimed_gpus == {"interactive": 4}
    assert current == {"interactive": 12, "batch": 4}
    assert desired == {"interactive": 2, "batch": 2}


@pytest.mark.parametrize("donor_priority", [100, 101])
def test_protected_priority_can_release_only_above_normalized_desired(donor_priority):
    policy = _policy(
        _pool("requester", priority=100),
        _pool("donor", priority=donor_priority),
    )
    current = {"requester": 4, "donor": 12}
    granted = plan_allocation(
        policy, current, "requester", 2, desired_replicas_by_pool={"donor": 2}
    )
    assert granted.plan.allocations_gpus == {"requester": 8, "donor": 8}
    pending = plan_allocation(
        policy, current, "requester", 3, desired_replicas_by_pool={"donor": 2}
    )
    assert pending.reason is AllocationReason.INSUFFICIENT_CAPACITY
    assert pending.plan is None
    normalized = plan_allocation(
        policy, current, "requester", 3, desired_replicas_by_pool={"donor": 0}
    )
    assert normalized.plan.allocations_gpus == {"requester": 12, "donor": 4}


def test_voluntary_releases_precede_disrupting_lower_priority_demand():
    policy = _policy(
        _pool("requester", priority=50),
        _pool("voluntary", priority=100),
        _pool("forced", priority=0),
        floor=20,
        ceiling=20,
    )
    decision = plan_allocation(
        policy,
        {"requester": 4, "voluntary": 8, "forced": 8},
        "requester",
        2,
        desired_replicas_by_pool={"voluntary": 1, "forced": 2},
    )
    assert decision.plan.reclaimed_gpus == {"voluntary": 4}


def test_free_capacity_does_not_eagerly_apply_voluntary_donor_reductions():
    policy = _policy(_pool("requester"), _pool("donor", priority=100), ceiling=24)
    decision = plan_allocation(
        policy,
        {"requester": 8, "donor": 8},
        "requester",
        3,
        desired_replicas_by_pool={"donor": 0},
    )
    assert decision.plan.allocations_gpus == {"requester": 12, "donor": 8}
    assert decision.plan.reclaimed_gpus == {}


def test_voluntary_and_forced_donors_can_form_one_complete_plan():
    policy = _policy(
        _pool("requester", priority=50),
        _pool("voluntary", priority=100),
        _pool("forced", priority=0),
        floor=20,
        ceiling=20,
    )
    current = {"requester": 4, "voluntary": 8, "forced": 8}
    decision = plan_allocation(
        policy,
        current,
        "requester",
        3,
        desired_replicas_by_pool={"voluntary": 1, "forced": 99},
    )
    assert decision.plan.allocations_gpus == {
        "requester": 12,
        "voluntary": 4,
        "forced": 4,
    }
    assert decision.plan.reclaimed_gpus == {"voluntary": 4, "forced": 4}


def test_unavailable_voluntary_donors_do_not_release_capacity():
    policy = _policy(_pool("requester"), _pool("donor", priority=100))
    decision = plan_allocation(
        policy,
        {"requester": 8, "donor": 8},
        "requester",
        3,
        desired_replicas_by_pool={"donor": 1},
        unavailable_pool_ids={"donor"},
    )
    assert decision.reason is AllocationReason.INSUFFICIENT_CAPACITY
    assert decision.plan is None


def test_partial_demand_mapping_does_not_authorize_omitted_protected_donors():
    policy = _policy(_pool("requester"), _pool("donor", priority=100))
    decision = plan_allocation(
        policy,
        {"requester": 8, "donor": 8},
        "requester",
        3,
        desired_replicas_by_pool={"requester": 0},
    )
    assert decision.reason is AllocationReason.INSUFFICIENT_CAPACITY
    assert decision.effective_replicas == 3
    assert decision.plan is None


def test_voluntary_search_exhaustion_never_authorizes_partial_release():
    policy = _policy(_pool("requester"), _pool("donor", priority=100))
    current = {"requester": 8, "donor": 8}
    desired = {"donor": 1}
    complete = plan_allocation(
        policy,
        current,
        "requester",
        3,
        desired_replicas_by_pool=desired,
    )
    assert complete.admitted
    for limit in range(1, complete.search_steps):
        pending = plan_allocation(
            policy,
            current,
            "requester",
            3,
            desired_replicas_by_pool=desired,
            max_search_steps=limit,
        )
        assert pending.reason is AllocationReason.SEARCH_LIMIT
        assert pending.plan is None
    assert current == {"requester": 8, "donor": 8}
    assert desired == {"donor": 1}


@pytest.mark.parametrize(
    "desired",
    [
        [],
        1,
        {"missing": 1},
        {"donor": True},
        {"donor": -1},
        {"donor": 1.0},
        {"donor": "1"},
    ],
)
def test_invalid_retained_demand_is_rejected_even_without_reclamation(desired):
    policy = _policy(_pool("requester"), _pool("donor", priority=100), ceiling=24)
    with pytest.raises(ValueError):
        plan_allocation(
            policy,
            {"requester": 8, "donor": 8},
            "requester",
            3,
            desired_replicas_by_pool=desired,
        )


def test_skip_a_coarse_donor_that_would_cross_the_exact_floor():
    policy = _policy(
        _pool("requester", cost=2, minimum=0, priority=100),
        _pool("a-coarse", cost=4, minimum=0),
        _pool("b-fine", cost=2, minimum=0, priority=1),
        floor=12,
        ceiling=12,
    )
    decision = plan_allocation(
        policy, {"requester": 2, "a-coarse": 8, "b-fine": 2}, "requester", 2
    )
    assert decision.plan.reclaimed_gpus == {"b-fine": 2}


def test_search_combines_donors_to_hit_an_exact_shared_band():
    policy = _policy(
        _pool("requester", cost=1, minimum=0, priority=100),
        _pool("four", cost=4, minimum=0),
        _pool("three", cost=3, minimum=0),
        floor=8,
        ceiling=8,
    )
    decision = plan_allocation(
        policy, {"requester": 1, "four": 4, "three": 3}, "requester", 8
    )
    assert decision.plan.reclaimed_gpus == {"four": 4, "three": 3}


def test_granularity_can_make_scalar_sufficient_donors_infeasible():
    policy = _policy(
        _pool("requester", cost=2, minimum=0, priority=100),
        _pool("donor", cost=4, minimum=0),
        floor=10,
        ceiling=10,
    )
    decision = plan_allocation(policy, {"requester": 2, "donor": 8}, "requester", 2)
    assert decision.reason is AllocationReason.GRANULARITY
    assert decision.plan is None


def test_priority_protection_precedes_excess_reclamation():
    policy = _policy(
        _pool("requester", cost=2, minimum=0, priority=100),
        _pool("low", cost=4, minimum=0, priority=1),
        _pool("high", cost=2, minimum=0, priority=10),
        floor=0,
        ceiling=8,
    )
    decision = plan_allocation(
        policy, {"requester": 2, "low": 4, "high": 2}, "requester", 2
    )
    assert decision.plan.reclaimed_gpus == {"low": 4}


def test_same_priority_prefers_less_excess_then_fewer_donors_then_pool_id():
    pools = (
        _pool("requester", cost=1, minimum=0, priority=100),
        _pool("a-small", cost=1, minimum=0),
        _pool("b-large", cost=2, minimum=0),
        _pool("c-large", cost=2, minimum=0),
        _pool("d-coarse", cost=3, minimum=0),
    )
    current = {"requester": 1, "a-small": 1, "b-large": 2, "c-large": 2, "d-coarse": 3}
    decision = plan_allocation(_policy(*pools, ceiling=9), current, "requester", 3)
    assert decision.plan.reclaimed_gpus == {"b-large": 2}
    reversed_decision = plan_allocation(
        _policy(*reversed(pools), ceiling=9),
        dict(reversed(current.items())),
        "requester",
        3,
    )
    assert reversed_decision == decision


def test_busy_pools_cannot_be_reclaimed_or_changed():
    policy = _policy(_pool("requester", priority=100), _pool("donor"))
    current = {"requester": 8, "donor": 8}
    blocked_donor = plan_allocation(
        policy, current, "requester", 3, unavailable_pool_ids={"donor"}
    )
    assert blocked_donor.reason is AllocationReason.INSUFFICIENT_CAPACITY
    blocked_requester = plan_allocation(
        policy, current, "requester", 1, unavailable_pool_ids={"requester"}
    )
    assert blocked_requester.reason is AllocationReason.REQUESTER_BUSY
    assert blocked_requester.plan is None


def test_search_exhaustion_never_returns_an_unproven_incumbent():
    policy = _policy(
        _pool("requester", cost=1, minimum=0, priority=100),
        _pool("a", cost=1, minimum=0),
        _pool("b", cost=1, minimum=0),
        ceiling=5,
    )
    current = {"requester": 1, "a": 2, "b": 2}
    complete = plan_allocation(policy, current, "requester", 3)
    assert complete.admitted
    for limit in range(1, complete.search_steps):
        limited = plan_allocation(
            policy, current, "requester", 3, max_search_steps=limit
        )
        assert limited.reason is AllocationReason.SEARCH_LIMIT
        assert limited.plan is None
        assert limited.search_steps <= limit


def test_many_donors_with_small_growth_fit_the_default_search_budget():
    donors = [
        _pool(f"donor-{index:02d}", cost=4, minimum=0, maximum=32)
        for index in range(32)
    ]
    current = {pool.pool_id: 32 for pool in donors} | {"requester": 1}
    policy = _policy(
        _pool("requester", cost=1, minimum=0, maximum=2, priority=100),
        *donors,
        ceiling=sum(current.values()),
    )

    decision = plan_allocation(policy, current, "requester", 2)

    assert decision.admitted
    assert decision.plan.reclaimed_gpus == {"donor-00": 4}
    assert decision.plan.allocations_gpus == current | {"requester": 2, "donor-00": 28}


@pytest.mark.parametrize(
    "value", [None, [], 3, {"requester": True}, {"requester": 1.0}, {}]
)
def test_invalid_current_vector_is_rejected(value):
    policy = _policy(_pool("requester"))
    with pytest.raises(ValueError):
        plan_allocation(policy, value, "requester", 2)


@pytest.mark.parametrize("amount", [0, 5, 20])
def test_current_allocations_must_be_within_pool_bounds_and_whole_workers(amount):
    policy = _policy(_pool("requester"))
    with pytest.raises(ValueError):
        plan_allocation(policy, {"requester": amount}, "requester", 2)


@pytest.mark.parametrize("amount", [4, 12])
def test_current_allocations_must_meet_exact_shared_bounds(amount):
    policy = _policy(_pool("requester"), floor=8, ceiling=8)
    with pytest.raises(ValueError, match="shared GPU bounds"):
        plan_allocation(policy, {"requester": amount}, "requester", 2)


@pytest.mark.parametrize("desired", [True, 2.0, "2", -1])
def test_invalid_desired_replicas_are_rejected(desired):
    policy = _policy(_pool("requester"))
    with pytest.raises(ValueError, match="desired_replicas"):
        plan_allocation(policy, {"requester": 8}, "requester", desired)


def test_unknown_pool_id_is_rejected():
    policy = _policy(_pool("requester"))
    with pytest.raises(ValueError, match="unknown requester"):
        plan_allocation(policy, {"requester": 8}, "missing", 1)
    with pytest.raises(ValueError, match="unknown pool"):
        plan_allocation(
            policy, {"requester": 8}, "requester", 1, unavailable_pool_ids={"missing"}
        )


@pytest.mark.parametrize("limit", [0, -1, 1.0, True])
def test_search_limit_must_be_a_positive_integer(limit):
    policy = _policy(_pool("requester"))
    with pytest.raises(ValueError, match="max_search_steps"):
        plan_allocation(
            policy, {"requester": 8}, "requester", 2, max_search_steps=limit
        )


def test_decisions_serialize_without_exposing_input_dictionaries():
    policy = _policy(_pool("requester"))
    current = {"requester": 8}
    decision = plan_allocation(policy, current, "requester", 2)
    serialized = asdict(decision)
    assert serialized["plan"]["allocations_gpus"] == current
    serialized["plan"]["allocations_gpus"]["requester"] = 0
    decision.plan.allocations_gpus["requester"] = 4
    assert current == {"requester": 8}


def _exhaustive_growth_oracle(
    policy, current, requester_id, desired_replicas, retained_demand=None
):
    """Enumerate final vectors directly; no dynamic-programming implementation."""
    pools = sorted(policy.pools, key=lambda pool: pool.pool_id)
    requester = next(pool for pool in pools if pool.pool_id == requester_id)
    retained_demand = retained_demand or {}
    demand_gpus = {
        pool.pool_id: (
            max(
                pool.min_replicas, min(pool.max_replicas, retained_demand[pool.pool_id])
            )
            * pool.gpu_per_replica
            if pool.pool_id in retained_demand
            else current[pool.pool_id]
        )
        for pool in pools
    }
    donors = [pool for pool in pools if pool.pool_id != requester_id]
    priorities = sorted(
        {pool.priority for pool in donors if pool.priority < requester.priority},
        reverse=True,
    )
    domains = []
    for pool in pools:
        if pool.pool_id == requester_id:
            domains.append([desired_replicas * pool.gpu_per_replica])
        else:
            minimum = pool.effective_min_gpus
            if pool.priority >= requester.priority:
                minimum = min(current[pool.pool_id], demand_gpus[pool.pool_id])
            domains.append(
                range(
                    minimum,
                    current[pool.pool_id] + 1,
                    pool.gpu_per_replica,
                )
            )

    candidates = []
    required = max(
        0,
        sum(current.values())
        - current[requester_id]
        + desired_replicas * requester.gpu_per_replica
        - policy.max_total_gpus,
    )
    for values in product(*domains):
        final = dict(zip((pool.pool_id for pool in pools), values, strict=True))
        if not policy.min_total_gpus <= sum(values) <= policy.max_total_gpus:
            continue
        reductions = {
            pool.pool_id: current[pool.pool_id] - final[pool.pool_id] for pool in donors
        }
        score = (
            tuple(
                sum(
                    max(
                        0,
                        min(current[pool.pool_id], demand_gpus[pool.pool_id])
                        - final[pool.pool_id],
                    )
                    for pool in donors
                    if pool.priority == priority
                )
                for priority in priorities
            ),
            sum(reductions.values()) - required,
            sum(amount > 0 for amount in reductions.values()),
            tuple(-reductions[pool.pool_id] for pool in donors),
        )
        candidates.append((score, final))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def test_all_small_growth_plans_match_exhaustive_final_vector_oracle():
    # 1,944 combinations exercise protected priorities, coarse worker sizes,
    # exact/narrow bands, full-target admission, and the complete objective.
    for costs, priorities, slack, headroom, growth in product(
        product((2, 3, 4), repeat=2),
        product((0, 1, 2), repeat=2),
        range(3),
        range(2),
        range(1, 5),
    ):
        current = {"requester": 2, "a": costs[0] * 2, "b": costs[1] * 2}
        total = sum(current.values())
        policy = _policy(
            _pool("requester", cost=1, minimum=0, maximum=8, priority=2),
            _pool(
                "a",
                cost=costs[0],
                minimum=0,
                maximum=costs[0] * 2,
                priority=priorities[0],
            ),
            _pool(
                "b",
                cost=costs[1],
                minimum=0,
                maximum=costs[1] * 2,
                priority=priorities[1],
            ),
            floor=total - slack,
            ceiling=total + headroom,
        )
        expected = _exhaustive_growth_oracle(policy, current, "requester", 2 + growth)
        result = plan_allocation(policy, current, "requester", 2 + growth)
        assert (
            result.plan.allocations_gpus if result.admitted else None
        ) == expected, (costs, priorities, slack, headroom, growth, result)


def test_voluntary_growth_matches_exhaustive_final_vectors():
    for costs, priorities, demand, slack, growth in product(
        product((2, 3), repeat=2),
        product((0, 2, 3), repeat=2),
        product((0, 1, 2), repeat=2),
        range(2),
        range(1, 4),
    ):
        current = {"requester": 2, "a": costs[0] * 2, "b": costs[1] * 2}
        total = sum(current.values())
        policy = _policy(
            _pool("requester", cost=1, minimum=0, maximum=8, priority=2),
            *(
                _pool(
                    name, cost=cost, minimum=cost, maximum=cost * 2, priority=priority
                )
                for name, cost, priority in zip(
                    ("a", "b"), costs, priorities, strict=True
                )
            ),
            floor=total - slack,
            ceiling=total,
        )
        desired = dict(zip(("a", "b"), demand, strict=True))
        expected = _exhaustive_growth_oracle(
            policy, current, "requester", 2 + growth, desired
        )
        result = plan_allocation(
            policy,
            current,
            "requester",
            2 + growth,
            desired_replicas_by_pool=desired,
        )
        assert (
            result.plan.allocations_gpus if result.admitted else None
        ) == expected, (
            costs,
            priorities,
            demand,
            slack,
            growth,
            result,
        )


def test_feasibility_matches_exhaustive_vectors_for_all_small_bands():
    for costs in product((2, 3, 4), repeat=2):
        ranges = [(1, 3, costs[0]), (0, 2, costs[1])]
        possible = {
            a * costs[0] + b * costs[1] for a, b in product(range(1, 4), range(3))
        }
        for lower in range(18):
            for upper in range(lower, 19):
                assert bounded_feasible_total(ranges, lower, upper) == any(
                    lower <= total <= upper for total in possible
                )


def test_feasibility_exhaustion_is_distinct_from_infeasibility():
    with pytest.raises(AllocationSearchLimit, match="unknown"):
        bounded_feasible_total(
            [(0, 10**30, 2)], 10**30 - 1, 10**30 - 1, max_search_steps=10
        )
    assert not bounded_feasible_total([(0, 2, 2)], 3, 3)
    assert bounded_feasible_total([(0, 2, 2)], 4, 4)
