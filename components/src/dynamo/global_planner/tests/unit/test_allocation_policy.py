# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation of explicit ownership and exact whole-replica budget intervals."""

import pytest
from pydantic import ValidationError

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


def _pool(**changes) -> PoolAllocationPolicy:
    fields = {
        "pool_id": "interactive",
        "participant_id": "prod/interactive",
        "component_name": "DecodeWorker",
        "role": "decode",
        "gpu_per_replica": 4,
        "min_gpus": 4,
        "max_gpus": 16,
        "priority": 100,
    }
    fields.update(changes)
    return PoolAllocationPolicy.model_validate(fields)


def test_normalizes_bounds_inward_to_whole_replicas():
    pool = _pool(min_gpus=5, max_gpus=15)
    assert (pool.min_replicas, pool.max_replicas) == (2, 3)
    assert (pool.effective_min_gpus, pool.effective_max_gpus) == (8, 12)
    assert (pool.min_gpus, pool.max_gpus) == (5, 15)


def test_bounds_use_exact_integer_arithmetic():
    floor = 2**60 + 1
    pool = _pool(min_gpus=floor, max_gpus=floor + 7)
    assert pool.effective_min_gpus == floor + 3
    assert pool.effective_max_gpus == floor + 7


@pytest.mark.parametrize("desired, expected", [(0, 2), (2, 2), (3, 3), (20, 3)])
def test_reports_effective_target_within_pool_bounds(desired, expected):
    assert _pool(min_gpus=5, max_gpus=15).normalize_replicas(desired) == expected


@pytest.mark.parametrize("value", [True, False, "2", 2.0, -1, None])
def test_rejects_noninteger_or_negative_replica_demand(value):
    with pytest.raises(ValueError, match="nonnegative integer"):
        _pool().normalize_replicas(value)


@pytest.mark.parametrize(
    "field", ["gpu_per_replica", "min_gpus", "max_gpus", "priority"]
)
@pytest.mark.parametrize("value", [True, False, "4", 4.0, -1])
def test_pool_numbers_are_strict_nonnegative_integers(field, value):
    with pytest.raises(ValidationError):
        _pool(**{field: value})


def test_zero_gpu_cost_is_not_a_managed_capacity_pool():
    with pytest.raises(ValidationError):
        _pool(gpu_per_replica=0)


def test_zero_bounds_allow_an_explicitly_disabled_pool():
    pool = _pool(min_gpus=0, max_gpus=0)
    assert pool.normalize_replicas(1) == 0
    assert AllocationPolicy(min_total_gpus=0, max_total_gpus=0, pools=[pool])


@pytest.mark.parametrize("minimum, maximum", [(8, 4), (5, 7), (1, 3)])
def test_rejects_empty_pool_intervals(minimum, maximum):
    with pytest.raises(ValidationError):
        _pool(min_gpus=minimum, max_gpus=maximum)


@pytest.mark.parametrize("field", ["pool_id", "participant_id", "component_name"])
@pytest.mark.parametrize(
    "value",
    [
        "",
        "/prod",
        "prod/",
        "prod//chat",
        "*",
        "prod/?",
        "[ab]",
        "prod chat",
        b"prod",
        3,
    ],
)
def test_requires_explicit_nonempty_identity_segments(field, value):
    with pytest.raises(ValidationError):
        _pool(**{field: value})


def test_component_name_is_one_component():
    with pytest.raises(ValidationError, match="single path segment"):
        _pool(component_name="DecodeWorker/OtherWorker")


@pytest.mark.parametrize(
    "role", ["", "both", "prefill,decode", "frontend", "aggregated"]
)
def test_pool_has_exactly_one_supported_worker_role(role):
    with pytest.raises(ValidationError):
        _pool(role=role)


@pytest.mark.parametrize("field", ["min_total_gpus", "max_total_gpus"])
@pytest.mark.parametrize("value", [True, False, "8", 8.0, -1])
def test_global_bounds_are_strict_nonnegative_integers(field, value):
    fields = {"min_total_gpus": 0, "max_total_gpus": 16, "pools": [_pool()]}
    fields[field] = value
    with pytest.raises(ValidationError):
        AllocationPolicy.model_validate(fields)


def test_rejects_reversed_global_bounds():
    with pytest.raises(ValidationError, match="must not exceed"):
        AllocationPolicy(min_total_gpus=9, max_total_gpus=8, pools=[_pool()])


def test_requires_at_least_one_pool():
    with pytest.raises(ValidationError):
        AllocationPolicy(min_total_gpus=0, max_total_gpus=8, pools=[])


def test_rejects_duplicate_pool_ids():
    pools = [_pool(), _pool(participant_id="prod/second")]
    with pytest.raises(ValidationError, match="duplicate pool_id"):
        AllocationPolicy(min_total_gpus=0, max_total_gpus=32, pools=pools)


@pytest.mark.parametrize(
    "second",
    [
        # The same worker, even under a different role.
        {"role": "prefill"},
        # A different worker in the same DGD shares its readiness.
        {"component_name": "PrefillWorker", "role": "prefill"},
    ],
)
def test_requires_separate_deployments_for_execution_isolation(second):
    pools = [_pool(), _pool(pool_id="second", **second)]
    with pytest.raises(ValidationError, match="only one pool per participant/DGD"):
        AllocationPolicy(min_total_gpus=0, max_total_gpus=32, pools=pools)


def test_distinct_deployments_can_use_same_component_name():
    pools = [_pool(), _pool(pool_id="second", participant_id="prod/second")]
    policy = AllocationPolicy(min_total_gpus=8, max_total_gpus=16, pools=pools)
    assert len(policy.pools) == 2


def test_rejects_global_interval_in_gap_between_feasible_vectors():
    # Scalar min/max sums span [0, 10], but 4a + 6b cannot equal 5.
    pools = [
        _pool(min_gpus=0, max_gpus=4),
        _pool(
            pool_id="second",
            participant_id="prod/second",
            min_gpus=0,
            max_gpus=6,
            gpu_per_replica=6,
        ),
    ]
    with pytest.raises(ValidationError, match="no feasible whole-replica allocation"):
        AllocationPolicy(min_total_gpus=5, max_total_gpus=5, pools=pools)


def test_fixed_global_total_admits_actual_heterogeneous_vector():
    pools = [
        _pool(min_gpus=0, max_gpus=8),
        _pool(
            pool_id="second",
            participant_id="prod/second",
            min_gpus=0,
            max_gpus=6,
            gpu_per_replica=6,
        ),
    ]
    policy = AllocationPolicy(min_total_gpus=10, max_total_gpus=10, pools=pools)
    assert policy.min_total_gpus == policy.max_total_gpus == 10


def test_policy_feasibility_exhaustion_is_reported_as_unverified():
    pool = _pool(gpu_per_replica=2, min_gpus=0, max_gpus=10**30)
    with pytest.raises(ValidationError, match="unknown") as exc:
        AllocationPolicy(
            min_total_gpus=10**30 - 1,
            max_total_gpus=10**30 - 1,
            pools=[pool],
        )
    assert "no feasible whole-replica allocation" not in str(exc.value)


@pytest.mark.parametrize("minimum, maximum", [(0, 3), (17, 20)])
def test_rejects_global_bounds_outside_all_pool_allocations(minimum, maximum):
    with pytest.raises(ValidationError, match="no feasible whole-replica allocation"):
        AllocationPolicy(
            min_total_gpus=minimum, max_total_gpus=maximum, pools=[_pool()]
        )


def test_policy_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        _pool(selector="prod/*")
    with pytest.raises(ValidationError):
        AllocationPolicy(min_total_gpus=0, max_total_gpus=16, pools=[_pool()], typo=1)


def test_validated_policy_is_immutable_and_round_trips():
    policy = AllocationPolicy(min_total_gpus=4, max_total_gpus=16, pools=[_pool()])
    with pytest.raises(ValidationError, match="frozen"):
        policy.max_total_gpus = 1
    with pytest.raises(ValidationError, match="frozen"):
        policy.pools[0].max_gpus = 0
    assert isinstance(policy.pools, tuple)
    assert AllocationPolicy.model_validate_json(policy.model_dump_json()) == policy
