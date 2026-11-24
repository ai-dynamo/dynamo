# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit policies for fixed-size, single-role capacity allocation.

Bounds apply to committed allocations and settled targets. This pure policy
model does not discover workers, change replicas, or establish physical GPU
release. A deployment must belong to one pool so future execution cannot rely
on independently scaling components behind a shared deployment readiness check.
"""

from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from dynamo.global_planner.allocation import bounded_feasible_total


class PoolAllocationPolicy(BaseModel):
    """One explicitly identified worker pool with a fixed GPU cost per replica."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pool_id: StrictStr
    participant_id: StrictStr = Field(
        description="Exact capacity backend deployment identity, e.g. 'prod/chat'."
    )
    component_name: StrictStr = Field(
        description="Exact worker component owned by this pool."
    )
    role: Literal["prefill", "decode"]
    gpu_per_replica: StrictInt = Field(gt=0)
    min_gpus: StrictInt = Field(ge=0)
    max_gpus: StrictInt = Field(ge=0)
    priority: StrictInt = Field(ge=0, description="Higher is more important.")

    @field_validator("pool_id", "participant_id", "component_name")
    @classmethod
    def _validate_identity(cls, value: str) -> str:
        if any(not segment for segment in value.split("/")):
            raise ValueError("identity must have nonempty path segments")
        if any(char.isspace() or char in "*?[]" for char in value):
            raise ValueError(
                "identity must be explicit, without whitespace or wildcards"
            )
        return value

    @field_validator("component_name")
    @classmethod
    def _validate_component_name(cls, value: str) -> str:
        if "/" in value:
            raise ValueError("component_name must be a single path segment")
        return value

    @model_validator(mode="after")
    def _validate_replica_interval(self) -> PoolAllocationPolicy:
        if self.min_gpus > self.max_gpus:
            raise ValueError("min_gpus must not exceed max_gpus")
        if self.min_replicas > self.max_replicas:
            raise ValueError(
                f"pool {self.pool_id!r} has no whole-replica allocation within "
                f"[{self.min_gpus}, {self.max_gpus}] GPUs at "
                f"{self.gpu_per_replica} GPUs per replica"
            )
        return self

    @property
    def min_replicas(self) -> int:
        """Smallest whole-replica allocation meeting the configured minimum."""
        return (self.min_gpus + self.gpu_per_replica - 1) // self.gpu_per_replica

    @property
    def max_replicas(self) -> int:
        """Largest whole-replica allocation within the configured maximum."""
        return self.max_gpus // self.gpu_per_replica

    @property
    def effective_min_gpus(self) -> int:
        return self.min_replicas * self.gpu_per_replica

    @property
    def effective_max_gpus(self) -> int:
        return self.max_replicas * self.gpu_per_replica

    def normalize_replicas(self, desired_replicas: int) -> int:
        """Return the effective demand, clamped to the feasible pool interval."""
        if type(desired_replicas) is not int or desired_replicas < 0:
            raise ValueError("desired_replicas must be a nonnegative integer")
        return min(self.max_replicas, max(self.min_replicas, desired_replicas))


class AllocationPolicy(BaseModel):
    """Disjoint pools whose exact shared bounds admit a whole-replica vector.

    Feasibility is bounded: a configuration that exceeds the search budget is
    rejected as unverified, rather than accepted using only scalar sum checks.
    Pool policies and their sequence are immutable after validation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_total_gpus: StrictInt = Field(ge=0)
    max_total_gpus: StrictInt = Field(ge=0)
    pools: tuple[PoolAllocationPolicy, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_ownership_and_budget(self) -> AllocationPolicy:
        if self.min_total_gpus > self.max_total_gpus:
            raise ValueError("min_total_gpus must not exceed max_total_gpus")

        # One pool per participant also makes worker ownership exclusive.
        pool_ids: set[str] = set()
        participants: set[str] = set()
        for pool in self.pools:
            if pool.pool_id in pool_ids:
                raise ValueError(f"duplicate pool_id {pool.pool_id!r}")
            pool_ids.add(pool.pool_id)
            if pool.participant_id in participants:
                raise ValueError(
                    "only one pool per participant/DGD is supported: "
                    f"{pool.participant_id!r} owns multiple pools"
                )
            participants.add(pool.participant_id)

        ranges = [
            (pool.min_replicas, pool.max_replicas, pool.gpu_per_replica)
            for pool in self.pools
        ]
        if not bounded_feasible_total(ranges, self.min_total_gpus, self.max_total_gpus):
            raise ValueError(
                "no feasible whole-replica allocation satisfies the global GPU bounds"
            )
        return self
