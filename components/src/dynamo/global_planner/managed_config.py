# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in deployment contract for live managed allocation."""

from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator

from dynamo.global_planner.allocation_policy import AllocationPolicy


class ManagedAllocationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy: AllocationPolicy
    caller_namespaces: dict[StrictStr, StrictStr] = Field(
        description="Exact pool ID to authorized Local Planner Dynamo namespace."
    )
    poll_interval_seconds: float = Field(default=2.0, gt=0, allow_inf_nan=False)
    operation_timeout_seconds: float = Field(default=1800.0, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _validate_live_contract(self) -> "ManagedAllocationConfig":
        if self.caller_namespaces.keys() != {p.pool_id for p in self.policy.pools}:
            raise ValueError(
                "caller_namespaces must bind every configured pool exactly once"
            )
        if any(
            not value or value.strip() != value
            for value in self.caller_namespaces.values()
        ):
            raise ValueError("caller namespaces must be nonempty exact names")
        if any(p.min_replicas < 1 for p in self.policy.pools):
            raise ValueError(
                "live managed allocation requires a positive minimum for each pool"
            )
        if self.poll_interval_seconds >= self.operation_timeout_seconds:
            raise ValueError("poll interval must be shorter than the operation timeout")
        return self
