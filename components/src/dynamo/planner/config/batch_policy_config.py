# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated, dependency-light configuration for batch scheduling policy."""

from __future__ import annotations

import math
from dataclasses import dataclass


def _require_non_negative_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be non-negative and finite")


def _require_positive_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


@dataclass(frozen=True)
class BatchSchedulingPolicyConfig:
    """Static assumptions for the single-pool POC policy."""

    pool_id: str
    work_class: str
    safe_rps_per_ready_replica: float
    cold_start_margin_s: float
    finalization_margin_s: float
    max_observation_age_s: float
    drain_lease_duration_s: float
    min_replicas: int
    max_replicas: int
    scale_from_zero_replicas: int = 1
    max_batch_admission_rps: float | None = None

    def __post_init__(self) -> None:
        if not self.pool_id:
            raise ValueError("pool_id must not be empty")
        if not self.work_class:
            raise ValueError("work_class must not be empty")
        _require_positive_finite(
            "safe_rps_per_ready_replica", self.safe_rps_per_ready_replica
        )
        _require_non_negative_finite("cold_start_margin_s", self.cold_start_margin_s)
        _require_non_negative_finite(
            "finalization_margin_s", self.finalization_margin_s
        )
        _require_non_negative_finite(
            "max_observation_age_s", self.max_observation_age_s
        )
        _require_positive_finite("drain_lease_duration_s", self.drain_lease_duration_s)
        if self.min_replicas < 0:
            raise ValueError("min_replicas must be non-negative")
        if self.max_replicas < self.min_replicas:
            raise ValueError("max_replicas must be >= min_replicas")
        if self.scale_from_zero_replicas <= 0:
            raise ValueError("scale_from_zero_replicas must be positive")
        if self.scale_from_zero_replicas > self.max_replicas:
            raise ValueError("scale_from_zero_replicas must be <= max_replicas")
        if self.max_batch_admission_rps is not None:
            _require_non_negative_finite(
                "max_batch_admission_rps", self.max_batch_admission_rps
            )


__all__ = ["BatchSchedulingPolicyConfig"]
