# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public Planner configuration owned by the Dynamo AISimulate adapter."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PlannerConfig(BaseModel):
    """Concrete Planner input for ``aisimulate predict --stack dynamo``."""

    model_config = ConfigDict(extra="forbid")

    policy: Literal["disabled", "enabled"] = "disabled"
    target: Literal["throughput", "latency", "sla", "load"] = "throughput"
    enable_throughput_scaling: bool = True
    enable_load_scaling: bool = False
    throughput_adjustment_interval_seconds: int = Field(default=180, gt=0)
    load_adjustment_interval_seconds: int = Field(default=5, gt=0)
    max_num_fpm_samples: int = Field(default=64, gt=0)
    fpm_sample_bucket_size: int = Field(default=16, gt=0)
    load_scaling_down_sensitivity: int = Field(default=80, ge=0, le=100)
    load_min_observations: int = Field(default=5, gt=0)
    load_predictor: Literal["constant", "arima", "prophet", "kalman"] = "arima"
    load_predictor_log1p: bool = False
    prophet_window_size: int = Field(default=50, gt=0)
    kalman_q_level: float = Field(default=1.0, gt=0)
    kalman_q_trend: float = Field(default=0.1, gt=0)
    kalman_r: float = Field(default=10.0, gt=0)
    kalman_min_points: int = Field(default=5, gt=0)
    max_num_gpus: int = Field(default=8, gt=0)
    min_workers: int = Field(default=1, ge=0)
    prefill_min_workers: int | None = Field(default=None, gt=0)
    decode_min_workers: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate_fields(self) -> PlannerConfig:
        root = math.isqrt(self.fpm_sample_bucket_size)
        if root * root != self.fpm_sample_bucket_size:
            raise ValueError("fpm_sample_bucket_size must be a perfect square")
        if (
            self.enable_load_scaling
            and self.load_adjustment_interval_seconds
            >= self.throughput_adjustment_interval_seconds
        ):
            raise ValueError(
                "load adjustment interval must be shorter than throughput interval"
            )
        return self
