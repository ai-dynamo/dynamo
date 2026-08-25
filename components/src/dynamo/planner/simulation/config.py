# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prediction and recommendation config owned by the Dynamo Planner adapter."""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal

from aisimulate.config.common import Choices, IntegerRange, NumericRange
from pydantic import BaseModel, ConfigDict, Field, model_validator

PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveFloat = Annotated[float, Field(gt=0)]
Policy = Literal["disabled", "enabled"]
Target = Literal["throughput", "latency", "sla", "load"]
Predictor = Literal["constant", "arima", "prophet", "kalman"]


class PlannerPredictionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy: Policy = "disabled"
    target: Target = "throughput"
    enable_throughput_scaling: bool = True
    enable_load_scaling: bool = False
    throughput_adjustment_interval_seconds: PositiveInt = 180
    load_adjustment_interval_seconds: PositiveInt = 5
    max_num_fpm_samples: PositiveInt = 64
    fpm_sample_bucket_size: PositiveInt = 16
    load_scaling_down_sensitivity: int = Field(default=80, ge=0, le=100)
    load_min_observations: PositiveInt = 5
    load_predictor: Predictor = "arima"
    load_predictor_log1p: bool = False
    prophet_window_size: PositiveInt = 50
    kalman_q_level: PositiveFloat = 1.0
    kalman_q_trend: PositiveFloat = 0.1
    kalman_r: PositiveFloat = 10.0
    kalman_min_points: PositiveInt = 5
    max_num_gpus: PositiveInt = 8
    min_workers: NonNegativeInt = 1
    prefill_min_workers: PositiveInt | None = None
    decode_min_workers: PositiveInt | None = None

    @model_validator(mode="after")
    def _validate_fields(self) -> PlannerPredictionConfig:
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


class PresetControl(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preset: Literal["default", False] | list[str | dict[str, Any]] | dict[
        str, Any
    ] = "default"
    type: Predictor | Choices[Predictor] | None = None

    @model_validator(mode="after")
    def _validate_off_mapping(self) -> PresetControl:
        if isinstance(self.preset, dict) and self.preset:
            raise ValueError("preset mapping must be empty to disable it")
        if isinstance(self.preset, list) and not self.preset:
            raise ValueError("preset list must be nonempty")
        return self


IntDomain = PositiveInt | Choices[PositiveInt] | IntegerRange
NonNegativeIntDomain = NonNegativeInt | Choices[NonNegativeInt] | IntegerRange
FloatDomain = PositiveFloat | Choices[PositiveFloat] | NumericRange


class PlannerRecommendationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scaling_policy: PresetControl | None = None
    fpm_sampling: PresetControl | None = None
    load_sensitivity: PresetControl | None = None
    load_predictor: PresetControl | None = None
    policy: Policy | Choices[Policy] = Field(
        default_factory=lambda: Choices[Policy](choices=["disabled", "enabled"])
    )
    target: Target | None = None
    enable_throughput_scaling: bool | Choices[bool] | None = None
    enable_load_scaling: bool | Choices[bool] | None = None
    throughput_adjustment_interval_seconds: IntDomain | None = None
    load_adjustment_interval_seconds: IntDomain | None = None
    max_num_fpm_samples: IntDomain | None = None
    fpm_sample_bucket_size: IntDomain | None = None
    load_scaling_down_sensitivity: NonNegativeIntDomain | None = None
    load_min_observations: IntDomain | None = None
    load_predictor_log1p: bool | Choices[bool] | None = None
    prophet_window_size: IntDomain | None = None
    kalman_q_level: FloatDomain | None = None
    kalman_q_trend: FloatDomain | None = None
    kalman_r: FloatDomain | None = None
    kalman_min_points: IntDomain | None = None
    max_num_gpus: PositiveInt = 8
    min_workers: NonNegativeIntDomain | None = None
    prefill_min_workers: IntDomain | None = None
    decode_min_workers: IntDomain | None = None

    @model_validator(mode="after")
    def _validate_preset_conflicts(self) -> PlannerRecommendationConfig:
        groups = {
            "scaling_policy": {
                "enable_throughput_scaling",
                "enable_load_scaling",
                "throughput_adjustment_interval_seconds",
                "load_adjustment_interval_seconds",
            },
            "fpm_sampling": {"max_num_fpm_samples", "fpm_sample_bucket_size"},
            "load_sensitivity": {
                "load_scaling_down_sensitivity",
                "load_min_observations",
            },
            "load_predictor": {
                "load_predictor_log1p",
                "prophet_window_size",
                "kalman_q_level",
                "kalman_q_trend",
                "kalman_r",
                "kalman_min_points",
            },
        }
        for group, knobs in groups.items():
            control = getattr(self, group)
            if control is None or control.preset in (False, {}):
                continue
            conflicts = sorted(
                name for name in knobs if getattr(self, name) is not None
            )
            if group == "load_predictor" and control.type is not None:
                conflicts.append("load_predictor.type")
            if conflicts:
                raise ValueError(
                    f"planner.{group}.preset cannot be combined with independent "
                    f"knobs {conflicts}"
                )
        return self
