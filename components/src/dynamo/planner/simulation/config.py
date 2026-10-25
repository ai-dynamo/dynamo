# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prediction and recommendation config owned by the Dynamo Planner adapter."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Annotated, Any, Literal, TypeAlias

from aisimulate.config.common import Choices, IntegerRange, NumericRange
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from dynamo.aisimulate.domains import CHOICE_FORM, CHOICES, KNOB_FORM, RANGE, SCALAR
from dynamo.planner.config.planner_config import PlannerConfig

PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveFloat = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]
Policy = Literal["disabled", "enabled"]
Predictor = Literal["constant", "arima", "prophet", "kalman"]
PolicyDomain: TypeAlias = Annotated[
    Annotated[Policy, SCALAR] | Annotated[Choices[Policy], CHOICES], CHOICE_FORM
]
PredictorDomain: TypeAlias = Annotated[
    Annotated[Predictor, SCALAR] | Annotated[Choices[Predictor], CHOICES],
    CHOICE_FORM,
]
BoolDomain: TypeAlias = Annotated[
    Annotated[bool, SCALAR] | Annotated[Choices[bool], CHOICES], CHOICE_FORM
]

# Public planner knobs whose name is also the ``PlannerConfig`` field name.
PLANNER_PASSTHROUGH: tuple[str, ...] = (
    "enable_throughput_scaling",
    "enable_load_scaling",
    "throughput_adjustment_interval_seconds",
    "load_adjustment_interval_seconds",
    "max_throughput_scaling_replicas",
    "max_num_fpm_samples",
    "fpm_sample_bucket_size",
    "load_scaling_down_sensitivity",
    "load_min_observations",
    "load_predictor",
    "load_predictor_log1p",
    "prophet_window_size",
    "kalman_q_level",
    "kalman_q_trend",
    "kalman_r",
    "kalman_min_points",
)
# Public planner knob -> ``PlannerConfig`` field, for every knob a prediction
# materializes into the planner configuration.
PLANNER_PUBLIC_TO_CONFIG: dict[str, str] = {
    **{name: name for name in PLANNER_PASSTHROUGH},
    "target": "optimization_target",
    "max_num_gpus": "max_gpu_budget",
    "min_num_gpus": "min_gpu_budget",
    "min_workers": "min_endpoint",
    "prefill_min_workers": "prefill_min_endpoint",
    "decode_min_workers": "decode_min_endpoint",
}
# Preset group -> the independent knobs its ``preset`` stands for.
PLANNER_PRESET_GROUPS: dict[str, frozenset[str]] = {
    "scaling_policy": frozenset(
        {
            "enable_throughput_scaling",
            "enable_load_scaling",
            "throughput_adjustment_interval_seconds",
            "load_adjustment_interval_seconds",
        }
    ),
    "fpm_sampling": frozenset({"max_num_fpm_samples", "fpm_sample_bucket_size"}),
    "load_sensitivity": frozenset(
        {"load_scaling_down_sensitivity", "load_min_observations"}
    ),
    "load_predictor": frozenset(
        {
            "load_predictor_log1p",
            "prophet_window_size",
            "kalman_q_level",
            "kalman_q_trend",
            "kalman_r",
            "kalman_min_points",
        }
    ),
}


class ScalingPolicyMapping(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    enable_throughput_scaling: bool
    enable_load_scaling: bool
    throughput_adjustment_interval_seconds: PositiveInt | None
    load_adjustment_interval_seconds: PositiveInt | None

    @model_validator(mode="after")
    def _validate_intervals(self) -> ScalingPolicyMapping:
        scaling_enabled = self.enable_throughput_scaling or self.enable_load_scaling
        intervals = (
            self.throughput_adjustment_interval_seconds,
            self.load_adjustment_interval_seconds,
        )
        if not scaling_enabled:
            if any(interval is not None for interval in intervals):
                raise ValueError(
                    "disabled scaling policy requires null adjustment intervals"
                )
            return self
        return self


class FpmSamplingMapping(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    max_num_fpm_samples: PositiveInt
    fpm_sample_bucket_size: PositiveInt

    @model_validator(mode="after")
    def _validate_bucket(self) -> FpmSamplingMapping:
        root = math.isqrt(self.fpm_sample_bucket_size)
        if root * root != self.fpm_sample_bucket_size:
            raise ValueError("fpm_sample_bucket_size must be a perfect square")
        return self


class LoadSensitivityMapping(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    load_scaling_down_sensitivity: int = Field(strict=True, ge=0, le=100)
    load_min_observations: PositiveInt


class LoadPredictorMapping(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    load_predictor: Predictor
    load_predictor_log1p: bool
    prophet_window_size: PositiveInt
    kalman_q_level: PositiveFloat
    kalman_q_trend: PositiveFloat
    kalman_r: PositiveFloat
    kalman_min_points: PositiveInt


class PresetControl(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preset: Literal["default", False] | list[str | dict[str, Any]] | dict[
        str, Any
    ] = "default"
    type: PredictorDomain | None = None

    @model_validator(mode="after")
    def _validate_off_mapping(self) -> PresetControl:
        if isinstance(self.preset, dict) and self.preset:
            raise ValueError("preset mapping must be empty to disable it")
        if isinstance(self.preset, list) and not self.preset:
            raise ValueError("preset list must be nonempty")
        return self


def _validate_preset_mapping(
    model: type[BaseModel], group: str, entry: dict[str, Any]
) -> None:
    """Validate one preset mapping, reporting its errors against the entry.

    Raised from a model validator, the nested ``ValidationError`` would place
    them at the top level of the config.
    """

    try:
        model.model_validate(entry)
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(map(str, item['loc']))}: {item['msg']}"
            for item in error.errors(include_url=False)
        )
        raise ValueError(f"planner.{group}.preset entry {entry}: {problems}") from None


IntDomain: TypeAlias = Annotated[
    Annotated[PositiveInt, SCALAR]
    | Annotated[Choices[PositiveInt], CHOICES]
    | Annotated[IntegerRange, RANGE],
    KNOB_FORM,
]
NonNegativeIntDomain: TypeAlias = Annotated[
    Annotated[NonNegativeInt, SCALAR]
    | Annotated[Choices[NonNegativeInt], CHOICES]
    | Annotated[IntegerRange, RANGE],
    KNOB_FORM,
]
FloatDomain: TypeAlias = Annotated[
    Annotated[PositiveFloat, SCALAR]
    | Annotated[Choices[PositiveFloat], CHOICES]
    | Annotated[NumericRange, RANGE],
    KNOB_FORM,
]
OptionalIntDomain: TypeAlias = Annotated[
    Annotated[PositiveInt, SCALAR]
    | Annotated[Choices[PositiveInt | None], CHOICES]
    | Annotated[IntegerRange, RANGE],
    KNOB_FORM,
]


class PlannerRecommendationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scaling_policy: PresetControl | None = None
    fpm_sampling: PresetControl | None = None
    load_sensitivity: PresetControl | None = None
    load_predictor: PresetControl | None = None
    policy: PolicyDomain = Field(
        default_factory=lambda: Choices[Policy](choices=["disabled", "enabled"])
    )
    enable_throughput_scaling: BoolDomain | None = None
    enable_load_scaling: BoolDomain | None = None
    throughput_adjustment_interval_seconds: IntDomain | None = None
    load_adjustment_interval_seconds: IntDomain | None = None
    max_throughput_scaling_replicas: PositiveInt = PlannerConfig.model_fields[
        "max_throughput_scaling_replicas"
    ].default
    max_num_fpm_samples: IntDomain | None = None
    fpm_sample_bucket_size: IntDomain | None = None
    load_scaling_down_sensitivity: NonNegativeIntDomain | None = None
    load_min_observations: IntDomain | None = None
    load_predictor_log1p: BoolDomain | None = None
    prophet_window_size: IntDomain | None = None
    kalman_q_level: FloatDomain | None = None
    kalman_q_trend: FloatDomain | None = None
    kalman_r: FloatDomain | None = None
    kalman_min_points: IntDomain | None = None
    max_num_gpus: PositiveInt = PlannerConfig.model_fields["max_gpu_budget"].default
    min_workers: NonNegativeIntDomain | None = None
    prefill_min_workers: OptionalIntDomain | None = None
    decode_min_workers: OptionalIntDomain | None = None

    @model_validator(mode="after")
    def _validate_preset_conflicts(self) -> PlannerRecommendationConfig:
        for group, knobs in PLANNER_PRESET_GROUPS.items():
            control = getattr(self, group)
            if control is None:
                conflicts = sorted(
                    name for name in knobs if getattr(self, name) is not None
                )
                if conflicts:
                    raise ValueError(
                        f"planner.{group} uses the default preset and cannot be "
                        f"combined with independent knobs {conflicts}"
                    )
                continue
            if control.preset in (False, {}):
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
            if isinstance(control.preset, list):
                mapping_models: dict[str, type[BaseModel]] = {
                    "scaling_policy": ScalingPolicyMapping,
                    "fpm_sampling": FpmSamplingMapping,
                    "load_sensitivity": LoadSensitivityMapping,
                    "load_predictor": LoadPredictorMapping,
                }
                mapping_model = mapping_models[group]
                for entry in control.preset:
                    if isinstance(entry, dict):
                        _validate_preset_mapping(mapping_model, group, entry)
        sensitivity = self.load_scaling_down_sensitivity
        sensitivity_values = (
            sensitivity.choices if isinstance(sensitivity, Choices) else [sensitivity]
        )
        if isinstance(sensitivity, IntegerRange):
            sensitivity_values = [sensitivity.range.max]
        if any(value is not None and value > 100 for value in sensitivity_values):
            raise ValueError("load_scaling_down_sensitivity must be <= 100")
        bucket = self.fpm_sample_bucket_size
        bucket_values = bucket.choices if isinstance(bucket, Choices) else [bucket]
        if isinstance(bucket, IntegerRange):
            step = bucket.range.step or 1
            candidates = range(bucket.range.min, bucket.range.max + 1, step)
            bucket_values = [
                value for value in candidates if math.isqrt(value) ** 2 == value
            ]
            if not bucket_values:
                raise ValueError(
                    "fpm_sample_bucket_size range contains no perfect squares"
                )
        for value in bucket_values:
            if value is None:
                continue
            root = math.isqrt(value)
            if root * root != value:
                raise ValueError(
                    "fpm_sample_bucket_size values must be perfect squares"
                )
        return self


# Retune knob domains leave value bounds (for example ``max_gpu_budget: -1``)
# to the ``PlannerConfig`` validation of each expanded value.
StrictInt = Annotated[int, Field(strict=True)]
StrictFiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]
StrictIntDomain: TypeAlias = Annotated[
    Annotated[StrictInt, SCALAR]
    | Annotated[Choices[StrictInt], CHOICES]
    | Annotated[IntegerRange, RANGE],
    KNOB_FORM,
]
StrictFloatDomain: TypeAlias = Annotated[
    Annotated[StrictFiniteFloat, SCALAR]
    | Annotated[Choices[StrictFiniteFloat], CHOICES]
    | Annotated[NumericRange, RANGE],
    KNOB_FORM,
]

# Public knobs and groups fixed when the planner starts. A retune domain that
# names one is rejected; retune keeps the running value.
PLANNER_RETUNE_STARTUP_ONLY = frozenset(
    {
        "policy",
        "scaling_policy",
        "fpm_sampling",
        "enable_throughput_scaling",
        "enable_load_scaling",
        "throughput_adjustment_interval_seconds",
        "load_adjustment_interval_seconds",
        "max_num_fpm_samples",
        "fpm_sample_bucket_size",
        "load_min_observations",
    }
)
# Runtime ``PlannerConfig`` fields without an AISimulate public name. Retune
# accepts them under their own names.
PLANNER_RETUNE_EXTENSIONS: tuple[str, ...] = (
    "prefill_scale_up_queue_tokens",
    "prefill_scale_down_queue_tokens",
    "decode_scale_up_kv_rate",
    "decode_scale_down_kv_rate",
)
# Public retune grid knob -> ``PlannerConfig`` field. Retune searches these
# only when written; omitted, each keeps its current value.
PLANNER_RETUNE_KNOBS: dict[str, str] = {
    **{
        name: PLANNER_PUBLIC_TO_CONFIG[name]
        for name in (
            "min_workers",
            "prefill_min_workers",
            "decode_min_workers",
            "max_num_gpus",
            "min_num_gpus",
            "max_throughput_scaling_replicas",
        )
    },
    **{name: name for name in PLANNER_RETUNE_EXTENSIONS},
}
# Preset groups retune searches the runtime part of:
# ``load_scaling_down_sensitivity`` in the goodput grid, the load predictor in
# the forecast-loss pre-sweep.
PLANNER_RETUNE_GROUPS: tuple[str, ...] = ("load_sensitivity", "load_predictor")
_PLANNER_FIXED_SLA = ("ttft_ms", "itl_ms")
_PLANNER_LIVE_NAME_HINTS = {
    live: public for public, live in PLANNER_RETUNE_KNOBS.items() if live != public
}


class PlannerRetuneConfig(PlannerRecommendationConfig):
    """``retune.planner``: the recommendation schema, restricted to what a
    running planner can change.

    Syntax, preset groups and the preset-conflict rules are the recommendation
    ones. ``PLANNER_RETUNE_STARTUP_ONLY`` knobs are rejected; ``max_num_gpus``
    and ``max_throughput_scaling_replicas``, scalars in recommend, accept a
    domain here, and like ``min_num_gpus`` and ``PLANNER_RETUNE_EXTENSIONS``
    keep their current value when omitted. The SLA (``ttft_ms``/``itl_ms``) is
    not a knob. Retune knobs do not accept ``null``, and role minimums do not
    list ``null`` (inherit ``min_workers``) yet.
    """

    policy: PolicyDomain | None = None
    # Recommend pins these two to one value; retune searches them, so the
    # override widens the field type on purpose.
    max_num_gpus: StrictIntDomain | None = None  # type: ignore[assignment]
    min_num_gpus: StrictIntDomain | None = None
    max_throughput_scaling_replicas: IntDomain | None = None  # type: ignore[assignment]
    prefill_scale_up_queue_tokens: StrictIntDomain | None = None
    prefill_scale_down_queue_tokens: StrictIntDomain | None = None
    decode_scale_up_kv_rate: StrictFloatDomain | None = None
    decode_scale_down_kv_rate: StrictFloatDomain | None = None

    @model_validator(mode="before")
    @classmethod
    def _reject_fixed_knobs_and_live_names(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        for name in _PLANNER_FIXED_SLA:
            if name in value:
                raise ValueError(
                    f"retune.planner.{name} is the planner SLA and stays fixed "
                    f"during retune; set retune.sla_{name} to change the SLA "
                    "that retune scores against"
                )
        for live, public in _PLANNER_LIVE_NAME_HINTS.items():
            if live in value:
                raise ValueError(
                    f"retune.planner.{live} is not a public planner name; "
                    f"write {public} instead"
                )
        startup = sorted(PLANNER_RETUNE_STARTUP_ONLY.intersection(value))
        if startup:
            raise ValueError(
                f"retune.planner cannot change {startup}: they are fixed when "
                "the planner starts, and retune keeps their running values"
            )
        return value

    @model_validator(mode="after")
    def _reject_null_knob_values(self) -> PlannerRetuneConfig:
        for name in PLANNER_RETUNE_KNOBS:
            if name in self.model_fields_set and getattr(self, name) is None:
                raise ValueError(
                    f"retune.planner.{name} does not accept null; omit it to "
                    "keep the current value"
                )
        for name in ("prefill_min_workers", "decode_min_workers"):
            domain = getattr(self, name)
            if isinstance(domain, Choices) and None in domain.choices:
                raise ValueError(
                    f"retune.planner.{name} does not accept null (inherit "
                    "min_workers) yet; list explicit worker counts"
                )
        return self
