# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo Planner implementation of the Sweeper sweep-configuration-provider ABI."""

from __future__ import annotations

import gzip
import itertools
import json
import math
import warnings
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from enum import Enum
from typing import Any, cast

from aisimulate.config.common import Choices, IntegerRange, NumericRange
from aisimulate.config_adapter import (
    PredictionAdapterContext,
    RecommendationAdapterContext,
)
from aisimulate.sweeper.provider import (
    AdapterReplaySpec,
    AdapterSearchPlan,
    CandidateContext,
    InfeasibleCandidate,
    JSONValue,
    RuntimeHookSpec,
    SearchSpaceFragment,
    SweepContext,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from tqdm import tqdm  # type: ignore[import-untyped]

from .config import (
    PlannerPredictionConfig,
    PlannerRecommendationConfig,
    ScalingPolicyMapping,
)
from .load_predictor import (
    LOAD_PREDICTOR_PRESETS,
    complete_predictor_preset,
    predictor_fields,
    sweep_load_predictor,
)
from .presets import (
    FPM_SAMPLING,
    LOAD_SENSITIVITY,
    SCALING_POLICIES,
    fpm_fields,
    load_sensitivity_fields,
    scaling_fields,
)

_SCALING_KEYS = frozenset(
    {
        "enable_throughput_scaling",
        "enable_load_scaling",
        "throughput_adjustment_interval_seconds",
        "load_adjustment_interval_seconds",
    }
)
_FPM_KEYS = frozenset({"max_num_fpm_samples", "fpm_sample_bucket_size"})
_LOAD_KEYS = frozenset({"load_scaling_down_sensitivity", "load_min_observations"})
_PREDICTOR_KEYS = frozenset(
    {
        "load_predictor",
        "load_predictor_log1p",
        "prophet_window_size",
        "kalman_q_level",
        "kalman_q_trend",
        "kalman_r",
        "kalman_min_points",
    }
)
_PROVIDER_API_VERSION = 1
_PLANNER_HOOK_API_VERSION = 1
_PLANNER_PASSTHROUGH = (
    "enable_throughput_scaling",
    "enable_load_scaling",
    "throughput_adjustment_interval_seconds",
    "load_adjustment_interval_seconds",
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
_HOOK = RuntimeHookSpec(
    provider="dynamo.planner",
    kind="scaling_policy",
    api_version=_PLANNER_HOOK_API_VERSION,
)


def _default_scaling_policies() -> list[str | dict[str, Any]]:
    return list(SCALING_POLICIES)


def _default_fpm_sampling() -> list[str | dict[str, Any]]:
    return list(FPM_SAMPLING)


def _default_load_sensitivity() -> list[str | dict[str, Any]]:
    return list(LOAD_SENSITIVITY)


def _default_load_predictors() -> list[str | dict[str, Any]]:
    return list(LOAD_PREDICTOR_PRESETS)


def _public_values(value: Any, defaults: list[Any]) -> list[Any]:
    if value is None:
        return list(defaults)
    if isinstance(value, Mapping) and set(value) == {"choices"}:
        choices = value["choices"]
        if not isinstance(choices, list) or not choices:
            raise ValueError("Planner choices must be a nonempty list")
        if len({repr(choice) for choice in choices}) != len(choices):
            raise ValueError("Planner choices must contain unique values")
        return list(choices)
    if isinstance(value, Mapping) and set(value) == {"range"}:
        raw = value["range"]
        if not isinstance(raw, Mapping):
            raise ValueError("Planner range must be a mapping")
        unknown = set(raw) - {"min", "max", "step", "scale"}
        if unknown:
            raise ValueError(f"Planner range has unknown fields {sorted(unknown)}")
        if "min" not in raw or "max" not in raw:
            raise ValueError("Planner range requires min and max")
        minimum, maximum = raw["min"], raw["max"]
        if (
            isinstance(minimum, bool)
            or isinstance(maximum, bool)
            or not isinstance(minimum, (int, float))
            or not isinstance(maximum, (int, float))
            or not math.isfinite(float(minimum))
            or not math.isfinite(float(maximum))
            or minimum > maximum
        ):
            raise ValueError("Planner range requires finite numeric min <= max")
        step = raw.get("step")
        if raw.get("scale", "linear") == "log":
            if isinstance(minimum, int) and isinstance(maximum, int):
                return list(range(minimum, maximum + 1))
            return [minimum, maximum]
        if (
            step is None
            or isinstance(step, bool)
            or not isinstance(step, (int, float))
            or step <= 0
        ):
            raise ValueError("Planner independent numeric ranges require step")
        values = []
        current = minimum
        while current <= maximum:
            values.append(current)
            current += step
        return values
    if isinstance(value, Mapping):
        raise ValueError("Planner domains must contain exactly choices or range")
    return [value]


def _independent_preset_mappings(
    group: str,
    control: Mapping[str, Any],
    planner: Mapping[str, Any],
) -> list[dict[str, Any]]:
    specifications: dict[str, tuple[dict[str, Any], list[Any]]] = {
        "scaling_policy": (
            {
                "enable_throughput_scaling": planner.get("enable_throughput_scaling"),
                "enable_load_scaling": planner.get("enable_load_scaling"),
                "throughput_adjustment_interval_seconds": planner.get(
                    "throughput_adjustment_interval_seconds"
                ),
                "load_adjustment_interval_seconds": planner.get(
                    "load_adjustment_interval_seconds"
                ),
            },
            [[False, True], [False, True], [180, 600], [5, 10]],
        ),
        "fpm_sampling": (
            {
                "max_num_fpm_samples": planner.get("max_num_fpm_samples"),
                "fpm_sample_bucket_size": planner.get("fpm_sample_bucket_size"),
            },
            [[32, 64, 128], [4, 16, 64]],
        ),
        "load_sensitivity": (
            {
                "load_scaling_down_sensitivity": planner.get(
                    "load_scaling_down_sensitivity"
                ),
                "load_min_observations": planner.get("load_min_observations"),
            },
            [[70, 80, 90], [3, 5, 8]],
        ),
        "load_predictor": (
            {
                "load_predictor": control.get("type"),
                "load_predictor_log1p": planner.get("load_predictor_log1p"),
                "prophet_window_size": planner.get("prophet_window_size"),
                "kalman_q_level": planner.get("kalman_q_level"),
                "kalman_q_trend": planner.get("kalman_q_trend"),
                "kalman_r": planner.get("kalman_r"),
                "kalman_min_points": planner.get("kalman_min_points"),
            },
            [
                ["constant", "arima", "prophet", "kalman"],
                [False, True],
                [20, 50],
                [1.0, 10.0],
                [0.1, 1.0],
                [5.0, 10.0],
                [3, 5],
            ],
        ),
    }
    values, defaults = specifications[group]
    names = list(values)
    dimensions = [
        _public_values(values[name], defaults[index])
        for index, name in enumerate(names)
    ]
    mappings = [
        dict(zip(names, combination, strict=True))
        for combination in itertools.product(*dimensions)
    ]
    if group == "scaling_policy":
        mappings = [
            mapping
            for mapping in mappings
            if not mapping["enable_load_scaling"]
            or mapping["load_adjustment_interval_seconds"]
            < mapping["throughput_adjustment_interval_seconds"]
        ]
    elif group == "fpm_sampling":
        mappings = [
            mapping
            for mapping in mappings
            if math.isqrt(mapping["fpm_sample_bucket_size"]) ** 2
            == mapping["fpm_sample_bucket_size"]
        ]
    return mappings


def _validate_preset_entries(
    name: str,
    values: list[str | dict[str, Any]],
    presets: frozenset[str],
    allowed_keys: frozenset[str],
    required_keys: frozenset[str],
) -> None:
    if not values:
        raise ValueError(f"{name}.preset must list at least one choice")
    for value in values:
        if isinstance(value, str):
            if value not in presets:
                raise ValueError(
                    f"{name}.preset has invalid choice {value!r}; "
                    f"allowed: {sorted(presets)}"
                )
            continue
        unknown = set(value) - allowed_keys
        missing = required_keys - set(value)
        if unknown:
            raise ValueError(
                f"{name}.preset mapping has unknown keys {sorted(unknown)}; "
                f"allowed: {sorted(allowed_keys)}"
            )
        if missing:
            raise ValueError(
                f"{name}.preset mapping is missing required keys {sorted(missing)}"
            )


class ScalingPolicySearch(BaseModel):
    """Preset choices that cover every scaling-policy knob."""

    model_config = ConfigDict(extra="forbid")

    preset: list[str | dict[str, Any]] = Field(
        default_factory=_default_scaling_policies
    )

    @model_validator(mode="after")
    def _validate_preset(self) -> ScalingPolicySearch:
        _validate_preset_entries(
            "scaling_policy",
            self.preset,
            frozenset(SCALING_POLICIES),
            _SCALING_KEYS,
            _SCALING_KEYS,
        )
        return self


class FpmSamplingSearch(BaseModel):
    """Preset choices that cover every FPM-sampling knob."""

    model_config = ConfigDict(extra="forbid")

    preset: list[str | dict[str, Any]] = Field(default_factory=_default_fpm_sampling)

    @model_validator(mode="after")
    def _validate_preset(self) -> FpmSamplingSearch:
        _validate_preset_entries(
            "fpm_sampling",
            self.preset,
            frozenset(FPM_SAMPLING),
            _FPM_KEYS,
            _FPM_KEYS,
        )
        return self


class LoadSensitivitySearch(BaseModel):
    """Preset choices that cover every load-sensitivity knob."""

    model_config = ConfigDict(extra="forbid")

    preset: list[str | dict[str, Any]] = Field(
        default_factory=_default_load_sensitivity
    )

    @model_validator(mode="after")
    def _validate_preset(self) -> LoadSensitivitySearch:
        _validate_preset_entries(
            "load_sensitivity",
            self.preset,
            frozenset(LOAD_SENSITIVITY),
            _LOAD_KEYS,
            _LOAD_KEYS,
        )
        return self


class LoadPredictorSearch(BaseModel):
    """Preset choices that cover every load-predictor knob."""

    model_config = ConfigDict(extra="forbid")

    preset: list[str | dict[str, Any]] = Field(default_factory=_default_load_predictors)

    @field_validator("preset", mode="after")
    @classmethod
    def _complete_custom_presets(
        cls, values: list[str | dict[str, Any]]
    ) -> list[str | dict[str, Any]]:
        del cls
        completed: list[str | dict[str, Any]] = []
        for value in values:
            if isinstance(value, str):
                completed.append(value)
                continue
            unknown = set(value) - _PREDICTOR_KEYS
            if unknown:
                raise ValueError(
                    "load_predictor.preset mapping has unknown keys "
                    f"{sorted(unknown)}; allowed: {sorted(_PREDICTOR_KEYS)}"
                )
            if "load_predictor" not in value:
                raise ValueError(
                    "load_predictor.preset mapping is missing required key "
                    "'load_predictor'"
                )
            completed.append(complete_predictor_preset(value))
        return completed

    @model_validator(mode="after")
    def _validate_preset(self) -> LoadPredictorSearch:
        _validate_preset_entries(
            "load_predictor",
            self.preset,
            frozenset(LOAD_PREDICTOR_PRESETS),
            _PREDICTOR_KEYS,
            _PREDICTOR_KEYS,
        )
        return self


_LEGACY_PRESET_REMOVAL_NOTE = (
    "Legacy flat Dynamo Planner Sweeper preset fields are deprecated and will be "
    "removed after the 1.5 release. Nest choices under each sub-item's 'preset' field."
)


class PlannerSearchSpace(BaseModel):
    """Validated Planner-owned search space."""

    model_config = ConfigDict(extra="forbid")

    scaling_policy: ScalingPolicySearch = Field(default_factory=ScalingPolicySearch)
    fpm_sampling: FpmSamplingSearch = Field(default_factory=FpmSamplingSearch)
    load_sensitivity: LoadSensitivitySearch = Field(
        default_factory=LoadSensitivitySearch
    )
    load_predictor: LoadPredictorSearch = Field(default_factory=LoadPredictorSearch)
    min_endpoint: int | None = Field(default=None, ge=1)
    prefill_min_endpoint: int | None = Field(default=None, ge=1)
    decode_min_endpoint: int | None = Field(default=None, ge=1)
    # ``None`` preserves the legacy provider behavior of inheriting the
    # candidate's GPU budget. The public schema always supplies its default 8.
    max_num_gpus: int | None = Field(default=None, ge=1)
    public_schema: bool = False
    public_policy: list[str] | None = None
    planner_target: str | None = None
    public_min_workers: list[int] | None = None
    public_prefill_min_workers: list[int | None] | None = None
    public_decode_min_workers: list[int | None] | None = None

    @model_validator(mode="before")
    @classmethod
    def _upgrade_legacy_flat_presets(cls, data: Any) -> Any:
        # Backward compatibility: remove this conversion after the 1.5 release.
        if not isinstance(data, Mapping):
            return data

        upgraded = dict(data)
        if "public_schema" not in upgraded:
            public_keys = {
                "policy",
                "target",
                "max_num_gpus",
                "min_workers",
                "prefill_min_workers",
                "decode_min_workers",
            }
            public_schema = bool(public_keys.intersection(upgraded))
            if public_schema:
                upgraded["public_schema"] = True
                upgraded.setdefault("max_num_gpus", 8)
            public_policy = upgraded.pop("policy", None)
            if public_schema:
                upgraded["public_policy"] = _public_values(
                    public_policy, ["disabled", "enabled"]
                )
            public_target = upgraded.pop("target", None)
            if public_target is not None:
                upgraded["planner_target"] = public_target
            for public_name, normalized_name, defaults in (
                ("min_workers", "public_min_workers", [1]),
                ("prefill_min_workers", "public_prefill_min_workers", [None]),
                ("decode_min_workers", "public_decode_min_workers", [None]),
            ):
                if public_name in upgraded:
                    upgraded[normalized_name] = _public_values(
                        upgraded.pop(public_name), defaults
                    )
        if upgraded.get("public_schema") is True:
            predictor_control = upgraded.get("load_predictor")
            if isinstance(predictor_control, Mapping):
                preset = predictor_control.get("preset")
                if isinstance(preset, list):
                    for entry in preset:
                        if not isinstance(entry, Mapping):
                            continue
                        missing = _PREDICTOR_KEYS - set(entry)
                        if missing:
                            raise ValueError(
                                "load_predictor.preset mapping is missing required "
                                f"keys {sorted(missing)}"
                            )
        for group in (
            "scaling_policy",
            "fpm_sampling",
            "load_sensitivity",
            "load_predictor",
        ):
            control = upgraded.get(group)
            if not isinstance(control, Mapping) or "preset" not in control:
                continue
            preset = control["preset"]
            if preset == "default":
                upgraded.pop(group)
            elif preset is False or preset == {}:
                upgraded[group] = {
                    "preset": _independent_preset_mappings(group, control, upgraded)
                }
        for public_knob in (
            _SCALING_KEYS
            | _FPM_KEYS
            | _LOAD_KEYS
            | (_PREDICTOR_KEYS - {"load_predictor"})
        ):
            upgraded.pop(public_knob, None)
        legacy_fields: list[str] = []
        for name in ("scaling_policy", "fpm_sampling", "load_sensitivity"):
            value = upgraded.get(name)
            if value is not None and not isinstance(value, (Mapping, BaseModel)):
                upgraded[name] = {"preset": value}
                legacy_fields.append(name)

        if "load_predictor_candidates" in upgraded:
            if "load_predictor" in upgraded:
                raise ValueError(
                    "load_predictor_candidates cannot be combined with load_predictor"
                )
            upgraded["load_predictor"] = {
                "preset": upgraded.pop("load_predictor_candidates")
            }
            legacy_fields.append("load_predictor_candidates")

        if legacy_fields:
            warnings.warn(
                f"{_LEGACY_PRESET_REMOVAL_NOTE} Legacy fields: "
                f"{', '.join(legacy_fields)}.",
                FutureWarning,
                stacklevel=4,
            )
        return upgraded

    @model_validator(mode="after")
    def _validate_public_fields(self) -> PlannerSearchSpace:
        if self.public_policy is not None:
            invalid = sorted(set(self.public_policy) - {"disabled", "enabled"})
            if invalid:
                raise ValueError(
                    f"planner.policy choices must be disabled or enabled; got {invalid}"
                )
        if self.planner_target not in (None, "throughput", "latency", "sla", "load"):
            raise ValueError("planner.target must be throughput, latency, sla, or load")
        domains = (
            ("min_workers", self.public_min_workers, 0),
            ("prefill_min_workers", self.public_prefill_min_workers, 1),
            ("decode_min_workers", self.public_decode_min_workers, 1),
        )
        for name, values, minimum in domains:
            if values is None:
                continue
            if not values:
                raise ValueError(f"planner.{name} choices cannot be empty")
            invalid = [
                value for value in values if value is not None and value < minimum
            ]
            if invalid:
                raise ValueError(
                    f"planner.{name} choices must be >= {minimum}; got {invalid}"
                )
        return self


def _plain(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


def _dynamo_trace_is_agentic(traffic: Mapping[str, JSONValue]) -> bool:
    source = traffic.get("source")
    if not isinstance(source, Mapping):
        return False
    paths = source.get("paths")
    if not isinstance(paths, list):
        return False
    for raw_path in paths:
        path = str(raw_path)
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as trace_file:
            for line in trace_file:
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, Mapping):
                    continue
                event = record.get("event", record)
                if not isinstance(event, Mapping):
                    continue
                event_type = event.get("event_type", record.get("event_type"))
                if event_type != "request_end":
                    continue
                if event.get("agent_context") is not None:
                    return True
    return False


def _planner_optimization_target(goal: Mapping[str, JSONValue]) -> str:
    target = str(_plain(goal.get("target", "throughput")))
    if target == "pareto":
        raw_objectives = goal.get("pareto_objectives")
        objectives = (
            {str(_plain(objective)) for objective in raw_objectives}
            if isinstance(raw_objectives, list)
            else set()
        )
        if objectives.intersection({"goodput", "goodput_per_gpu"}):
            return "sla"
        return "throughput"
    return {
        "throughput": "throughput",
        "throughput_per_gpu": "throughput",
        "throughput_per_user": "throughput",
        "ttft": "latency",
        "e2e_latency": "latency",
        "goodput": "sla",
        "goodput_per_gpu": "sla",
    }[target]


def _policy_filter(
    policies: list[str | dict[str, Any]],
    *,
    optimization_target: str,
    sla: Mapping[str, JSONValue] | None,
) -> tuple[list[str | dict[str, Any]], list[str | dict[str, Any]]]:
    kept: list[str | dict[str, Any]] = []
    dropped: list[str | dict[str, Any]] = []
    for policy in policies:
        fields = scaling_fields(policy)
        uses_throughput = bool(fields["enable_throughput_scaling"])
        allowed = not uses_throughput or optimization_target == "sla"
        if (
            allowed
            and optimization_target == "sla"
            and sla is not None
            and (sla.get("ttft_ms") is None or sla.get("itl_ms") is None)
        ):
            allowed = not uses_throughput
        (kept if allowed else dropped).append(policy)
    return kept, dropped


def _deployment_modes(
    core_search_space: Mapping[str, JSONValue],
) -> list[str]:
    raw_modes = core_search_space.get("deployment_mode", ["agg", "disagg"])
    if not isinstance(raw_modes, list):
        raise TypeError("core deployment_mode must be a list")
    return [str(mode) for mode in raw_modes]


def _json_choices(values: list[str | dict[str, Any]]) -> list[JSONValue]:
    return cast(list[JSONValue], deepcopy(values))


def _int_value(values: Mapping[str, JSONValue], name: str) -> int:
    value = values[name]
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise TypeError(f"Planner candidate field {name} must be numeric")
    return int(value)


def _float_value(values: Mapping[str, JSONValue], name: str) -> float:
    value = values[name]
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise TypeError(f"Planner candidate field {name} must be numeric")
    return float(value)


class DynamoPlannerSweepConfigProvider:
    """Planner search-space preparation and replay-spec materialization."""

    name = "dynamo.planner"
    section = "planner"
    config_adapter_api_version = 3
    # Provider-owned constant: importing the consumer's current API_VERSION would
    # let an older wheel accidentally self-certify against a newer core package.
    api_version = _PROVIDER_API_VERSION

    def compile_prediction(
        self,
        config: Mapping[str, JSONValue],
        context: PredictionAdapterContext,
    ) -> AdapterReplaySpec:
        public = PlannerPredictionConfig.model_validate(config)
        if public.policy == "enabled":
            if not (public.enable_throughput_scaling or public.enable_load_scaling):
                raise ValueError(
                    "planner.policy=enabled requires at least one scaling mode"
                )
            source = context.traffic.get("source")
            trace_format = source.get("format") if isinstance(source, Mapping) else None
            if trace_format in {"mooncake-delta", "agentic_mooncake"} or (
                trace_format == "dynamo" and _dynamo_trace_is_agentic(context.traffic)
            ):
                raise ValueError(f"{trace_format} requires planner.policy=disabled")
            if public.enable_throughput_scaling:
                raw_sla = context.evaluation.get("sla")
                if (
                    public.target != "sla"
                    or not isinstance(raw_sla, Mapping)
                    or raw_sla.get("ttft_ms") is None
                    or raw_sla.get("itl_ms") is None
                ):
                    raise ValueError(
                        "Planner throughput scaling requires target='sla' and "
                        "evaluation.sla.ttft_ms/itl_ms"
                    )
        else:
            return AdapterReplaySpec(config={"policy": "disabled"})
        concrete = public.model_dump(mode="json", exclude_none=True)
        sample = _prediction_sample(context.engine)
        raw_sla = context.evaluation.get("sla")
        sla = raw_sla if isinstance(raw_sla, Mapping) else None
        planner_config = _planner_config_payload(
            concrete,
            sample=sample,
            optimization_target=public.target,
            sla=sla,
            min_endpoint=public.min_workers,
            prefill_min_endpoint=public.prefill_min_workers,
            decode_min_endpoint=public.decode_min_workers,
            max_num_gpus=public.max_num_gpus,
        )
        public_config: dict[str, JSONValue] = {
            "policy": "enabled",
            "target": public.target,
            "enable_throughput_scaling": public.enable_throughput_scaling,
            "enable_load_scaling": public.enable_load_scaling,
            "throughput_adjustment_interval_seconds": public.throughput_adjustment_interval_seconds,
            "load_adjustment_interval_seconds": public.load_adjustment_interval_seconds,
            "max_num_gpus": public.max_num_gpus,
            "min_workers": public.min_workers,
        }
        if public.prefill_min_workers is not None:
            public_config["prefill_min_workers"] = public.prefill_min_workers
        if public.decode_min_workers is not None:
            public_config["decode_min_workers"] = public.decode_min_workers
        if public.enable_throughput_scaling:
            public_config.update(
                max_num_fpm_samples=public.max_num_fpm_samples,
                fpm_sample_bucket_size=public.fpm_sample_bucket_size,
                load_predictor=public.load_predictor,
                load_predictor_log1p=public.load_predictor_log1p,
                prophet_window_size=public.prophet_window_size,
                kalman_q_level=public.kalman_q_level,
                kalman_q_trend=public.kalman_q_trend,
                kalman_r=public.kalman_r,
                kalman_min_points=public.kalman_min_points,
            )
        if public.enable_load_scaling:
            public_config.update(
                load_scaling_down_sensitivity=public.load_scaling_down_sensitivity,
                load_min_observations=public.load_min_observations,
            )
        return AdapterReplaySpec(
            config=public_config,
            runtime_hooks=(
                RuntimeHookSpec(
                    provider=_HOOK.provider,
                    kind=_HOOK.kind,
                    api_version=_HOOK.api_version,
                    config={"planner_config": planner_config},
                ),
            ),
        )

    def compile_recommendation(
        self,
        config: Mapping[str, JSONValue],
        context: RecommendationAdapterContext,
    ) -> AdapterSearchPlan:
        public = PlannerRecommendationConfig.model_validate(config)
        policies = (
            set(public.policy.choices)
            if isinstance(public.policy, Choices)
            else {public.policy}
        )
        if policies == {"disabled"}:
            modes = _deployment_modes(context.sweep.core_search_space)
            return AdapterSearchPlan(
                fragment=SearchSpaceFragment(
                    choices_by_branch={mode: {"policy": ["disabled"]} for mode in modes}
                ),
                state={"forced_disabled": True},
            )
        source = context.traffic.get("source")
        trace_format = source.get("format") if isinstance(source, Mapping) else None
        if "enabled" in policies and (
            trace_format in {"mooncake-delta", "agentic_mooncake"}
            or (trace_format == "dynamo" and _dynamo_trace_is_agentic(context.traffic))
        ):
            raise ValueError(f"{trace_format} requires planner.policy=disabled")
        normalized = public.model_dump(mode="python", exclude_none=True)
        normalized_space = PlannerSearchSpace.model_validate(normalized)
        plan = self.generate_search_space(normalized, context.sweep)
        independent_groups = [
            group
            for group in ("scaling_policy", "fpm_sampling", "load_sensitivity")
            if (control := getattr(public, group)) is not None
            and control.preset in (False, {})
        ]
        if not independent_groups:
            return plan
        choices_by_branch = deepcopy(plan.fragment.choices_by_branch)
        log_discrete_by_branch: dict[str, list[str]] = {
            branch: [] for branch in choices_by_branch
        }
        for branch, branch_choices in choices_by_branch.items():
            for group in independent_groups:
                branch_choices.pop(group, None)
                mappings = getattr(normalized_space, group).preset
                keys = (
                    list(mappings[0])
                    if mappings and isinstance(mappings[0], dict)
                    else []
                )
                for key in keys:
                    values = []
                    for mapping in mappings:
                        assert isinstance(mapping, dict)
                        value = mapping[key]
                        if value not in values:
                            values.append(value)
                    branch_choices[key] = values
                    domain = getattr(public, key, None)
                    if (
                        isinstance(domain, (IntegerRange, NumericRange))
                        and domain.range.scale == "log"
                    ):
                        log_discrete_by_branch[branch].append(key)
        state = deepcopy(plan.state)
        assert isinstance(state, dict)
        state["independent_groups"] = independent_groups
        return replace(
            plan,
            fragment=replace(
                plan.fragment,
                choices_by_branch=choices_by_branch,
                log_discrete_choices_by_branch=log_discrete_by_branch,
            ),
            state=state,
        )

    def materialize_candidate(
        self,
        plan: AdapterSearchPlan,
        selection: Mapping[str, JSONValue],
        context: CandidateContext,
    ) -> AdapterReplaySpec:
        if isinstance(plan.state, dict) and plan.state.get("forced_disabled") is True:
            return AdapterReplaySpec(config={"policy": "disabled"})
        materialized = dict(selection)
        if isinstance(plan.state, dict):
            raw_space = plan.state.get("search_space")
            if isinstance(raw_space, dict):
                space = PlannerSearchSpace.model_validate(raw_space)
                for group in plan.state.get("independent_groups", []):
                    mappings = getattr(space, group).preset
                    keys = (
                        list(mappings[0])
                        if mappings and isinstance(mappings[0], dict)
                        else []
                    )
                    group_mapping = {key: materialized.pop(key) for key in keys}
                    materialized[group] = group_mapping
                    if group == "scaling_policy" and (
                        group_mapping["enable_throughput_scaling"]
                        or group_mapping["enable_load_scaling"]
                    ):
                        try:
                            ScalingPolicyMapping.model_validate(group_mapping)
                        except ValueError as exc:
                            raise InfeasibleCandidate(
                                f"independent Planner scaling-policy selection: {exc}"
                            ) from exc
        return self.materialize_replay(plan, materialized, context)

    def generate_search_space(
        self,
        search_spec: Mapping[str, JSONValue],
        context: SweepContext,
    ) -> AdapterSearchPlan:
        space = PlannerSearchSpace.model_validate(search_spec)
        optimization_target = _planner_optimization_target(context.goal)
        raw_sla = context.goal.get("sla")
        sla = raw_sla if isinstance(raw_sla, Mapping) else None
        kept, dropped = _policy_filter(
            space.scaling_policy.preset,
            optimization_target=optimization_target,
            sla=sla,
        )
        if dropped and space.public_schema:
            raise ValueError(
                "Planner scaling_policy contains throughput-scaling choices that "
                "are incompatible with the selected optimization target/SLA: "
                f"{dropped}"
            )
        if dropped and context.show_progress:
            if optimization_target != "sla":
                target = str(_plain(context.goal.get("target", "throughput")))
                tqdm.write(
                    f"smart-sweep: dropped {len(dropped)} throughput-scaling "
                    f"policy option(s) for target={target} (needs SLA): {dropped}"
                )
            else:
                tqdm.write(
                    f"smart-sweep: dropped {len(dropped)} planner-scaling policy "
                    f"option(s) for e2e-only SLA (planner needs ttft_ms+itl_ms): "
                    f"{dropped}"
                )
        if not kept:
            if optimization_target != "sla":
                raise ValueError(
                    "every Planner scaling_policy enables throughput scaling, "
                    "which requires a goodput target"
                )
            raise ValueError(
                "every Planner scaling_policy enables throughput scaling, but an "
                "e2e-only SLA cannot seed the Planner's TTFT/ITL scaling target"
            )

        trace_path = context.workload.get("trace_path")
        trace_format = context.workload.get("trace_format")
        raw_trace_paths = context.workload.get("trace_paths")
        trace_paths = (
            [str(path) for path in raw_trace_paths]
            if isinstance(raw_trace_paths, list)
            else None
        )
        predictor_result = sweep_load_predictor(
            policies=kept,
            candidates=space.load_predictor.preset,
            trace_path=str(trace_path) if trace_path is not None else None,
            trace_paths=trace_paths,
            trace_format=str(trace_format) if trace_format is not None else None,
            show_progress=context.show_progress,
        )
        scaling_possible = any(
            fields["enable_throughput_scaling"] or fields["enable_load_scaling"]
            for fields in (scaling_fields(policy) for policy in kept)
        )
        local_choices: dict[str, list[JSONValue]] = {
            "scaling_policy": _json_choices(kept)
        }
        if space.public_policy is not None:
            local_choices["policy"] = cast(list[JSONValue], list(space.public_policy))
        if scaling_possible:
            local_choices["fpm_sampling"] = _json_choices(space.fpm_sampling.preset)
            local_choices["load_sensitivity"] = _json_choices(
                space.load_sensitivity.preset
            )
        choices_by_branch: dict[str, dict[str, list[JSONValue]]] = {}
        for mode in _deployment_modes(context.core_search_space):
            branch_choices = deepcopy(local_choices)
            if space.public_min_workers is not None:
                branch_choices["min_workers"] = list(space.public_min_workers)
            if mode == "disagg" and space.public_prefill_min_workers is not None:
                branch_choices["prefill_min_workers"] = list(
                    space.public_prefill_min_workers
                )
            if mode == "disagg" and space.public_decode_min_workers is not None:
                branch_choices["decode_min_workers"] = list(
                    space.public_decode_min_workers
                )
            choices_by_branch[mode] = branch_choices
        fragment = SearchSpaceFragment(choices_by_branch=choices_by_branch)
        state: dict[str, JSONValue] = {
            "search_space": space.model_dump(mode="json"),
            "optimization_target": optimization_target,
            "sla": dict(sla) if sla is not None else None,
            "load_predictor": predictor_result.to_state(),
        }
        diagnostics: dict[str, JSONValue] = {
            "dropped_scaling_policies": _json_choices(dropped),
            "load_predictor_reason": predictor_result.reason,
            "load_predictor_losses": predictor_result.to_state()["losses"],
        }
        return AdapterSearchPlan(
            fragment=fragment,
            state=state,
            diagnostics=diagnostics,
            potential_runtime_hooks=(
                (_HOOK,)
                if scaling_possible
                or (
                    space.public_policy is not None and "enabled" in space.public_policy
                )
                else ()
            ),
        )

    def materialize_replay(
        self,
        plan: AdapterSearchPlan,
        selection: Mapping[str, JSONValue],
        context: CandidateContext,
    ) -> AdapterReplaySpec:
        if not isinstance(plan.state, dict):
            raise TypeError("Planner adapter search plan state must be a mapping")
        state = plan.state
        raw_space = state["search_space"]
        if not isinstance(raw_space, dict):
            raise TypeError("Planner adapter search-space state must be a mapping")
        space = PlannerSearchSpace.model_validate(raw_space)
        public_schema = space.public_schema

        scaling_entry = selection["scaling_policy"]
        if not isinstance(scaling_entry, (str, dict)):
            raise TypeError(
                "Planner scaling_policy selection must be a string or mapping"
            )
        scaling = scaling_fields(scaling_entry)
        scaling_enabled = bool(
            scaling["enable_throughput_scaling"] or scaling["enable_load_scaling"]
        )
        candidate_config: dict[str, JSONValue] = {
            "scaling_policy": deepcopy(scaling_entry),
            **scaling,
        }
        policy_enabled = (
            str(selection.get("policy")) == "enabled"
            if public_schema
            else scaling_enabled
        )
        if not policy_enabled or not scaling_enabled:
            return AdapterReplaySpec(
                config=({"policy": "disabled"} if public_schema else candidate_config)
            )

        if scaling["enable_throughput_scaling"]:
            fpm_entry = selection["fpm_sampling"]
            if not isinstance(fpm_entry, (str, dict)):
                raise TypeError("Planner FPM selection must be a string or mapping")
            candidate_config.update(fpm_fields(fpm_entry))
        if scaling["enable_load_scaling"]:
            load_entry = selection["load_sensitivity"]
            if not isinstance(load_entry, (str, dict)):
                raise TypeError(
                    "Planner load-sensitivity selection must be a string or mapping"
                )
            candidate_config.update(load_sensitivity_fields(load_entry))

        if scaling["enable_throughput_scaling"]:
            predictor_state = state["load_predictor"]
            if not isinstance(predictor_state, dict):
                raise TypeError("Planner load-predictor state must be a mapping")
            winners = predictor_state["best_by_interval"]
            if not isinstance(winners, dict):
                raise TypeError("Planner load-predictor winners must be a mapping")
            interval = scaling["throughput_adjustment_interval_seconds"]
            winner = winners.get(str(interval))
            if isinstance(winner, (str, dict)):
                candidate_config.update(predictor_fields(winner))

        min_endpoint = (
            _int_value(selection, "min_workers")
            if selection.get("min_workers") is not None
            else space.min_endpoint
        )
        prefill_min_endpoint = (
            _int_value(selection, "prefill_min_workers")
            if selection.get("prefill_min_workers") is not None
            else space.prefill_min_endpoint
        )
        decode_min_endpoint = (
            _int_value(selection, "decode_min_workers")
            if selection.get("decode_min_workers") is not None
            else space.decode_min_endpoint
        )
        planner_target = space.planner_target or str(state["optimization_target"])
        planner_config = _planner_config_payload(
            candidate_config,
            sample=context.sample,
            optimization_target=planner_target,
            sla=state["sla"] if isinstance(state["sla"], dict) else None,
            min_endpoint=min_endpoint,
            prefill_min_endpoint=prefill_min_endpoint,
            decode_min_endpoint=decode_min_endpoint,
            max_num_gpus=space.max_num_gpus,
        )
        hook = RuntimeHookSpec(
            provider=_HOOK.provider,
            kind=_HOOK.kind,
            api_version=_HOOK.api_version,
            config={"planner_config": planner_config},
        )
        if not public_schema:
            return AdapterReplaySpec(
                config={
                    "scaling_policy": deepcopy(scaling_entry),
                    **planner_config,
                },
                runtime_hooks=(hook,),
            )

        effective_max_num_gpus = (
            space.max_num_gpus
            if space.max_num_gpus is not None
            else _int_value(context.sample, "gpu_budget")
        )
        public_config: dict[str, JSONValue] = {
            "policy": "enabled",
            "target": planner_target,
            **{
                key: value
                for key, value in candidate_config.items()
                if key in _PLANNER_PASSTHROUGH
            },
            "max_num_gpus": effective_max_num_gpus,
            "min_workers": min_endpoint if min_endpoint is not None else 1,
        }
        if prefill_min_endpoint is not None:
            public_config["prefill_min_workers"] = prefill_min_endpoint
        if decode_min_endpoint is not None:
            public_config["decode_min_workers"] = decode_min_endpoint
        return AdapterReplaySpec(config=public_config, runtime_hooks=(hook,))


def _prediction_sample(engine: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    public_mode = str(engine.get("mode", "aggregated"))
    mode = "disagg" if public_mode == "disaggregated" else "agg"
    workers = engine.get("workers")
    if not isinstance(workers, Mapping):
        raise TypeError("Planner prediction requires engine.workers")
    sample: dict[str, JSONValue] = {"deployment_mode": mode}
    roles = ("prefill", "decode") if mode == "disagg" else ("aggregated",)
    for role in roles:
        worker = workers.get(role)
        if not isinstance(worker, Mapping):
            raise TypeError(f"Planner prediction requires engine.workers.{role}")
        parallel = worker.get("parallelism")
        if not isinstance(parallel, Mapping):
            raise TypeError(f"Planner prediction requires {role} parallelism")
        prefix = "" if role == "aggregated" else f"{role}_"
        sample[f"{prefix}tp"] = int(parallel.get("tensor", 1))
        sample[f"{prefix}attention_dp"] = int(parallel.get("attention_data", 1))
    return sample


def _planner_config_payload(
    candidate_config: Mapping[str, JSONValue],
    *,
    sample: Mapping[str, JSONValue],
    optimization_target: str,
    sla: Mapping[str, JSONValue] | None,
    min_endpoint: int | None,
    prefill_min_endpoint: int | None,
    decode_min_endpoint: int | None,
    max_num_gpus: int | None = None,
) -> dict[str, JSONValue]:
    """Build the exact PlannerConfig payload used by the pre-refactor Sweeper."""

    mode = str(sample["deployment_mode"])
    payload: dict[str, JSONValue] = {
        "mode": mode,
        "optimization_target": optimization_target,
        "report_interval_hours": None,
        "live_dashboard_port": 0,
        "metric_pulling_prometheus_extra_query_params": None,
    }
    for key in _PLANNER_PASSTHROUGH:
        if key in candidate_config:
            payload[key] = candidate_config[key]
    if max_num_gpus is not None:
        payload["max_gpu_budget"] = max_num_gpus
    elif sample.get("gpu_budget") is not None:
        payload["max_gpu_budget"] = _int_value(sample, "gpu_budget")
    if sample.get("min_gpu_budget") is not None:
        payload["min_gpu_budget"] = _int_value(sample, "min_gpu_budget")
    if min_endpoint is not None:
        payload["min_endpoint"] = min_endpoint
    if mode in ("disagg", "prefill") and prefill_min_endpoint is not None:
        payload["prefill_min_endpoint"] = prefill_min_endpoint
    if mode in ("disagg", "decode") and decode_min_endpoint is not None:
        payload["decode_min_endpoint"] = decode_min_endpoint

    if mode == "disagg":
        payload["prefill_engine_num_gpu"] = _int_value(
            sample, "prefill_tp"
        ) * _int_value(sample, "prefill_attention_dp")
        payload["decode_engine_num_gpu"] = _int_value(sample, "decode_tp") * _int_value(
            sample, "decode_attention_dp"
        )
    else:
        payload["decode_engine_num_gpu"] = _int_value(sample, "tp") * _int_value(
            sample, "attention_dp"
        )

    if optimization_target == "sla" and sla is not None:
        if sla.get("ttft_ms") is not None:
            payload["ttft_ms"] = _float_value(sla, "ttft_ms")
        if sla.get("itl_ms") is not None:
            payload["itl_ms"] = _float_value(sla, "itl_ms")
    if optimization_target == "load":
        if mode in ("disagg", "prefill"):
            payload["prefill_scale_up_queue_tokens"] = 1
            payload["prefill_scale_down_queue_tokens"] = 0
        if mode in ("agg", "disagg", "decode"):
            payload["decode_scale_up_kv_rate"] = 90.0
            payload["decode_scale_down_kv_rate"] = 70.0
    return payload


def create_provider() -> DynamoPlannerSweepConfigProvider:
    """Create the entry-point registered Planner sweep configuration provider."""

    return DynamoPlannerSweepConfigProvider()
