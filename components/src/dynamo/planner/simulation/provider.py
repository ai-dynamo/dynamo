# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo Planner implementation of the Sweeper sweep-configuration-provider ABI."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from copy import deepcopy
from enum import Enum
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from tqdm import tqdm  # type: ignore[import-untyped]

from aisimulate.sweeper.provider import (
    AdapterReplaySpec,
    AdapterSearchPlan,
    CandidateContext,
    JSONValue,
    RuntimeHookSpec,
    SearchSpaceFragment,
    SweepContext,
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

    @model_validator(mode="before")
    @classmethod
    def _upgrade_legacy_flat_presets(cls, data: Any) -> Any:
        # Backward compatibility: remove this conversion after the 1.5 release.
        if not isinstance(data, Mapping):
            return data

        upgraded = dict(data)
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
                stacklevel=2,
            )
        return upgraded


def _plain(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


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
        uses_scaling = uses_throughput or bool(fields["enable_load_scaling"])
        allowed = not uses_throughput or optimization_target == "sla"
        if (
            allowed
            and optimization_target == "sla"
            and sla is not None
            and (sla.get("ttft_ms") is None or sla.get("itl_ms") is None)
        ):
            allowed = not uses_scaling
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
    # Provider-owned constant: importing the consumer's current API_VERSION would
    # let an older wheel accidentally self-certify against a newer core package.
    api_version = _PROVIDER_API_VERSION

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
                "every Planner scaling_policy enables scaling, but an e2e-only "
                "SLA cannot seed the Planner's TTFT/ITL scaling target"
            )

        trace_path = context.workload.get("trace_path")
        predictor_result = sweep_load_predictor(
            policies=kept,
            candidates=space.load_predictor.preset,
            trace_path=str(trace_path) if trace_path is not None else None,
            show_progress=context.show_progress,
        )
        planner_enabled = any(
            fields["enable_throughput_scaling"] or fields["enable_load_scaling"]
            for fields in (scaling_fields(policy) for policy in kept)
        )
        local_choices: dict[str, list[JSONValue]] = {
            "scaling_policy": _json_choices(kept)
        }
        if planner_enabled:
            local_choices["fpm_sampling"] = _json_choices(space.fpm_sampling.preset)
            local_choices["load_sensitivity"] = _json_choices(
                space.load_sensitivity.preset
            )
        fragment = SearchSpaceFragment(
            choices_by_branch={
                mode: deepcopy(local_choices)
                for mode in _deployment_modes(context.core_search_space)
            }
        )
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
            potential_runtime_hooks=(_HOOK,) if planner_enabled else (),
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

        scaling_entry = selection["scaling_policy"]
        if not isinstance(scaling_entry, (str, dict)):
            raise TypeError(
                "Planner scaling_policy selection must be a string or mapping"
            )
        scaling = scaling_fields(scaling_entry)
        enabled = bool(
            scaling["enable_throughput_scaling"] or scaling["enable_load_scaling"]
        )
        candidate_config: dict[str, JSONValue] = {
            "scaling_policy": deepcopy(scaling_entry),
            **scaling,
        }
        if not enabled:
            return AdapterReplaySpec(config=candidate_config)

        fpm_entry = selection["fpm_sampling"]
        load_entry = selection["load_sensitivity"]
        if not isinstance(fpm_entry, (str, dict)) or not isinstance(
            load_entry, (str, dict)
        ):
            raise TypeError("Planner composite selections must be strings or mappings")
        candidate_config.update(fpm_fields(fpm_entry))
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

        planner_config = _planner_config_payload(
            candidate_config,
            sample=context.sample,
            optimization_target=str(state["optimization_target"]),
            sla=state["sla"] if isinstance(state["sla"], dict) else None,
            min_endpoint=space.min_endpoint,
            prefill_min_endpoint=space.prefill_min_endpoint,
            decode_min_endpoint=space.decode_min_endpoint,
        )
        hook = RuntimeHookSpec(
            provider=_HOOK.provider,
            kind=_HOOK.kind,
            api_version=_HOOK.api_version,
            config={"planner_config": planner_config},
        )
        reported_config: dict[str, JSONValue] = {
            "scaling_policy": deepcopy(scaling_entry),
            **planner_config,
        }
        return AdapterReplaySpec(config=reported_config, runtime_hooks=(hook,))


def _planner_config_payload(
    candidate_config: Mapping[str, JSONValue],
    *,
    sample: Mapping[str, JSONValue],
    optimization_target: str,
    sla: Mapping[str, JSONValue] | None,
    min_endpoint: int | None,
    prefill_min_endpoint: int | None,
    decode_min_endpoint: int | None,
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
    if sample.get("gpu_budget") is not None:
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
    return payload


def create_provider() -> DynamoPlannerSweepConfigProvider:
    """Create the entry-point registered Planner sweep configuration provider."""

    return DynamoPlannerSweepConfigProvider()
