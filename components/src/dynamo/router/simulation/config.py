# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prediction and recommendation config owned by the Dynamo Router adapter."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Annotated, Any, Literal, TypeAlias

from aisimulate.config.common import Choices, NumericRange
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dynamo.aisimulate.domains import (
    CHOICE_FORM,
    CHOICES,
    KNOB_FORM,
    RANGE,
    SCALAR,
    stepped_numeric_values,
)

MODES = frozenset({"kv_router", "round_robin"})
OVERLAP_SCORE_CREDIT_DEFAULTS = (0.0, 0.5, 1.0)
PREFILL_LOAD_SCALE_DEFAULTS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
TEMPERATURE_DEFAULTS = (0.0, 0.2, 0.5, 1.0)
OVERLAP_SCORE_CREDITS = frozenset(OVERLAP_SCORE_CREDIT_DEFAULTS)
PREFILL_LOAD_SCALES = frozenset(PREFILL_LOAD_SCALE_DEFAULTS)
TEMPERATURES = frozenset(TEMPERATURE_DEFAULTS)
LOAD_MODELS = frozenset({"none", "ais"})
StrictFiniteFloat = Annotated[float, Field(strict=True, ge=0.0, allow_inf_nan=False)]
Policy = Literal["round_robin", "kv_router"]
LoadModelType = Literal["none", "ais"]

RouterKnobDomain: TypeAlias = Annotated[
    Annotated[StrictFiniteFloat, SCALAR]
    | Annotated[Choices[StrictFiniteFloat], CHOICES]
    | Annotated[NumericRange, RANGE],
    KNOB_FORM,
]
PolicyDomain: TypeAlias = Annotated[
    Annotated[Policy, SCALAR] | Annotated[Choices[Policy], CHOICES], CHOICE_FORM
]
LoadModelTypeDomain: TypeAlias = Annotated[
    Annotated[LoadModelType, SCALAR] | Annotated[Choices[LoadModelType], CHOICES],
    CHOICE_FORM,
]
# Public numeric knob -> AISimulate default domain, in plan dimension order.
ROUTER_KNOB_DEFAULTS: dict[str, tuple[float, ...]] = {
    "overlap_score_credit": OVERLAP_SCORE_CREDIT_DEFAULTS,
    "prefill_load_scale": PREFILL_LOAD_SCALE_DEFAULTS,
    "temperature": TEMPERATURE_DEFAULTS,
}
# Public numeric knob -> KV router ``PolicyParameters`` name (the replay hook's
# ``router_config`` key and the live PATCH ``/router_policy`` key).
ROUTER_PUBLIC_TO_LIVE: dict[str, str] = {
    "overlap_score_credit": "overlap_score_credit",
    "prefill_load_scale": "prefill_load_scale",
    "temperature": "router_temperature",
}
# ``PolicyParameters`` without an AISimulate public name. Retune accepts them
# under their live names; recommend and predict do not search them.
ROUTER_RETUNE_EXTENSIONS: tuple[str, ...] = (
    "overlap_score_credit_decay",
    "decode_active_request_weight",
)
_STARTUP_OVERLAP_ACCOUNTING = (
    "the startup KvRouterConfig also feeds it to the router's overlap "
    "accounting (effective cached tokens), which a runtime update does not "
    "change, so a candidate replay would score a router no update can produce"
)
# ``PolicyParameters`` retune rejects, mapped to the reason its error gives.
ROUTER_NOT_RETUNABLE: dict[str, str] = {
    "host_cache_hit_weight": _STARTUP_OVERLAP_ACCOUNTING,
    "disk_cache_hit_weight": _STARTUP_OVERLAP_ACCOUNTING,
    "shared_cache_multiplier": (
        "replay reports no shared cache hits, so every value would score the same"
    ),
}


class PrefillLoadModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: LoadModelType = "none"


class PrefillLoadRecommendationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: LoadModelTypeDomain = Field(
        default_factory=lambda: Choices[LoadModelType](choices=["none", "ais"])
    )


class RouterPredictionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy: Policy = "round_robin"
    prefill_load_model: PrefillLoadModel = Field(default_factory=PrefillLoadModel)
    overlap_score_credit: StrictFiniteFloat | None = None
    prefill_load_scale: StrictFiniteFloat | None = None
    temperature: StrictFiniteFloat | None = None

    @field_validator(
        "overlap_score_credit",
        "prefill_load_scale",
        "temperature",
        mode="before",
    )
    @classmethod
    def _reject_explicit_null(cls, value: Any) -> Any:
        if value is None:
            raise ValueError("Router numeric knobs do not accept null")
        return value

    @model_validator(mode="after")
    def _validate_policy(self) -> RouterPredictionConfig:
        if self.policy == "round_robin":
            conflicts = {
                "overlap_score_credit",
                "prefill_load_scale",
                "temperature",
            }.intersection(self.model_fields_set)
            if self.prefill_load_model.type != "none" or conflicts:
                raise ValueError(
                    "round_robin requires prefill_load_model.type='none' and "
                    f"rejects KV-router knobs {sorted(conflicts)}"
                )
        else:
            if self.overlap_score_credit is None:
                self.overlap_score_credit = 1.0
            if self.prefill_load_scale is None:
                self.prefill_load_scale = 1.0
            if self.temperature is None:
                self.temperature = 0.0
        return self


class RouterRecommendationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy: PolicyDomain = Field(
        default_factory=lambda: Choices[Policy](choices=["round_robin", "kv_router"])
    )
    prefill_load_model: PrefillLoadRecommendationModel = Field(
        default_factory=PrefillLoadRecommendationModel
    )
    overlap_score_credit: RouterKnobDomain | None = None
    prefill_load_scale: RouterKnobDomain | None = None
    temperature: RouterKnobDomain | None = None

    @field_validator(
        "overlap_score_credit",
        "prefill_load_scale",
        "temperature",
        mode="before",
    )
    @classmethod
    def _reject_explicit_null(cls, value: Any) -> Any:
        if value is None:
            raise ValueError("Router numeric knobs do not accept null")
        return value

    @model_validator(mode="before")
    @classmethod
    def _apply_conditional_defaults(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        raw_policy = value.get("policy")
        round_robin_only = raw_policy == "round_robin" or (
            isinstance(raw_policy, Mapping)
            and set(raw_policy) == {"choices"}
            and raw_policy["choices"] == ["round_robin"]
        )
        if round_robin_only:
            resolved = dict(value)
            resolved.setdefault("prefill_load_model", {"type": "none"})
            return resolved
        return value

    @model_validator(mode="after")
    def _validate_policy(self) -> RouterRecommendationConfig:
        policies = (
            set(self.policy.choices)
            if isinstance(self.policy, Choices)
            else {self.policy}
        )
        if policies == {"round_robin"}:
            conflicts = {
                "overlap_score_credit",
                "prefill_load_scale",
                "temperature",
            }.intersection(self.model_fields_set)
            load_types = (
                set(self.prefill_load_model.type.choices)
                if isinstance(self.prefill_load_model.type, Choices)
                else {self.prefill_load_model.type}
            )
            if load_types != {"none"}:
                raise ValueError(
                    "router.policy=round_robin rejects prefill_load_model.type "
                    f"{sorted(load_types)}; only 'none' is allowed"
                )
            if conflicts:
                raise ValueError(
                    "router.policy=round_robin rejects KV-router fields "
                    f"{sorted(conflicts)}"
                )
        elif "round_robin" in policies:
            load_types = (
                set(self.prefill_load_model.type.choices)
                if isinstance(self.prefill_load_model.type, Choices)
                else {self.prefill_load_model.type}
            )
            if "none" not in load_types:
                raise ValueError(
                    "router policy domain containing round_robin requires "
                    "prefill_load_model.type to include none"
                )
        for name in (
            "overlap_score_credit",
            "prefill_load_scale",
            "temperature",
        ):
            domain = getattr(self, name)
            if isinstance(domain, NumericRange) and domain.range.min < 0:
                raise ValueError(f"router.{name} range requires min >= 0")
        return self


# Live names a retune domain might be written with -> the public name to use.
_ROUTER_LIVE_NAME_HINTS = {
    **{
        live: public for public, live in ROUTER_PUBLIC_TO_LIVE.items() if live != public
    },
    "router_mode": "policy",
    "router_prefill_load_model": "prefill_load_model.type",
}


class RouterRetuneConfig(RouterRecommendationConfig):
    """``retune.router``: the recommendation schema plus retune extension knobs.

    Syntax, validation and default domains are the recommendation ones. Two
    differences follow from retuning a running KV router:

    - ``policy`` is fixed at startup and retune requires ``kv_router``, so it
      defaults to ``kv_router`` and rejects any other domain.
      ``prefill_load_model.type`` is fixed at startup too; the caller checks
      an explicit value against the startup one.
    - The ``ROUTER_RETUNE_EXTENSIONS`` knobs have no AISimulate default
      domain: omitted, they keep their current value.
    """

    policy: PolicyDomain = "kv_router"
    overlap_score_credit_decay: RouterKnobDomain | None = None
    decode_active_request_weight: RouterKnobDomain | None = None

    @model_validator(mode="before")
    @classmethod
    def _reject_live_names_and_other_policies(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        for live, public in _ROUTER_LIVE_NAME_HINTS.items():
            if live in value:
                raise ValueError(
                    f"retune.router.{live} is not a public router name; "
                    f"write {public} instead"
                )
        for name, reason in ROUTER_NOT_RETUNABLE.items():
            if name in value:
                raise ValueError(f"retune.router.{name} is not retunable: {reason}")
        policy = value.get("policy", "kv_router")
        if policy != "kv_router" and policy != {"choices": ["kv_router"]}:
            raise ValueError(
                "retune.router.policy is fixed at startup and retune requires "
                f"kv_router; got {policy!r}"
            )
        return value

    @field_validator(*ROUTER_RETUNE_EXTENSIONS, mode="before")
    @classmethod
    def _reject_explicit_null_extension(cls, value: Any) -> Any:
        if value is None:
            raise ValueError("Router numeric knobs do not accept null")
        return value

    @model_validator(mode="after")
    def _validate_extension_ranges(self) -> RouterRetuneConfig:
        for name in ROUTER_RETUNE_EXTENSIONS:
            domain = getattr(self, name)
            if isinstance(domain, NumericRange) and domain.range.min < 0:
                raise ValueError(f"router.{name} range requires min >= 0")
        return self


class RouterSearchSpace(BaseModel):
    """Normalized legacy Sweeper provider input."""

    model_config = ConfigDict(extra="forbid")

    mode: list[str] = Field(default_factory=lambda: ["kv_router", "round_robin"])
    overlap_score_credit: list[float] = Field(
        default_factory=lambda: list(OVERLAP_SCORE_CREDIT_DEFAULTS)
    )
    prefill_load_scale: list[float] = Field(
        default_factory=lambda: list(PREFILL_LOAD_SCALE_DEFAULTS)
    )
    temperature: list[float] = Field(default_factory=lambda: list(TEMPERATURE_DEFAULTS))

    prefill_load_model_type: list[str] = Field(default_factory=lambda: ["none"])
    active_decode_blocks_threshold: int | None = None
    active_prefill_tokens_threshold: int | None = None
    active_prefill_tokens_threshold_frac: float | None = None
    no_admission_control: bool = False
    public_schema: bool = False

    @model_validator(mode="before")
    @classmethod
    def _normalize_public_schema(cls, value: Any) -> Any:
        if not isinstance(value, Mapping) or "policy" not in value:
            return value
        public = dict(value)
        normalized = {
            "public_schema": True,
            "mode": domain_choices(public.pop("policy"), ["kv_router", "round_robin"]),
            "overlap_score_credit": domain_choices(
                public.pop("overlap_score_credit", None),
                list(OVERLAP_SCORE_CREDIT_DEFAULTS),
            ),
            "prefill_load_scale": domain_choices(
                public.pop("prefill_load_scale", None),
                list(PREFILL_LOAD_SCALE_DEFAULTS),
            ),
            "temperature": domain_choices(
                public.pop("temperature", None), list(TEMPERATURE_DEFAULTS)
            ),
        }
        load_model = public.pop("prefill_load_model", None)
        raw_type = load_model.get("type") if isinstance(load_model, Mapping) else None
        normalized["prefill_load_model_type"] = domain_choices(
            raw_type, ["none", "ais"]
        )
        normalized.update(public)
        return normalized

    @model_validator(mode="after")
    def _validate_choices(self) -> RouterSearchSpace:
        string_domains: tuple[
            tuple[str, list[str], frozenset[str]],
            ...,
        ] = (
            ("mode", self.mode, MODES),
            ("prefill_load_model_type", self.prefill_load_model_type, LOAD_MODELS),
        )
        for name, string_values, allowed in string_domains:
            if not string_values:
                raise ValueError(f"{name} must list at least one choice")
            string_invalid = [value for value in string_values if value not in allowed]
            if string_invalid:
                raise ValueError(
                    f"{name} has invalid choices {string_invalid}; "
                    f"allowed: {sorted(allowed)}"
                )
        numeric_domains: tuple[
            tuple[str, list[float], frozenset[float]],
            ...,
        ] = (
            ("overlap_score_credit", self.overlap_score_credit, OVERLAP_SCORE_CREDITS),
            ("prefill_load_scale", self.prefill_load_scale, PREFILL_LOAD_SCALES),
            ("temperature", self.temperature, TEMPERATURES),
        )
        for name, numeric_values, legacy_allowed in numeric_domains:
            if not numeric_values:
                raise ValueError(f"{name} must list at least one choice")
            numeric_invalid = (
                [
                    value
                    for value in numeric_values
                    if not math.isfinite(value) or value < 0.0
                ]
                if self.public_schema
                else [value for value in numeric_values if value not in legacy_allowed]
            )
            if numeric_invalid:
                raise ValueError(f"{name} has invalid choices {numeric_invalid}")
        if "kv_router" not in self.mode:
            return self
        admission_pins = {
            "active_decode_blocks_threshold": self.active_decode_blocks_threshold,
            "active_prefill_tokens_threshold": self.active_prefill_tokens_threshold,
            "active_prefill_tokens_threshold_frac": self.active_prefill_tokens_threshold_frac,
        }
        enabled = [name for name, value in admission_pins.items() if value is not None]
        if self.no_admission_control:
            enabled.append("no_admission_control")
        if enabled:
            raise ValueError(
                "Router admission-control knobs are not supported by the Dynamo "
                "replay API; remove " + ", ".join(enabled)
            )
        return self


def domain_choices(value: Any, default: list[Any]) -> list[Any]:
    if value is None:
        return list(default)
    if isinstance(value, Mapping) and set(value) == {"choices"}:
        choices = value["choices"]
        if not isinstance(choices, list) or not choices:
            raise ValueError("Router choices must be a nonempty list")
        if len({repr(choice) for choice in choices}) != len(choices):
            raise ValueError("Router choices must contain unique values")
        return list(choices)
    if isinstance(value, Mapping) and set(value) == {"range"}:
        raw = value["range"]
        if not isinstance(raw, Mapping):
            raise ValueError("Router range must be a mapping")
        unknown = set(raw) - {"min", "max", "step", "scale"}
        if unknown:
            raise ValueError(f"Router range has unknown fields {sorted(unknown)}")
        if "min" not in raw or "max" not in raw:
            raise ValueError("Router range requires min and max")
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
            raise ValueError("Router range requires finite numeric min <= max")
        step = raw.get("step")
        if (
            step is None
            or isinstance(step, bool)
            or not isinstance(step, (int, float))
            or step <= 0
            or raw.get("scale", "linear") != "linear"
        ):
            raise ValueError("Router numeric ranges require a stepped linear range")
        return stepped_numeric_values(minimum, maximum, step)
    if isinstance(value, Mapping):
        raise ValueError("Router domains must contain exactly choices or range")
    return [value]
