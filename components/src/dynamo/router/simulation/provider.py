# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo Router implementation of the Sweeper sweep-configuration-provider ABI."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aisimulate.sweeper.provider import (
    AdapterReplaySpec,
    AdapterSearchPlan,
    CandidateContext,
    JSONValue,
    RuntimeHookSpec,
    SearchSpaceFragment,
    SweepContext,
)

_PROVIDER_API_VERSION = 1
_ROUTER_HOOK_API_VERSION = 1
_MODES = frozenset({"kv_router", "round_robin"})
_OVERLAP_SCORE_CREDITS = frozenset({0.0, 0.5, 1.0})
_PREFILL_LOAD_SCALES = frozenset({0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0})
_TEMPERATURES = frozenset({0.0, 0.2, 0.5, 1.0})
_HOOK = RuntimeHookSpec(
    provider="dynamo.router",
    kind="placement_policy",
    api_version=_ROUTER_HOOK_API_VERSION,
)


class RouterSearchSpace(BaseModel):
    """Validated Router-owned search space."""

    model_config = ConfigDict(extra="forbid")

    mode: list[str] = Field(default_factory=lambda: ["kv_router", "round_robin"])
    overlap_score_credit: list[float] = Field(default_factory=lambda: [0.0, 0.5, 1.0])
    prefill_load_scale: list[float] = Field(
        default_factory=lambda: [
            0.0,
            0.25,
            0.5,
            1.0,
            2.0,
            4.0,
            8.0,
            16.0,
            32.0,
        ]
    )
    temperature: list[float] = Field(default_factory=lambda: [0.0, 0.2, 0.5, 1.0])
    active_decode_blocks_threshold: int | None = None
    active_prefill_tokens_threshold: int | None = None
    active_prefill_tokens_threshold_frac: float | None = None
    no_admission_control: bool = False

    @model_validator(mode="after")
    def _validate_choices(self) -> RouterSearchSpace:
        specifications = (
            ("mode", self.mode, _MODES),
            (
                "overlap_score_credit",
                self.overlap_score_credit,
                _OVERLAP_SCORE_CREDITS,
            ),
            ("prefill_load_scale", self.prefill_load_scale, _PREFILL_LOAD_SCALES),
            ("temperature", self.temperature, _TEMPERATURES),
        )
        for name, values, allowed in specifications:
            if not values:
                raise ValueError(f"{name} must list at least one choice")
            invalid = [value for value in values if value not in allowed]
            if invalid:
                raise ValueError(
                    f"{name} has invalid choices {invalid}; allowed: {sorted(allowed)}"
                )

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


def _deployment_modes(
    core_search_space: Mapping[str, JSONValue],
) -> list[str]:
    raw_modes = core_search_space.get("deployment_mode", ["agg", "disagg"])
    if not isinstance(raw_modes, list):
        raise TypeError("core deployment_mode must be a list")
    return [str(mode) for mode in raw_modes]


def _float_selection(selection: Mapping[str, JSONValue], name: str) -> float:
    value = selection[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Router {name} selection must be numeric")
    return float(value)


class DynamoRouterSweepConfigProvider:
    """Router search-space preparation and replay-spec materialization."""

    name = "dynamo.router"
    # Keep the implemented provider ABI independent from the installed consumer.
    api_version = _PROVIDER_API_VERSION

    def generate_search_space(
        self,
        search_spec: Mapping[str, JSONValue],
        context: SweepContext,
    ) -> AdapterSearchPlan:
        space = RouterSearchSpace.model_validate(search_spec)
        choices: dict[str, list[JSONValue]] = {"mode": list(space.mode)}
        kv_router_possible = "kv_router" in space.mode
        if kv_router_possible:
            choices.update(
                overlap_score_credit=list(space.overlap_score_credit),
                prefill_load_scale=list(space.prefill_load_scale),
                temperature=list(space.temperature),
            )
        fragment = SearchSpaceFragment(
            choices_by_branch={
                mode: deepcopy(choices)
                for mode in _deployment_modes(context.core_search_space)
            }
        )
        return AdapterSearchPlan(
            fragment=fragment,
            state={"search_space": space.model_dump(mode="json")},
            potential_runtime_hooks=(_HOOK,) if kv_router_possible else (),
        )

    def materialize_replay(
        self,
        plan: AdapterSearchPlan,
        selection: Mapping[str, JSONValue],
        context: CandidateContext,
    ) -> AdapterReplaySpec:
        del context
        if not isinstance(plan.state, dict):
            raise TypeError("Router adapter search plan state must be a mapping")
        raw_space = plan.state["search_space"]
        if not isinstance(raw_space, dict):
            raise TypeError("Router adapter search-space state must be a mapping")
        RouterSearchSpace.model_validate(raw_space)

        mode = str(selection["mode"])
        if mode == "round_robin":
            return AdapterReplaySpec(config={"mode": mode})
        if mode != "kv_router":
            raise ValueError(f"unsupported Router mode {mode!r}")

        router_config: dict[str, JSONValue] = {
            "overlap_score_credit": _float_selection(selection, "overlap_score_credit"),
            "prefill_load_scale": _float_selection(selection, "prefill_load_scale"),
            "router_temperature": _float_selection(selection, "temperature"),
        }
        concrete_config: dict[str, JSONValue] = {
            "mode": mode,
            **router_config,
        }
        hook = RuntimeHookSpec(
            provider=_HOOK.provider,
            kind=_HOOK.kind,
            api_version=_HOOK.api_version,
            config={"router_mode": mode, "router_config": router_config},
        )
        return AdapterReplaySpec(
            config=concrete_config,
            runtime_hooks=(hook,),
        )


def create_provider() -> DynamoRouterSweepConfigProvider:
    """Create the entry-point registered Router sweep configuration provider."""

    return DynamoRouterSweepConfigProvider()
