# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo Router implementation of the Sweeper sweep-configuration-provider ABI."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

from aisimulate.config.common import Choices, NumericRange
from aisimulate.config_adapter import (
    PredictionAdapterContext,
    RecommendationAdapterContext,
)
from aisimulate.sweeper.provider import (
    AdapterReplaySpec,
    AdapterSearchPlan,
    CandidateContext,
    ConditionalSearchSpace,
    JSONValue,
    RuntimeHookSpec,
    SearchSpaceFragment,
    SweepContext,
)

from .config import (
    OVERLAP_SCORE_CREDIT_DEFAULTS,
    PREFILL_LOAD_SCALE_DEFAULTS,
    TEMPERATURE_DEFAULTS,
    RouterPredictionConfig,
    RouterRecommendationConfig,
    RouterSearchSpace,
    _stepped_numeric_values,
)

_PROVIDER_API_VERSION = 1
_ROUTER_HOOK_API_VERSION = 1
_HOOK = RuntimeHookSpec(
    provider="dynamo.router",
    kind="placement_policy",
    api_version=_ROUTER_HOOK_API_VERSION,
)


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


def _aic_payload(
    *,
    backend: str,
    system: str,
    model: str,
    backend_version: str | None,
    tp: int,
    attention_dp: int,
    moe_tp: int,
    moe_ep: int,
) -> dict[str, JSONValue]:
    return {
        "aic_backend": backend,
        "aic_system": system,
        "aic_model_path": model,
        "aic_backend_version": backend_version,
        "aic_tp_size": tp,
        "aic_attention_dp_size": attention_dp,
        "aic_moe_tp_size": moe_tp if moe_tp * moe_ep > 1 else None,
        "aic_moe_ep_size": moe_ep if moe_tp * moe_ep > 1 else None,
    }


def _aic_perf_config_from_candidate(
    context: CandidateContext, *, enabled: bool
) -> dict[str, JSONValue] | None:
    if not enabled:
        return None
    sample = context.sample
    mode = str(sample["deployment_mode"])
    prefix = "prefill_" if mode == "disagg" else ""
    return _aic_payload(
        backend=str(sample["backend"]),
        system=str(sample["hardware_sku"]),
        model=str(sample["model_name"]),
        backend_version=str(sample.get("backend_version") or "") or None,
        tp=int(sample[f"{prefix}tp"]),
        attention_dp=int(sample[f"{prefix}attention_dp"]),
        moe_tp=int(sample.get(f"{prefix}moe_tp", 1)),
        moe_ep=int(sample.get(f"{prefix}moe_ep", 1)),
    )


def _aic_perf_config_from_prediction(
    context: PredictionAdapterContext, *, enabled: bool
) -> dict[str, JSONValue] | None:
    if not enabled:
        return None
    engine = context.engine
    mode = str(engine.get("mode", "aggregated"))
    role = "prefill" if mode == "disaggregated" else "aggregated"
    workers = engine.get("workers")
    if not isinstance(workers, Mapping) or not isinstance(workers.get(role), Mapping):
        raise TypeError(f"Router AIC load model requires engine.workers.{role}")
    worker = workers[role]
    parallel = worker.get("parallelism")
    if not isinstance(parallel, Mapping):
        raise TypeError(f"Router AIC load model requires {role} parallelism")
    return _aic_payload(
        backend=str(engine["backend"]),
        system=str(engine["hardware"]),
        model=str(engine["model"]),
        backend_version=str(engine.get("backend_version") or "") or None,
        tp=int(parallel.get("tensor", 1)),
        attention_dp=int(parallel.get("attention_data", 1)),
        moe_tp=int(parallel.get("moe_tensor", 1)),
        moe_ep=int(parallel.get("moe_expert", 1)),
    )


class DynamoRouterSweepConfigProvider:
    """Router search-space preparation and replay-spec materialization."""

    name = "dynamo.router"
    section = "router"
    config_adapter_api_version = 3
    # Keep the implemented provider ABI independent from the installed consumer.
    api_version = _PROVIDER_API_VERSION

    def compile_prediction(
        self,
        config: Mapping[str, JSONValue],
        context: PredictionAdapterContext,
    ) -> AdapterReplaySpec:
        public = RouterPredictionConfig.model_validate(config)
        concrete = public.model_dump(mode="json", exclude_none=True)
        if public.policy == "round_robin":
            return AdapterReplaySpec(config=concrete)
        router_config: dict[str, JSONValue] = {
            "overlap_score_credit": public.overlap_score_credit
            if public.overlap_score_credit is not None
            else 1.0,
            "prefill_load_scale": public.prefill_load_scale
            if public.prefill_load_scale is not None
            else 1.0,
            "router_temperature": public.temperature
            if public.temperature is not None
            else 0.0,
            "router_prefill_load_model": public.prefill_load_model.type,
        }
        return AdapterReplaySpec(
            config=concrete,
            runtime_hooks=(
                RuntimeHookSpec(
                    provider=_HOOK.provider,
                    kind=_HOOK.kind,
                    api_version=_HOOK.api_version,
                    config={
                        "router_mode": public.policy,
                        "router_config": router_config,
                        "aic_perf_config": _aic_perf_config_from_prediction(
                            context,
                            enabled=public.prefill_load_model.type == "aic",
                        ),
                    },
                ),
            ),
        )

    def compile_recommendation(
        self,
        config: Mapping[str, JSONValue],
        context: RecommendationAdapterContext,
    ) -> AdapterSearchPlan:
        public = RouterRecommendationConfig.model_validate(config)
        return self._compile_public_search(public, context.sweep)

    def _compile_public_search(
        self,
        public: RouterRecommendationConfig,
        context: SweepContext,
    ) -> AdapterSearchPlan:
        policies = (
            list(public.policy.choices)
            if isinstance(public.policy, Choices)
            else [public.policy]
        )
        load_model = public.prefill_load_model.type
        load_models = (
            list(load_model.choices)
            if isinstance(load_model, Choices)
            else [load_model]
        )
        numeric_specs = {
            "overlap_score_credit": (
                public.overlap_score_credit,
                list(OVERLAP_SCORE_CREDIT_DEFAULTS),
            ),
            "prefill_load_scale": (
                public.prefill_load_scale,
                list(PREFILL_LOAD_SCALE_DEFAULTS),
            ),
            "temperature": (public.temperature, list(TEMPERATURE_DEFAULTS)),
        }
        modes = _deployment_modes(context.core_search_space)
        if policies == ["round_robin"]:
            fragment = SearchSpaceFragment(
                choices_by_branch={mode: {"mode": ["round_robin"]} for mode in modes}
            )
            return AdapterSearchPlan(fragment=fragment, state={"public_schema": True})

        if set(policies) == {"round_robin", "kv_router"}:
            expanded: dict[str, list[float]] = {}
            ranges: dict[str, tuple[float, float]] = {}
            log_ranges: list[str] = []
            for name, (value, defaults) in numeric_specs.items():
                if isinstance(value, NumericRange):
                    raw = value.range
                    if raw.min == raw.max:
                        expanded[name] = [raw.min]
                        continue
                    if raw.step is None:
                        ranges[name] = (raw.min, raw.max)
                        if raw.scale == "log":
                            log_ranges.append(name)
                    else:
                        expanded[name] = _stepped_numeric_values(
                            raw.min, raw.max, raw.step
                        )
                elif isinstance(value, Choices):
                    expanded[name] = list(value.choices)
                elif value is None:
                    expanded[name] = list(defaults)
                else:
                    expanded[name] = [float(value)]
            conditional = ConditionalSearchSpace(
                selector="mode",
                values=["kv_router"],
                choices={
                    "prefill_load_model_type": list(load_models),
                    **{name: list(values) for name, values in expanded.items()},
                },
                float_ranges=deepcopy(ranges),
                log_float_ranges=list(log_ranges),
            )
            fragment = SearchSpaceFragment(
                choices_by_branch={mode: {"mode": list(policies)} for mode in modes},
                conditional_by_branch={mode: [deepcopy(conditional)] for mode in modes},
            )
            return AdapterSearchPlan(
                fragment=fragment,
                state={"public_schema": True, "conditional_public": True},
                potential_runtime_hooks=(_HOOK,),
            )

        choices: dict[str, list[JSONValue]] = {
            "mode": ["kv_router"],
            "prefill_load_model_type": list(load_models),
        }
        ranges = {}
        log_ranges = []
        for name, (value, defaults) in numeric_specs.items():
            if isinstance(value, NumericRange):
                raw = value.range
                if raw.min == raw.max:
                    choices[name] = [raw.min]
                elif raw.step is not None:
                    choices[name] = _stepped_numeric_values(raw.min, raw.max, raw.step)
                else:
                    ranges[name] = (raw.min, raw.max)
                    if raw.scale == "log":
                        log_ranges.append(name)
            elif isinstance(value, Choices):
                choices[name] = list(value.choices)
            elif value is None:
                choices[name] = list(defaults)
            else:
                choices[name] = [value]
        return AdapterSearchPlan(
            fragment=SearchSpaceFragment(
                choices_by_branch={mode: deepcopy(choices) for mode in modes},
                float_ranges_by_branch={mode: deepcopy(ranges) for mode in modes},
                log_float_ranges_by_branch={mode: list(log_ranges) for mode in modes},
            ),
            state={"public_schema": True},
            potential_runtime_hooks=(_HOOK,),
        )

    def materialize_candidate(
        self,
        plan: AdapterSearchPlan,
        selection: Mapping[str, JSONValue],
        context: CandidateContext,
    ) -> AdapterReplaySpec:
        return self.materialize_replay(plan, selection, context)

    def generate_search_space(
        self,
        search_spec: Mapping[str, JSONValue],
        context: SweepContext,
    ) -> AdapterSearchPlan:
        public_schema = "policy" in search_spec or "prefill_load_model" in search_spec
        space = RouterSearchSpace.model_validate(search_spec)
        choices: dict[str, list[JSONValue]] = {"mode": list(space.mode)}
        kv_router_possible = "kv_router" in space.mode
        if kv_router_possible:
            choices.update(
                overlap_score_credit=list(space.overlap_score_credit),
                prefill_load_scale=list(space.prefill_load_scale),
                temperature=list(space.temperature),
                prefill_load_model_type=list(space.prefill_load_model_type),
            )
        fragment = SearchSpaceFragment(
            choices_by_branch={
                mode: deepcopy(choices)
                for mode in _deployment_modes(context.core_search_space)
            }
        )
        return AdapterSearchPlan(
            fragment=fragment,
            state={
                "search_space": space.model_dump(mode="json"),
                "public_schema": public_schema,
            },
            potential_runtime_hooks=(_HOOK,) if kv_router_possible else (),
        )

    def materialize_replay(
        self,
        plan: AdapterSearchPlan,
        selection: Mapping[str, JSONValue],
        context: CandidateContext,
    ) -> AdapterReplaySpec:
        if not isinstance(plan.state, dict):
            raise TypeError("Router adapter search plan state must be a mapping")
        raw_space = plan.state.get("search_space")
        if raw_space is not None:
            if not isinstance(raw_space, dict):
                raise TypeError("Router adapter search-space state must be a mapping")
            RouterSearchSpace.model_validate(raw_space)
        public_schema = plan.state.get("public_schema") is True

        configuration = selection.get("configuration")
        if isinstance(configuration, Mapping):
            selection = {**selection, **configuration}

        mode = str(selection["mode"])
        if mode == "round_robin":
            load_model = str(selection.get("prefill_load_model_type", "none"))
            if load_model != "none":
                raise ValueError("round_robin requires prefill_load_model.type='none'")
            return AdapterReplaySpec(
                config=(
                    {
                        "policy": mode,
                        "prefill_load_model": {"type": "none"},
                    }
                    if public_schema
                    else {"mode": mode}
                )
            )
        if mode != "kv_router":
            raise ValueError(f"unsupported Router mode {mode!r}")

        router_config: dict[str, JSONValue] = {
            "overlap_score_credit": _float_selection(selection, "overlap_score_credit"),
            "prefill_load_scale": _float_selection(selection, "prefill_load_scale"),
            "router_temperature": _float_selection(selection, "temperature"),
        }
        load_model = str(selection.get("prefill_load_model_type", "none"))
        if public_schema:
            router_config["router_prefill_load_model"] = load_model
        concrete_config: dict[str, JSONValue]
        if public_schema:
            concrete_config = {
                "policy": mode,
                "prefill_load_model": {"type": load_model},
                "overlap_score_credit": router_config["overlap_score_credit"],
                "prefill_load_scale": router_config["prefill_load_scale"],
                "temperature": router_config["router_temperature"],
            }
        else:
            concrete_config = {"mode": mode, **router_config}
        aic_perf_config = _aic_perf_config_from_candidate(
            context,
            enabled=public_schema and load_model == "aic",
        )
        hook_config: dict[str, JSONValue] = {
            "router_mode": mode,
            "router_config": router_config,
        }
        if public_schema:
            hook_config["aic_perf_config"] = aic_perf_config
        hook = RuntimeHookSpec(
            provider=_HOOK.provider,
            kind=_HOOK.kind,
            api_version=_HOOK.api_version,
            config=hook_config,
        )
        return AdapterReplaySpec(
            config=concrete_config,
            runtime_hooks=(hook,),
        )


def create_provider() -> DynamoRouterSweepConfigProvider:
    """Create the entry-point registered Router sweep configuration provider."""

    return DynamoRouterSweepConfigProvider()
