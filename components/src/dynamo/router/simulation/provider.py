# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo Router implementation of the Sweeper sweep-configuration-provider ABI."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

from aisimulate.config.common import Choices
from aisimulate.config_adapter import (
    PredictionAdapterContext,
    RecommendationAdapterContext,
)
from aisimulate.sweeper.provider import (
    AdapterReplaySpec,
    AdapterSearchPlan,
    CandidateContext,
    ConditionalSearchSpace,
    InfeasibleCandidate,
    JSONValue,
    RuntimeHookSpec,
    SearchSpaceFragment,
    SweepContext,
)

from dynamo.aisimulate.domains import ContinuousDomain, expand_domain
from dynamo.common.ais_perf_model import REMOVE_IN, router_perf_config
from dynamo.common.configuration.utils import Deprecated

from .config import (
    ROUTER_KNOB_DEFAULTS,
    ROUTER_PUBLIC_TO_LIVE,
    USE_ENGINE_PERF_MODEL,
    RouterPredictionConfig,
    RouterRecommendationConfig,
    RouterSearchSpace,
)

_PROVIDER_API_VERSION = 1
_ROUTER_HOOK_API_VERSION = 1
# KV-router knob values a prediction materializes when the config omits them;
# ``RouterPredictionConfig`` fills the same values for ``policy: kv_router``.
_PREDICTION_DEFAULTS: dict[str, float] = {
    "overlap_score_credit": 1.0,
    "prefill_load_scale": 1.0,
    "temperature": 0.0,
}
_HOOK = RuntimeHookSpec(
    provider="dynamo.router",
    kind="placement_policy",
    api_version=_ROUTER_HOOK_API_VERSION,
)
# The deprecated way to give the router an AIS perf model, and its replacement.
_OLD_ROUTER_SECTION = (
    "router.prefill_load_model.type: ais without router.ais_perf_model"
)
# The old form also works with an engine that has no AIS perf model, where
# {use: engine} is an error: the replacement names the form for that engine too.
_NEW_ROUTER_SECTION = (
    "router.ais_perf_model: {use: engine} (or a perf model config when the "
    "engine does not use an AIS perf model)"
)


def _deployment_modes(
    core_search_space: Mapping[str, JSONValue],
) -> list[str]:
    raw_modes = core_search_space.get("deployment_mode", ["agg", "disagg"])
    if not isinstance(raw_modes, list):
        raise TypeError("core deployment_mode must be a list")
    return [str(mode) for mode in raw_modes]


def _may_select_ais(public: RouterRecommendationConfig) -> bool:
    """Return whether a search can pick the KV router with the AIS load model."""
    policies = (
        public.policy.choices if isinstance(public.policy, Choices) else [public.policy]
    )
    load_model = public.prefill_load_model.type
    load_models = (
        load_model.choices if isinstance(load_model, Choices) else [load_model]
    )
    return "kv_router" in policies and "ais" in load_models


def _writes_load_model_type(public: RouterRecommendationConfig) -> bool:
    """Return whether the section writes ``prefill_load_model.type``.

    A section that does not write it searches the default load models.
    """
    return (
        "prefill_load_model" in public.model_fields_set
        and "type" in public.prefill_load_model.model_fields_set
    )


def _float_selection(selection: Mapping[str, JSONValue], name: str) -> float:
    value = selection[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Router {name} selection must be numeric")
    return float(value)


def _ais_perf_config_from_candidate(
    context: CandidateContext,
    *,
    enabled: bool,
    core_search_space: Mapping[str, JSONValue],
    engine_only: bool = False,
) -> dict[str, JSONValue] | None:
    """Return the router perf config of a candidate with the AIS load model.

    With ``engine_only`` the config must come from the candidate's engine:
    the result is ``None`` when that engine has no AIS perf model.
    """
    if not enabled:
        return None
    sample = context.sample
    role = "prefill" if str(sample["deployment_mode"]) == "disagg" else "agg"
    resolved = sample.get("forward_pass_estimators", {})
    config = (
        resolved.get(role, {}).get("config") if isinstance(resolved, Mapping) else None
    )
    if isinstance(config, Mapping):
        return deepcopy(dict(config))
    # Custom AIC timing is deliberately absent from the resolver's estimator map.
    timing = sample.get(f"{role}_timing_model")
    if isinstance(timing, Mapping) and timing.get("provider") in ("aic", "ais"):
        config = timing.get("config")
        if isinstance(config, Mapping):
            capacity_fields = {
                "gpu_memory_utilization",
                "mem_fraction_static",
                "free_gpu_memory_fraction",
                "cuda_graph_reserved_bytes",
            }
            return deepcopy(
                {
                    key: value
                    for key, value in config.items()
                    if key not in capacity_fields
                }
            )
    if engine_only:
        return None
    # Fixed/polynomial worker timing does not create a worker estimator. The
    # Router may still select an AIS load estimate: use the same Core-owned
    # role request builder, retaining policies, roots and estimator controls.
    from aisimulate.sweeper.config import SearchSpace
    from aisimulate.sweeper.forward_pass_estimator import ForwardPassEstimatorResolver

    request_space = dict(core_search_space)
    request_space.setdefault("model_name", sample["model_name"])
    request_space.setdefault("hardware_sku", sample["hardware_sku"])
    request_space.setdefault("backend", [sample["backend"]])
    request_space.setdefault("backend_version", sample.get("backend_version"))
    prefix = "prefill_" if role == "prefill" else ""
    request_sample = dict(sample)
    for field in ("pp", "moe_tp", "moe_ep"):
        request_sample.setdefault(prefix + field, 1)
    request_sample.setdefault(role + "_block_size", None)
    resolver = ForwardPassEstimatorResolver(SearchSpace.model_validate(request_space))
    return resolver._request(request_sample, role).to_dict()


def _ais_perf_config_from_prediction(
    context: PredictionAdapterContext, *, enabled: bool
) -> dict[str, JSONValue] | None:
    if not enabled:
        return None
    from dataclasses import fields

    from aisimulate_core.sdk import ForwardPassPerfModelConfig

    engine = context.engine
    role = (
        "prefill"
        if str(engine.get("mode", "aggregated")) == "disaggregated"
        else "aggregated"
    )
    workers = engine.get("workers")
    if not isinstance(workers, Mapping) or not isinstance(workers.get(role), Mapping):
        raise TypeError(f"AIS router load model requires engine.workers.{role}")
    worker = workers[role]
    parallel = worker.get("parallelism", {})
    timing = worker.get("timing", {})
    # Controls are selected using the upstream schema, including newly added
    # canonical fields; worker timing settings override engine-level settings.
    names = {field.name for field in fields(ForwardPassPerfModelConfig)}
    config = {
        key: deepcopy(value)
        for key, value in engine.items()
        if key in names and value is not None
    }
    config.update(
        {
            key: deepcopy(value)
            for key, value in timing.items()
            if key in names and value is not None
        }
    )
    fpm_parquet_path = timing.get("fpm_parquet_path")
    if fpm_parquet_path is not None:
        interpolation = config.setdefault("estimator_config", {}).setdefault(
            "fpm_interpolation", {}
        )
        if interpolation.get("fpm_parquet_path", fpm_parquet_path) != fpm_parquet_path:
            raise ValueError(
                "conflicting fpm_parquet_path and "
                "estimator_config.fpm_interpolation.fpm_parquet_path"
            )
        interpolation["fpm_parquet_path"] = fpm_parquet_path
    config.update(
        model=engine["model"],
        system=worker.get("hardware") or engine["hardware"],
        backend=engine["backend"],
        worker_type=role,
        tp=parallel.get("tensor", 1),
        pp=parallel.get("pipeline", 1),
        attention_dp=parallel.get("attention_data", 1),
    )
    moe_tp, moe_ep = parallel.get("moe_tensor", 1), parallel.get("moe_expert", 1)
    if moe_tp * moe_ep > 1:
        config.update(moe_tp_size=moe_tp, moe_ep_size=moe_ep)
    cache = worker.get("kv_cache", {})
    if cache.get("block_size") is not None:
        config["kv_block_size"] = cache["block_size"]
    speculation = engine.get("speculation")
    if isinstance(speculation, Mapping):
        config["speculation"] = {
            "kind": speculation["kind"],
            "params": {"num_speculative_tokens": speculation["num_speculative_tokens"]},
        }
    return ForwardPassPerfModelConfig(**config).to_dict()


class DynamoRouterSweepConfigProvider:
    """Router search-space preparation and replay-spec materialization."""

    name = "dynamo.router"
    section = "router"
    config_adapter_api_version = 3
    # Keep the implemented provider ABI independent from the installed consumer.
    api_version = _PROVIDER_API_VERSION
    # One object warns once about the deprecated router section: a
    # recommendation compiles a prediction for each candidate it selects.
    _warned_old_router_section = False

    def compile_prediction(
        self,
        config: Mapping[str, JSONValue],
        context: PredictionAdapterContext,
    ) -> AdapterReplaySpec:
        public = RouterPredictionConfig.model_validate(config)
        concrete = public.model_dump(mode="json", exclude_none=True)
        if public.policy == "round_robin":
            return AdapterReplaySpec(config=concrete)
        router_config: dict[str, JSONValue] = {}
        for name, default in _PREDICTION_DEFAULTS.items():
            value = getattr(public, name)
            router_config[ROUTER_PUBLIC_TO_LIVE[name]] = (
                value if value is not None else default
            )
        router_config["router_prefill_load_model"] = public.prefill_load_model.type
        written = public.ais_perf_model
        old_form = written is None and public.prefill_load_model.type == "ais"
        perf_config: dict[str, JSONValue]
        if written is None:
            perf_config = {
                "ais_perf_config": _ais_perf_config_from_prediction(
                    context, enabled=old_form
                )
            }
        elif written == USE_ENGINE_PERF_MODEL:
            # The replay runner fills the config from the engine it builds.
            perf_config = {"ais_perf_config": None, "ais_perf_config_from_engine": True}
        else:
            worker_type = (
                "prefill"
                if str(context.engine.get("mode", "aggregated")) == "disaggregated"
                else "aggregated"
            )
            perf_config = {"ais_perf_config": router_perf_config(written, worker_type)}
        result = AdapterReplaySpec(
            config=concrete,
            runtime_hooks=(
                RuntimeHookSpec(
                    provider=_HOOK.provider,
                    kind=_HOOK.kind,
                    api_version=_HOOK.api_version,
                    config={
                        "router_mode": public.policy,
                        "router_config": router_config,
                        **(
                            {"conversation_affinity": concrete["conversation_affinity"]}
                            if public.conversation_affinity is not None
                            else {}
                        ),
                        **perf_config,
                    },
                ),
            ),
        )
        if old_form and not self._warned_old_router_section:
            # The flag is set after the call: a warning that a filter turns
            # into an error was not shown, and the next compile raises it again.
            Deprecated(_NEW_ROUTER_SECTION, remove_in=REMOVE_IN).warn(
                _OLD_ROUTER_SECTION, stacklevel=2
            )
            self._warned_old_router_section = True
        return result

    def compile_recommendation(
        self,
        config: Mapping[str, JSONValue],
        context: RecommendationAdapterContext,
    ) -> AdapterSearchPlan:
        public = RouterRecommendationConfig.model_validate(config)
        plan = self._compile_public_search(public, context.sweep)
        # Only a section that writes the load model type wrote the old form.
        # Without it ais is a default choice, and those candidates read the
        # perf model of their engine.
        old_form = (
            public.ais_perf_model is None
            and _writes_load_model_type(public)
            and _may_select_ais(public)
        )
        if old_form and not self._warned_old_router_section:
            # Set after the call, as in compile_prediction.
            Deprecated(_NEW_ROUTER_SECTION, remove_in=REMOVE_IN).warn(
                _OLD_ROUTER_SECTION, stacklevel=2
            )
            self._warned_old_router_section = True
        return plan

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
        modes = _deployment_modes(context.core_search_space)
        written = public.ais_perf_model
        if policies == ["round_robin"]:
            fragment = SearchSpaceFragment(
                choices_by_branch={mode: {"mode": ["round_robin"]} for mode in modes}
            )
            return AdapterSearchPlan(
                fragment=fragment,
                state={
                    "public_schema": True,
                    "ais_perf_model": deepcopy(written),
                    "core_search_space": deepcopy(dict(context.core_search_space)),
                },
            )
        # A section that writes neither the key nor the load model type
        # searches ais as a default choice, as if it wrote {use: engine}.
        by_default = written is None and not _writes_load_model_type(public)
        if by_default:
            written = USE_ENGINE_PERF_MODEL
        # A written value is read here, once per searched worker type, and
        # the plan keeps what was read: its candidates stay the same when a
        # file the value names changes later.
        perf_configs: dict[str, JSONValue] = {}
        if written is not None and written != USE_ENGINE_PERF_MODEL:
            # A value this read rejects or a searched mode without an entry is
            # an error before the search: otherwise each candidate of that mode
            # that reads the value fails on its own, and the recommendation
            # silently comes from the other candidates or the search ends
            # without a candidate. A value that only the perf model rejects
            # still fails each candidate that reads it.
            for mode in modes:
                worker_type = "prefill" if mode == "disagg" else "aggregated"
                perf_configs[worker_type] = router_perf_config(written, worker_type)
        perf_model_state: dict[str, JSONValue] = {
            "ais_perf_model": deepcopy(written),
            "ais_perf_model_by_default": by_default,
            "ais_perf_configs": perf_configs,
        }

        # One expansion for both KV-router plan shapes: finite domains become
        # choices, continuous ones go to the sampler's float ranges.
        expanded: dict[str, list[JSONValue]] = {}
        ranges: dict[str, tuple[float, float]] = {}
        log_ranges: list[str] = []
        for name, defaults in ROUTER_KNOB_DEFAULTS.items():
            domain = expand_domain(getattr(public, name), defaults)
            if isinstance(domain, ContinuousDomain):
                ranges[name] = (domain.minimum, domain.maximum)
                if domain.log:
                    log_ranges.append(name)
            else:
                expanded[name] = domain

        if set(policies) == {"round_robin", "kv_router"}:
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
                state={
                    "public_schema": True,
                    "conditional_public": True,
                    **perf_model_state,
                    "core_search_space": deepcopy(dict(context.core_search_space)),
                },
                potential_runtime_hooks=(_HOOK,),
            )

        choices: dict[str, list[JSONValue]] = {
            "mode": ["kv_router"],
            "prefill_load_model_type": list(load_models),
            **expanded,
        }
        return AdapterSearchPlan(
            fragment=SearchSpaceFragment(
                choices_by_branch={mode: deepcopy(choices) for mode in modes},
                float_ranges_by_branch={mode: deepcopy(ranges) for mode in modes},
                log_float_ranges_by_branch={mode: list(log_ranges) for mode in modes},
            ),
            state={
                "public_schema": True,
                **perf_model_state,
                "core_search_space": deepcopy(dict(context.core_search_space)),
            },
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
                "core_search_space": deepcopy(dict(context.core_search_space)),
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
            live: _float_selection(selection, public)
            for public, live in ROUTER_PUBLIC_TO_LIVE.items()
        }
        load_model = str(selection.get("prefill_load_model_type", "none"))
        if public_schema:
            router_config["router_prefill_load_model"] = load_model
        concrete_config: dict[str, JSONValue]
        if public_schema:
            concrete_config = {
                "policy": mode,
                "prefill_load_model": {"type": load_model},
                **{
                    public: router_config[live]
                    for public, live in ROUTER_PUBLIC_TO_LIVE.items()
                },
            }
        else:
            concrete_config = {"mode": mode, **router_config}
        uses_ais = public_schema and load_model == "ais"
        # Absent in a plan made by generate_search_space, None when the
        # router section writes the ais load model without the key: both keep
        # the old sources.
        written = plan.state.get("ais_perf_model")
        ais_perf_config: dict[str, JSONValue] | None
        if not uses_ais or written is None:
            ais_perf_config = _ais_perf_config_from_candidate(
                context,
                enabled=uses_ais,
                core_search_space=plan.state.get("core_search_space", {}),
            )
        else:
            concrete_config["ais_perf_model"] = deepcopy(written)
            worker_type = (
                "prefill"
                if str(context.sample["deployment_mode"]) == "disagg"
                else "aggregated"
            )
            if written == USE_ENGINE_PERF_MODEL:
                ais_perf_config = _ais_perf_config_from_candidate(
                    context,
                    enabled=True,
                    core_search_space=plan.state.get("core_search_space", {}),
                    engine_only=True,
                )
                if ais_perf_config is None:
                    if plan.state.get("ais_perf_model_by_default"):
                        # The section did not ask for the ais load model, so
                        # the search goes on without this candidate.
                        raise InfeasibleCandidate(
                            "router.prefill_load_model.type: ais needs an AIS "
                            f"perf model on the {worker_type} engine; it has none"
                        )
                    raise ValueError(
                        "router.ais_perf_model: {use: engine} needs an AIS perf "
                        f"model on the {worker_type} engine; it has none"
                    )
            else:
                ais_perf_config = deepcopy(plan.state["ais_perf_configs"][worker_type])
        hook_config: dict[str, JSONValue] = {
            "router_mode": mode,
            "router_config": router_config,
        }
        if public_schema:
            hook_config["ais_perf_config"] = ais_perf_config
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
