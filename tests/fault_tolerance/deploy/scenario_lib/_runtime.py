# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Bridge between scenario YAML (declarative) and the framework's
# imperative Event / Report / Check / LoadConfig classes.
#
# Design principle: NO hardcoded registries.
# Every Event / Report / Check class is a @dataclass and inherits from
# its base (Event / Report / Check). At import time we walk the
# subclasses transitively and build the kind → class map automatically.
# A YAML's ``kind: ClassName`` directly instantiates ``ClassName(**params)``.
# Adding a new event/report/check class anywhere in the framework
# automatically makes it available in scenario YAML — no registry edit.
#
# The one exception is params that aren't YAML-natural (e.g.,
# StartLoad.load_config is a LoadConfig object). Those go through
# small per-class adapters declared in _PARAM_ADAPTERS.

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Optional

# Import the framework modules so __subclasses__() walks see every class.
# Order matters only for the side-effect of importing.
from tests.fault_tolerance.deploy import checks as _checks_mod  # noqa: F401
from tests.fault_tolerance.deploy import events as _events_mod  # noqa: F401
from tests.fault_tolerance.deploy import reports as _reports_mod  # noqa: F401
from tests.fault_tolerance.deploy.checks import Check
from tests.fault_tolerance.deploy.events import Event, StartLoad
from tests.fault_tolerance.deploy.reports import Report
from tests.utils.managed_load import LoadConfig

from ._schema import Admission, Router, Rung, Scenario, Shape

# --------------------------------------------------------------------------- #
# Auto-discovered class registries
# --------------------------------------------------------------------------- #


def _all_subclasses(cls) -> list[type]:
    """Walk the subclass tree depth-first. Skips abstract bases
    (those with abstract methods)."""
    out: list[type] = []
    seen: set[type] = set()
    stack = list(cls.__subclasses__())
    while stack:
        sub = stack.pop(0)
        if sub in seen:
            continue
        seen.add(sub)
        # Skip the abstract base class itself
        if not getattr(sub, "__abstractmethods__", None):
            out.append(sub)
        stack.extend(sub.__subclasses__())
    return out


def _build_registry(base: type) -> dict[str, type]:
    """``__name__`` → class. Last-one-wins on name collisions; if you
    have two classes with the same __name__ in different modules, you
    have bigger problems."""
    return {cls.__name__: cls for cls in _all_subclasses(base)}


EVENT_REGISTRY = _build_registry(Event)
REPORT_REGISTRY = _build_registry(Report)
CHECK_REGISTRY = _build_registry(Check)


# --------------------------------------------------------------------------- #
# Param adapters: convert YAML-native types to non-YAML kwargs (LoadConfig, etc.)
# --------------------------------------------------------------------------- #
#
# Per-class adapter: (params: dict, ctx: AdapterContext) → dict.
# Called BEFORE cls(**params). Lets us turn a YAML ``load_ref: warmup``
# into an actual LoadConfig built from the scenario's rung definition.


class AdapterContext:
    """What a param adapter is allowed to see — scenario context + the
    model name from the deployment."""

    def __init__(self, scenario: Scenario, served_model: str):
        self.scenario = scenario
        self.served_model = served_model
        self.rungs_by_name: dict[str, Rung] = (
            {r.name: r for r in scenario.load.rungs}
            if scenario.load and scenario.load.rungs
            else {}
        )


def _start_load_adapter(params: dict, ctx: AdapterContext) -> dict:
    """Resolve a ``load_ref: <rung_name>`` into a concrete LoadConfig
    built from the named rung's shape + common settings."""
    params = dict(params)
    if "load_ref" in params:
        rung_name = params.pop("load_ref")
        rung = ctx.rungs_by_name.get(rung_name)
        if rung is None:
            raise ValueError(
                f"{ctx.scenario.path}: StartLoad load_ref={rung_name!r} but "
                f"no matching rung in load.rungs"
            )
        if ctx.scenario.load is None:
            raise ValueError(
                f"{ctx.scenario.path}: StartLoad load_ref needs scenario.load"
            )
        params["load_config"] = build_load_config(
            rung,
            ctx.scenario.load.shape,
            ctx.scenario.load.common,
            ctx.served_model,
        )
        params.setdefault("name", rung_name)
    return params


# kind → adapter callable. Keys are class __name__.
_PARAM_ADAPTERS: dict[str, Callable[[dict, AdapterContext], dict]] = {
    "StartLoad": _start_load_adapter,
}


# --------------------------------------------------------------------------- #
# Generic kind → instance
# --------------------------------------------------------------------------- #


def _instantiate(
    kind: str,
    params: dict,
    registry: dict[str, type],
    ctx: Optional[AdapterContext] = None,
    *,
    scenario_path: str,
    bucket_name: str,
):
    cls = registry.get(kind)
    if cls is None:
        raise ValueError(
            f"{scenario_path}: unknown {bucket_name} kind {kind!r}. "
            f"Known: {sorted(registry)}"
        )
    adapter = _PARAM_ADAPTERS.get(kind)
    effective_params = adapter(params, ctx) if adapter and ctx else dict(params)
    try:
        return cls(**effective_params)
    except TypeError as e:
        # Show the field set the class actually accepts to help users
        # diagnose "unknown kwarg" errors quickly.
        if dataclasses.is_dataclass(cls):
            allowed = sorted(f.name for f in dataclasses.fields(cls))
            raise TypeError(
                f"{scenario_path}: {bucket_name} {kind} rejected kwargs "
                f"{sorted(effective_params)}. Allowed: {allowed}. ({e})"
            )
        raise


# --------------------------------------------------------------------------- #
# Public API used by test_router_modes.py
# --------------------------------------------------------------------------- #


def apply_deployment(spec, deployment) -> None:
    """Apply deployment overrides (image, units OR explicit replicas,
    model_cache, log_pvc).

    Backend-agnostic: iterates ``spec.worker_services()`` instead of
    hard-coding vllm service names, so mocker / trtllm / sglang scenarios
    work via the same path.

    Replica resolution order per service:
      1. ``deployment.replicas[<service_name>]`` if set — exact value wins
      2. Otherwise: ``template_base_replicas * deployment.units``

    Mixing modes is allowed: set replicas for some services, fall back
    to units * base for others.
    """
    worker_names = spec.worker_services()
    all_names = ["Frontend"] + worker_names
    overrides = deployment.replicas or {}

    for name in all_names:
        if name in overrides:
            spec[name].replicas = int(overrides[name])
        else:
            spec[name].replicas = spec[name].replicas * deployment.units

    if deployment.image:
        for name in all_names:
            spec[name].image = deployment.image

    if deployment.memory_request:
        for name in all_names:
            spec[name].set_memory_request(deployment.memory_request)

    if deployment.cpu_request:
        for name in all_names:
            spec[name].set_cpu_request(deployment.cpu_request)

    if deployment.model:
        # Worker services only — the Frontend has no ``--model`` arg (the
        # setter is a no-op there). ``served_model`` is read from the worker
        # spec right after this, so the load + router pick up the override.
        for name in spec.worker_services():
            spec[name].model = deployment.model

    if deployment.model_cache_pvc:
        spec.enable_model_cache(deployment.model_cache_pvc)


def apply_mocker_planner_profile_fixup(spec) -> None:
    """The bundled mocker DGD templates (examples/backends/mocker/deploy/
    {agg,disagg}.yaml) hard-code ``--planner-profile-data <path>`` to a
    profiler-style results directory. mocker tries to convert that
    directory to NPZ via ``dynamo.planner.core`` which imports
    ``pmdarima`` — not installed in the runtime images.

    Mocker's ``resolve_planner_profile_data`` accepts None and returns
    early (``npz_path=None``), causing mocker to fall back to default
    response timing — fine for memory-leak investigation where exact
    prefill latency isn't measured.

    Cleanest fix: STRIP ``--planner-profile-data`` + its value from
    every worker service's args list. No-op for non-mocker backends
    (other backends don't carry this flag).
    """
    for svc in spec.worker_services():
        spec_svc = spec[svc]._spec
        main = spec_svc.get("extraPodSpec", {}).get("mainContainer", {})
        args = main.get("args")
        if not args or "--planner-profile-data" not in args:
            continue
        # Remove flag + value pair
        idx = args.index("--planner-profile-data")
        # value follows; if missing or starts with '-' (next flag), only strip the flag
        if idx + 1 < len(args) and not args[idx + 1].startswith("-"):
            del args[idx : idx + 2]
        else:
            del args[idx]


def apply_pull_secrets(spec, secrets: list[str] = None) -> None:
    """Add image pull secret(s) to every service. Necessary on shared
    clusters (aws-dev-02, aks-dev) where private nvcr.io/nvidian/*
    images require the org's `ngc-pull-secret` and the default SA's
    secret list gets reset by a controller (so per-pod is the
    reliable place to declare).

    Idempotent: skips secrets already present on the service.
    """
    if secrets is None:
        secrets = ["ngc-pull-secret"]
    for name in ["Frontend"] + spec.worker_services():
        eps = spec[name]._spec.setdefault("extraPodSpec", {})
        ips = eps.setdefault("imagePullSecrets", [])
        for sec in secrets:
            if not any(s.get("name") == sec for s in ips):
                ips.append({"name": sec})


# Transport-level env vars that must be consistent across FE + every worker.
# Listed here so that ``apply_router`` propagates them to ALL services, not
# just Frontend — otherwise the FE-only setting causes the EventPublisher on
# each worker to silently fall back to the default transport (NATS), which
# the FE (configured for ZMQ) then ignores → 0 kv_metrics events flow worker
# → FE → busy-detection dies → routing layer's threshold-based exclusion
# becomes a no-op. Add new transport-level envs here, not to a worker-only
# block.
_TRANSPORT_LEVEL_ENVS = frozenset(
    {
        "DYN_EVENT_PLANE",
        "DYN_DISCOVERY_BACKEND",
        "DYN_STORE_KV",
    }
)


def apply_router(spec, router: Router) -> None:
    """Apply DYN_ROUTER_MODE + DYN_ROUTER_* knobs to the Frontend.

    Transport-level envs (see ``_TRANSPORT_LEVEL_ENVS``) are additionally
    propagated to every worker service so the event-plane is consistent
    across the cluster.
    """
    fe = spec["Frontend"]
    fe.set_env_var("DYN_ROUTER_MODE", router.mode)
    for key, value in router.knobs.items():
        fe.set_env_var(key, value)
        # Transport-level envs must be on every service, not just FE.
        if key in _TRANSPORT_LEVEL_ENVS:
            for svc in spec.worker_services():
                spec[svc].set_env_var(key, value)


def apply_admission(spec, admission: Optional[Admission]) -> None:
    """Apply admission-control knobs to every worker service.
    No-op if None or empty."""
    if admission is None or not admission.knobs:
        return
    for svc in spec.worker_services():
        for key, value in admission.knobs.items():
            spec[svc].set_env_var(key, value)


# --------------------------------------------------------------------------- #
# Workload-shape → LoadConfig
# --------------------------------------------------------------------------- #

# Production-shaped traffic distribution (ISL/OSL pairs with realistic
# long-tail mix). Same as the n3 routing-threshold + memory-stability default.
_PROD_SEQ_DIST = "100,200:5;500,200:15;1000,200:20;1600,200:30;3400,200:20;7000,200:10"


def build_load_config(
    rung: Rung,
    shape: Shape,
    common,
    served_model: str,
) -> LoadConfig:
    """Translate (rung, shape, common) → LoadConfig.

    Shape semantics:
      - no_prefix       — no prefix knobs → AIPerf random prompts (~0% block hit)
      - same_prefix     — shared_system_prompt_length → ~100% prefix share
      - partial_prefix  — pool of num_prefix_prompts × prefix_prompt_length
      - long_isl        — large input_tokens_mean
      - high_qps        — short input/output, high lifecycle turnover
      - custom          — pass through what the YAML sets
    """
    kwargs: dict[str, Any] = {
        "model_name": served_model,
        "tokenizer": served_model,
        "concurrency": rung.concurrency,
        "duration_minutes": rung.duration_minutes,
        "request_timeout_seconds": common.request_timeout_seconds,
        "streaming": common.streaming,
        "ignore_eos": common.ignore_eos,
        "connection_reuse_strategy": common.connection_reuse_strategy,
        "warmup_requests": common.warmup_requests,
    }
    # Open-loop: fixed arrival rate (aiperf --request-rate); concurrency stays
    # as a max-in-flight cap. ManagedLoad already emits --request-rate.
    if rung.request_rate is not None:
        kwargs["request_rate"] = rung.request_rate
    if common.goodput is not None:
        kwargs["goodput"] = list(common.goodput)
    if common.request_cancellation_rate is not None:
        kwargs["request_cancellation_rate"] = common.request_cancellation_rate
    if common.aiperf_ref is not None:
        kwargs["aiperf_ref"] = common.aiperf_ref

    s = rung.shape or shape

    # aiperf RNG seed override (per-rung). None => LoadConfig keeps the
    # historical default (100), so existing scenarios are byte-identical.
    if s.random_seed is not None:
        kwargs["random_seed"] = s.random_seed

    if s.type == "no_prefix":
        kwargs["seq_dist"] = s.seq_dist or _PROD_SEQ_DIST

    elif s.type == "same_prefix":
        kwargs["shared_system_prompt_length"] = s.shared_system_prompt_length or 2000
        kwargs["seq_dist"] = s.seq_dist or _PROD_SEQ_DIST

    elif s.type == "partial_prefix":
        kwargs["num_prefix_prompts"] = s.num_prefix_prompts or 15
        kwargs["prefix_prompt_length"] = s.prefix_prompt_length or 600
        kwargs["seq_dist"] = s.seq_dist or _PROD_SEQ_DIST

    elif s.type == "long_isl":
        if s.seq_dist:
            kwargs["seq_dist"] = s.seq_dist
        else:
            mean = s.input_tokens_mean or 80000
            kwargs["input_tokens_mean"] = mean
            kwargs["input_tokens_stddev"] = s.input_tokens_stddev or max(1, mean // 10)
            kwargs["output_tokens_mean"] = s.output_tokens_mean or 200
            kwargs["output_tokens_stddev"] = s.output_tokens_stddev or 20

    elif s.type == "high_qps":
        kwargs["input_tokens_mean"] = s.input_tokens_mean or 256
        kwargs["input_tokens_stddev"] = s.input_tokens_stddev or 32
        kwargs["output_tokens_mean"] = s.output_tokens_mean or 64
        kwargs["output_tokens_stddev"] = s.output_tokens_stddev or 8

    elif s.type == "custom":
        if s.seq_dist:
            kwargs["seq_dist"] = s.seq_dist
        for attr in (
            "num_prefix_prompts",
            "prefix_prompt_length",
            "shared_system_prompt_length",
            "input_tokens_mean",
            "input_tokens_stddev",
            "output_tokens_mean",
            "output_tokens_stddev",
        ):
            v = getattr(s, attr)
            if v is not None:
                kwargs[attr] = v

    return LoadConfig(**kwargs)


# --------------------------------------------------------------------------- #
# Event / Report / Check materialisation
# --------------------------------------------------------------------------- #


def build_scenario_events(scenario: Scenario, served_model: str) -> list[Event]:
    """Materialise the scenario's event sequence.

    If scenario.events is non-empty, instantiate each via the auto
    EVENT_REGISTRY (with StartLoad load_ref resolution).

    Otherwise generate the canonical sequence from scenario.load.rungs:
      1. WaitForModelReady(timeout=2400)
      2. ResourcePoller(services=[Frontend, VllmPrefillWorker, VllmDecodeWorker], interval_s=10)
      3. For each rung: StartLoad + WaitForLoadCompletion
    """
    if scenario.events:
        ctx = AdapterContext(scenario, served_model)
        return [
            _instantiate(
                e.kind,
                e.params,
                EVENT_REGISTRY,
                ctx,
                scenario_path=scenario.path or "<unknown>",
                bucket_name="event",
            )
            for e in scenario.events
        ]
    return _generate_default_events(scenario, served_model)


def _generate_default_events(scenario: Scenario, served_model: str) -> list[Event]:
    if scenario.load is None or not scenario.load.rungs:
        raise ValueError(
            f"{scenario.path}: scenario has no events: and no load.rungs — "
            f"nothing to run"
        )
    # Use the registry to construct so dataclass field-validation runs.
    WaitForModelReady = EVENT_REGISTRY["WaitForModelReady"]
    ResourcePoller = EVENT_REGISTRY["ResourcePoller"]
    WaitForLoadCompletion = EVENT_REGISTRY["WaitForLoadCompletion"]

    events: list[Event] = [
        WaitForModelReady(timeout=2400),
        ResourcePoller(
            services=["Frontend", "VllmPrefillWorker", "VllmDecodeWorker"],
            interval_s=10,
        ),
    ]
    for rung in scenario.load.rungs:
        load_config = build_load_config(
            rung, scenario.load.shape, scenario.load.common, served_model
        )
        events.append(StartLoad(load_config=load_config, name=rung.name))
        events.append(WaitForLoadCompletion(name=rung.name))
    return events


def build_reports(scenario: Scenario) -> list:
    """Instantiate the scenario's reports via REPORT_REGISTRY."""
    return [
        _instantiate(
            r.kind,
            r.params,
            REPORT_REGISTRY,
            None,
            scenario_path=scenario.path or "<unknown>",
            bucket_name="report",
        )
        for r in scenario.reports
    ]


def build_checks(scenario: Scenario) -> list:
    """Instantiate the scenario's checks via CHECK_REGISTRY."""
    return [
        _instantiate(
            c.kind,
            c.params,
            CHECK_REGISTRY,
            None,
            scenario_path=scenario.path or "<unknown>",
            bucket_name="check",
        )
        for c in scenario.checks
    ]
