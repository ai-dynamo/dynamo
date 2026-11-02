# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frontend router startup parsed from the DGD spec, and the spec premises.

Live retune replays each window in DGDR with AISimulate's router defaults
plus the searched knobs, while the live frontends start from whatever their
argv and env say. The two agree only when the frontend leaves every
unsearched routing field at its default (premise P4), so the planner reads
the frontend's startup from the DynamoGraphDeployment spec, never from a
running pod:

- ``find_frontend`` picks the one standalone frontend component.
- ``parse_frontend`` rebuilds the frontend's effective argv (the operator's
  ``command``/``args`` defaulting) and env (operator-injected, graph ``env``
  and the main container's, with Kubernetes ``$(VAR)`` expansion), then
  parses them with the frontend's own parser (``build_frontend_parser``),
  which reads env defaults from that env alone, so the planner process's
  environment is neither read nor modified. A shell-wrapped command, a
  ``--router-policy-config`` file, router env set through ``valueFrom`` and
  an argument whose unreadable ``$(VAR)`` could set a router flag make the
  frontend unparseable. ``envFrom`` sources are assumed to set no router
  env; ``evaluate_premises`` warns about them, and P14 compares the
  routers' policy parameters with the parsed startup.
- ``evaluate_premises`` checks P4 and P7-P10 and warns on P6 and P7; P14
  is the router actuator's. P4 also parses every worker component's router
  flags and env the same way: a worker that resolves a router mode
  advertises its own router configuration, which replaces the frontend's
  for its worker set. ``p4_ok`` alone decides whether a retune controller
  can be built.

Every function is pure (a dict in, a value out, no Kubernetes calls) and
never reads the planner's own environment.
"""

from __future__ import annotations

import argparse
import json
import logging
import posixpath
import re
import shlex
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Optional

from dynamo.common.configuration.groups.kv_router_args import (
    _KV_ROUTER_FIELDS,
    KvRouterConfigBase,
)
from dynamo.common.configuration.groups.router_args import (
    RouterConfigBase,
    add_worker_router_arguments,
)
from dynamo.common.configuration.utils import ArgEnv
from dynamo.frontend.frontend_args import build_frontend_parser
from dynamo.llm import KvRouterConfig
from dynamo.planner.retune.domains import (
    OVERLAP_SCORE_CREDIT,
    expand_router_domain,
    grid_points,
)
from dynamo.planner.retune.live.contracts import FrontendStartup, PremiseReport

if TYPE_CHECKING:
    from dynamo.planner.config.planner_config import PlannerConfig, RetuneConfig

logger = logging.getLogger(__name__)

FRONTEND_TYPE = "frontend"
PLANNER_TYPE = "planner"
EPP_TYPE = "epp"
# Component types that run inference workers; v1beta1 ``type`` is optional.
WORKER_TYPES = frozenset({"worker", "prefill", "decode", ""})

MAIN_CONTAINER = "main"
# The operator's frontend invocation (component_frontend.go GetBaseContainer).
DEFAULT_FRONTEND_COMMAND = ("python3",)
DEFAULT_FRONTEND_ARGS = ("-m", "dynamo.frontend")
FRONTEND_MODULE = "dynamo.frontend"

# Env the operator injects into every component (component_common.go
# getCommonContainer); user env of the same name overrides it.
ENV_PARENT_DGD_NAME = "DYN_PARENT_DGD_K8S_NAME"
ENV_PARENT_DGD_NAMESPACE = "DYN_PARENT_DGD_K8S_NAMESPACE"
_OPERATOR_FIELD_REF_ENV = ("POD_NAME", "POD_NAMESPACE", "POD_UID")

# The frontend's request trace recipe (P10).
ENV_POD_NAME = "DYN_POD_NAME"
ENV_TRACE = "DYN_REQUEST_TRACE"
ENV_TRACE_SINKS = "DYN_REQUEST_TRACE_SINKS"
ENV_TRACE_FORMAT = "DYN_REQUEST_TRACE_FILE_FORMAT"
ENV_TRACE_RECORDS = "DYN_REQUEST_TRACE_RECORDS"
ENV_TRACE_PATH = "DYN_REQUEST_TRACE_FILE_PATH"
# lib/truthy is_truthy; lib/llm/src/request_trace/config.rs parse_file_format.
_TRUTHY = frozenset({"1", "true", "on", "yes"})
_GZ_FORMATS = frozenset({"jsonl_gz", "jsonl.gz"})

ENV_WORKER_SELECTION_POLICY = "DYN_ROUTER_WORKER_SELECTION_POLICY"
ENV_LOAD_AWARE = "DYN_ROUTER_LOAD_AWARE"
# A worker that resolves a router mode (router_args.add_worker_router_arguments)
# advertises a RouterConfig in its model card.
ENV_ROUTER_MODE = "DYN_ROUTER_MODE"
_ROUTER_MODE_DEST = "router_mode"
_ROUTER_ENV_PREFIX = "DYN_ROUTER_"
# Router env the Rust router reads directly (no frontend flag) that tunes
# transport or startup only, never which worker is picked.
_TRANSPORT_ROUTER_ENV = frozenset(
    {
        "DYN_ROUTER_ACTIVE_SEQUENCE_DIRECT_ZMQ",
        "DYN_ROUTER_ACTIVE_SEQUENCE_ZMQ_RCVHWM",
        "DYN_ROUTER_MODEL_CARD_WAIT_SECS",
        "DYN_ROUTER_ZMQ_ENDPOINTS_PER_SUB",
    }
)
# The deprecated overlap_score_weight: KvRouterArgGroup reads the first name
# when it is set and the second otherwise.
_OVERLAP_WEIGHT_ENV = ("DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT", "DYN_OVERLAP_SCORE_WEIGHT")

# RouterConfig (not KvRouterConfig) fields that change routing or admission:
# session affinity pins requests and the busy thresholds reject them. The
# DGDR replay models neither.
_ROUTER_ADMISSION_FIELDS = (
    "session_affinity_ttl_secs",
    "active_decode_blocks_threshold",
    "active_prefill_tokens_threshold",
    "active_prefill_tokens_threshold_frac",
)
# KvRouterConfig fields serde skips (to_json drops them and from_json
# rejects them), so they are checked on the parsed flags.
_SERDE_SKIPPED_DEFAULTS: dict[str, Any] = {
    "router_approximate_cache_policy": "ttl",
    "router_prefill_policy": None,
    "router_decode_policy": None,
}
# KvRouterConfig JSON keys a frontend may set to a non-default value: they
# do not change which worker the router picks, or another premise covers
# them.
TOLERATED_ROUTER_FIELDS = frozenset(
    {
        "router_replica_sync",  # P8
        "router_event_threads",  # indexer thread count
        # Hashing of router-derived active-sequence identities.
        "router_tracking_hash",
        "router_tracking_key_file",
        "router_tracking_key_id",
        "serve_indexer",  # serves this router's indexer to other routers
        "enable_session_prefix_index",  # tracked, not consumed by routing
        "skip_initial_worker_wait",  # startup only
    }
)
# Fields with their own P4 message.
_LOAD_MODEL = "router_prefill_load_model"
_CONDITIONAL_DISAGG = "conditional_disagg_enabled"

_TLS_DESTS = ("tls_cert_path", "tls_key_path")
# Parsed destinations whose value the startup depends on.
_RELEVANT_DESTS = frozenset(
    {
        *_KV_ROUTER_FIELDS,
        *_ROUTER_ADMISSION_FIELDS,
        *_TLS_DESTS,
        "router_mode",
        "session_affinity_mode",
        "load_aware",
        "conditional_disagg_config",
    }
)
# runtime_status.go ResolveComponentRuntimeStatus, in its priority order.
_MODEL_FLAGS = ("--served-model-name", "--model-name", "--model", "--model-path")
_SHELLS = frozenset({"sh", "bash", "dash", "zsh", "ash", "ksh"})
# Python options that may precede ``-m``; the second set takes a value.
_PYTHON_FLAGS = frozenset({"-u", "-B", "-E", "-I", "-O", "-OO", "-P", "-q", "-s", "-S"})
_PYTHON_FLAGS_WITH_VALUE = frozenset({"-X", "-W"})
# Marks an argument part that references env the planner cannot read.
_OPAQUE = "\x00opaque:"
_OPAQUE_REPR = repr(_OPAQUE)[1:-1]


class FrontendParseError(ValueError):
    """The frontend's router configuration cannot be derived from the spec."""


# ------------------------------------------------------------------ API


def worker_types(planner_mode: str) -> tuple[str, ...]:
    """Router worker types of a planner mode, as the retune controller uses them."""

    return ("prefill", "decode") if planner_mode == "disagg" else ("aggregated",)


def canonical_json(value: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def find_frontend(dgd: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return the name and spec of the DGD's one standalone frontend component.

    Reads the v1beta1 ``spec.components`` list (the planner reads every DGD
    as v1beta1). Raises ``FrontendParseError`` ("P7: ...") unless there
    is exactly one standalone frontend, naming an EPP or ``frontendSidecar``
    layout when that is why.
    """

    components = _components(dgd)
    frontends = [c for c in components if _type(c) == FRONTEND_TYPE]
    if len(frontends) == 1:
        return str(frontends[0].get("name", "")), dict(frontends[0])
    if frontends:
        names = ", ".join(sorted(str(c.get("name")) for c in frontends))
        raise FrontendParseError(
            f"P7: the DGD has {len(frontends)} frontend components ({names}); "
            "live retune needs exactly one"
        )
    sidecars = _sidecar_components(components)
    if sidecars:
        raise FrontendParseError(
            f"P7: the frontend runs as a frontendSidecar of {', '.join(sidecars)}; "
            "live retune needs a standalone frontend component"
        )
    if any(_type(c) == EPP_TYPE for c in components):
        raise FrontendParseError(
            "P7: the DGD routes through an EPP component and has no standalone "
            "frontend component"
        )
    raise FrontendParseError("P7: the DGD has no frontend component")


def served_model_names(
    dgd: Mapping[str, Any],
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Served model name of each worker component, and the undetermined ones.

    Uses the operator's ``status.components[<name>].servedModelName`` while
    the status is current, else the first of ``--served-model-name``,
    ``--model-name``, ``--model`` and ``--model-path`` in the main
    container's command and args (the operator's order), after Kubernetes
    ``$(VAR)`` expansion against the component's env, skipping values that
    reference env the planner cannot read.
    """

    projected = _projected_status(dgd)
    models: dict[str, str] = {}
    unknown: list[str] = []
    for component in _components(dgd):
        if _type(component) not in WORKER_TYPES:
            continue
        name = str(component.get("name", ""))
        model = (projected.get(name) or {}).get("servedModelName") or _flag_model(
            _model_flags(component, _component_env(dgd, component))
        )
        if model:
            models[name] = str(model)
        else:
            unknown.append(name)
    return models, tuple(sorted(unknown))


def served_model_name(dgd: Mapping[str, Any]) -> Optional[str]:
    """The DGD's served model when exactly one name is known, else ``None``."""

    names = set(served_model_names(dgd)[0].values())
    return next(iter(names)) if len(names) == 1 else None


def parse_frontend(
    dgd: Mapping[str, Any], *, planner_mode: str
) -> FrontendStartup | str:
    """Parse the frontend's startup router configuration from the DGD spec.

    Returns the startup, or a ``"P4: ..."`` / ``"P7: ..."`` reason when it
    cannot be derived. ``router_config`` holds only the keys whose serde
    value differs from ``KvRouterConfig()``; ``startup_maps`` resolves
    ``policy_parameters`` for each worker type of ``planner_mode``. A
    frontend that parses but violates other P4 conditions (a non-KV router
    mode, a non-default field) still returns its startup; see
    ``evaluate_premises``.
    """

    analysis = _analyze(dgd, planner_mode)
    if analysis.startup is None:
        return analysis.error or "P4: the frontend configuration cannot be parsed"
    return analysis.startup


def evaluate_premises(
    dgd: Mapping[str, Any], *, planner_config: PlannerConfig
) -> PremiseReport:
    """Evaluate the spec premises P4 and P7-P10, and the P6 and P7 warnings.

    Parses the frontend as ``parse_frontend`` does (cached, so cheap) and
    reports its startup; P4 also covers what a ``FrontendStartup`` cannot
    carry (serde-skipped flags, session affinity, unread router env) and
    fails for a worker component that advertises its own router
    configuration. A malformed DGD never raises; every problem becomes a
    reason. Raises ``ValueError`` only when ``planner_config.retune.dgdr`` is
    unset.
    """

    retune = planner_config.retune
    if retune is None or retune.dgdr is None:
        raise ValueError("evaluate_premises requires planner_config.retune.dgdr")
    dgdr = retune.dgdr
    mode = str(planner_config.mode)
    analysis = _analyze(dgd, mode)
    startup = analysis.startup
    components = _components(dgd)
    p4: list[str] = []
    others: list[str] = []
    warnings: list[str] = []

    # P4: a parsed KV router whose unsearched fields are the defaults and
    # whose startup passes the retune controller's constructor checks.
    if startup is None:
        p4.append(analysis.error or "P4: the frontend configuration cannot be parsed")
    else:
        p4.extend(analysis.p4_violations)
        p4.extend(_startup_errors(startup, retune, mode))
        if dgdr.startup_router_config is not None:
            mismatch = _startup_config_mismatch(dgdr.startup_router_config, startup)
            if mismatch is not None:
                p4.append(mismatch)
    # A worker's advertised router configuration replaces the frontend's for
    # its worker set (watcher.rs effective_router_config), whatever the
    # frontend's flags say.
    p4.extend(_worker_router_errors(dgd, components, warnings))

    # P7: one served model, one standalone frontend, no EPP, no TLS.
    models, unknown = served_model_names(dgd)
    distinct = sorted(set(models.values()))
    if len(distinct) > 1:
        others.append(
            f"P7: the DGD serves {len(distinct)} models ({', '.join(distinct)}); "
            "live retune needs one"
        )
    if unknown:
        warnings.append(
            f"P7: cannot determine the served model of {', '.join(unknown)}; "
            "live retune needs the DGD to serve one model"
        )
    # Every DGDR Run simulates retune.dgdr.model; an alias given with
    # --served-model-name can hide the HF id, so a miss only warns.
    names = _worker_model_names(dgd, components)
    if names and dgdr.model not in names:
        warnings.append(
            f"P7: retune.dgdr.model {dgdr.model!r} is not among the workers' "
            f"--model/--model-path/--served-model-name values "
            f"({', '.join(sorted(names))}); every DGDR Run simulates "
            "retune.dgdr.model, so make sure it is the model the DGD serves"
        )
    if analysis.component is None:
        if analysis.error is not None and analysis.error.startswith("P7"):
            others.append(analysis.error)
    else:
        if any(_type(c) == EPP_TYPE for c in components):
            others.append("P7: the DGD routes through an EPP component (GAIE)")
        sidecars = _sidecar_components(components)
        if sidecars:
            others.append(f"P7: {', '.join(sidecars)} run a frontendSidecar")
        if analysis.tls:
            others.append(
                "P7: the frontend serves TLS (--tls-cert-path / DYN_TLS_CERT_PATH)"
            )

    if analysis.env is not None:
        for source in analysis.env.env_from:
            if _env_from_may_set_router_env(source):
                warnings.append(
                    f"P4: the frontend container uses envFrom {_describe_env_from(source)}; "
                    "the planner cannot read it and assumes it sets no router env "
                    "(P14 still checks the routers' policy parameters)"
                )

    # P8: one frontend replica, or synchronized router replicas.
    if startup is not None and startup.replicas > 1 and not startup.replica_sync:
        others.append(
            f"P8: the frontend has {startup.replicas} replicas without "
            "router_replica_sync; AISimulate models a single router"
        )

    # P9: worker replicas only change through the DGD spec, and one planner.
    for component in components:
        if _type(component) in WORKER_TYPES and _scaling_adapter(component):
            others.append(
                f"P9: worker component {component.get('name')} has a "
                "scalingAdapter; its replicas could change during a window"
            )
    others.extend(_planner_replica_errors(components, warnings))

    # P10: the frontend records request_end traces onto the claim.
    if analysis.component is not None and analysis.env is not None:
        others.extend(
            _trace_recipe_errors(
                analysis,
                dgd_name=_dgd_name(dgd),
                claim_name=dgdr.claim_name,
                segments_dir=dgdr.segments_dir,
                warnings=warnings,
            )
        )

    # P6 (warn only): AISimulate resolves the model by its HF id.
    warning = _model_warning(dgdr.model)
    if warning is not None:
        warnings.append(warning)

    reasons = tuple(_unique([*p4, *others]))
    return PremiseReport(
        p4_ok=not p4,
        spec_ok=not reasons,
        reasons=reasons,
        warnings=tuple(_unique(warnings)),
        startup=startup,
    )


# --------------------------------------------------------- frontend facts


@dataclass(frozen=True)
class _Env:
    """A component main container's env as far as the planner can know it.

    ``entries`` is the merged raw env keyed by name (operator-injected, then
    graph ``env``, then the container's; later wins). ``values`` holds each
    name whose value the planner can compute, after ``$(VAR)`` expansion;
    ``opaque`` the names it cannot (``valueFrom``, or a reference to one).
    """

    entries: Mapping[str, Mapping[str, Any]]
    values: Mapping[str, str]
    opaque: frozenset[str]
    env_from: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class _Analysis:
    component: Optional[str] = None
    spec: Optional[Mapping[str, Any]] = None
    env: Optional[_Env] = None
    startup: Optional[FrontendStartup] = None
    error: Optional[str] = None
    # P4 violations a FrontendStartup cannot show.
    p4_violations: tuple[str, ...] = ()
    tls: bool = False


@dataclass(frozen=True)
class _Parsed:
    """The cacheable outcome of parsing one frontend argv and env."""

    error: Optional[str] = None
    router_mode: str = ""
    router_config: Optional[Mapping[str, Any]] = None
    startup_maps: Optional[Mapping[str, Mapping[str, float]]] = None
    replica_sync: bool = False
    p4_violations: tuple[str, ...] = ()
    tls: bool = False


def _analyze(dgd: Mapping[str, Any], planner_mode: str) -> _Analysis:
    try:
        name, spec = find_frontend(dgd)
    except FrontendParseError as error:
        return _Analysis(error=str(error))
    env = _component_env(dgd, spec)
    try:
        argv = _effective_argv(spec)
    except FrontendParseError as error:
        return _Analysis(component=name, spec=spec, env=env, error=str(error))
    parsed = _parse_cached(
        canonical_json(
            {
                "argv": list(argv),
                "values": dict(env.values),
                "opaque": sorted(env.opaque),
            }
        ),
        worker_types(planner_mode),
    )
    if parsed.error is not None:
        return _Analysis(component=name, spec=spec, env=env, error=parsed.error)
    assert parsed.router_config is not None and parsed.startup_maps is not None
    router_config = dict(parsed.router_config)
    startup_maps = {wt: dict(params) for wt, params in parsed.startup_maps.items()}
    startup = FrontendStartup(
        router_mode=parsed.router_mode,
        router_config=router_config,
        startup_maps=startup_maps,
        frontend_component=name,
        replicas=_replicas(spec),
        replica_sync=parsed.replica_sync,
        # Replicas stay out: scaling the frontend is P8's concern and must not
        # reset the persisted state.
        fingerprint_source=canonical_json(
            {
                "router_mode": parsed.router_mode,
                "router_config": router_config,
                "startup_maps": startup_maps,
            }
        ),
    )
    return _Analysis(
        component=name,
        spec=spec,
        env=env,
        startup=startup,
        p4_violations=parsed.p4_violations,
        tls=parsed.tls,
    )


@lru_cache(maxsize=32)
def _parse_cached(key: str, wts: tuple[str, ...]) -> _Parsed:
    """Parse once per distinct frontend (revalidation re-parses every round)."""

    inputs = json.loads(key)
    try:
        return _parse(
            inputs["argv"],
            inputs["values"],
            frozenset(inputs["opaque"]),
            wts,
        )
    except FrontendParseError as error:
        return _Parsed(error=str(error))
    except Exception as error:  # noqa: BLE001 - fail closed, never crash the planner
        logger.exception("live retune could not parse the frontend configuration")
        return _Parsed(
            error=(
                "P4: the planner failed to parse the frontend configuration "
                f"({type(error).__name__}: {error})"
            )
        )


def _parse(
    argv: Sequence[str],
    values: Mapping[str, str],
    opaque: frozenset[str],
    wts: tuple[str, ...],
) -> _Parsed:
    args = _frontend_args(argv)
    parser, registered = _frontend_parser(values)

    def relevant(name: str) -> bool:
        if name in registered:
            return registered[name] in _RELEVANT_DESTS
        return name in _OVERLAP_WEIGHT_ENV or (
            name.startswith(_ROUTER_ENV_PREFIX) and name not in _TRANSPORT_ROUTER_ENV
        )

    # Router env the planner cannot read makes the startup unknowable.
    hidden = sorted(name for name in opaque if relevant(name))
    if hidden:
        raise FrontendParseError(
            f"P4: frontend router env {', '.join(hidden)} is set through valueFrom "
            "(or references such a value); the planner cannot read it, so set "
            "it with a literal value"
        )

    # An unreadable $(VAR) becomes a marker the parser carries as a value: a
    # non-router flag keeps it unconverted, a router flag fails the parse,
    # and one left over where a flag could stand fails below.
    expanded = [_expand_arg(arg, values, opaque) for arg in args]
    namespace, unknown = parser.parse_known_args(expanded)
    hidden_flag = _opaque_flag_position(unknown)
    if hidden_flag is not None:
        original = dict(zip(expanded, args)).get(hidden_flag, hidden_flag)
        raise FrontendParseError(
            f"P4: frontend arg {original!r} references env the planner cannot "
            "read and is not the value of a frontend flag, so it could set a "
            "router flag"
        )
    unresolved = sorted(
        dest
        for dest, value in vars(namespace).items()
        if dest in _RELEVANT_DESTS and _OPAQUE in str(value)
    )
    if unresolved:
        raise FrontendParseError(
            f"P4: frontend args set {', '.join(unresolved)} from env the planner "
            "cannot read"
        )

    view = _RouterView.from_cli_args(namespace)
    # FrontendConfig.validate: the load-aware preset implies KV routing.
    if view.load_aware:
        view.router_mode = "kv"
    if view.router_policy_config is not None:
        raise FrontendParseError(
            "P4: the frontend sets --router-policy-config "
            f"({view.router_policy_config}); the planner cannot read that file, "
            "and live retune needs the builtin default policy"
        )
    try:
        view.apply_load_aware_preset()
        view.apply_conditional_disagg_config()
        kwargs = {name: getattr(view, name) for name in _KV_ROUTER_FIELDS}
        config = KvRouterConfig(**kwargs)
    except (ValueError, TypeError) as error:
        raise FrontendParseError(
            f"P4: the frontend's router configuration is invalid: {error}"
        ) from None

    full = json.loads(config.to_json())
    defaults = json.loads(KvRouterConfig().to_json())
    router_config = {
        key: value
        for key, value in full.items()
        if key not in defaults or defaults[key] != value
    }
    try:
        maps = _policy_maps(canonical_json(router_config), wts)
    except ValueError as error:
        raise FrontendParseError(
            f"P4: cannot resolve the frontend's startup policy parameters: {error}"
        ) from None

    violations: list[str] = []
    if kwargs["overlap_score_weight"] is not None:
        violations.append(
            "P4: the frontend sets the deprecated overlap_score_weight, which "
            "overrides prefill_load_scale"
        )
    for name, default in _SERDE_SKIPPED_DEFAULTS.items():
        if kwargs[name] != default:
            violations.append(
                f"P4: the frontend sets {name}={kwargs[name]!r} (default "
                f"{default!r}), which live retune cannot model"
            )
    if view.load_aware:
        violations.append(
            f"P4: the frontend enables the load-aware preset (--load-aware / "
            f"{ENV_LOAD_AWARE})"
        )
    for name in _ROUTER_ADMISSION_FIELDS:
        value = getattr(view, name)
        if value is not None:
            violations.append(
                f"P4: the frontend sets {name}={value!r}; live retune cannot "
                "model session affinity or busy-worker rejection"
            )
    for name in sorted(values):
        if (
            not name.startswith(_ROUTER_ENV_PREFIX)
            or name in registered
            or name in _TRANSPORT_ROUTER_ENV
        ):
            continue
        if name == ENV_WORKER_SELECTION_POLICY:
            violations.append(
                f"P4: the frontend sets {name}, which selects a worker-selection "
                "policy instance instead of the builtin default policy"
            )
        else:
            violations.append(
                f"P4: the frontend sets {name}, a router setting live retune "
                "cannot model"
            )

    return _Parsed(
        router_mode=_router_mode(str(view.router_mode)),
        router_config=router_config,
        startup_maps=maps,
        replica_sync=bool(full.get("router_replica_sync", False)),
        p4_violations=tuple(violations),
        tls=any(getattr(view, dest, None) for dest in _TLS_DESTS),
    )


class _RouterView(RouterConfigBase, KvRouterConfigBase):
    """The router fields of ``FrontendConfig``, without its other checks."""

    tls_cert_path: Optional[str] = None
    tls_key_path: Optional[str] = None


@lru_cache(maxsize=256)
def _policy_maps(
    router_config_json: str, wts: tuple[str, ...]
) -> dict[str, dict[str, float]]:
    config = KvRouterConfig.from_json(router_config_json)
    return {
        wt: {key: float(value) for key, value in config.policy_parameters(wt).items()}
        for wt in wts
    }


def _router_mode(cli_mode: str) -> str:
    """``--router-mode`` in the ``RetuneEngineSpec.router_mode`` spelling."""

    return "kv_router" if cli_mode == "kv" else cli_mode.replace("-", "_")


# ----------------------------------------------------------- argv and env


def _effective_argv(spec: Mapping[str, Any]) -> tuple[str, ...]:
    """The frontend container's argv after the operator's defaulting.

    graph.go: an authored frontend ``command`` drops the default args; with
    none, the command stays ``python3`` and the args default to
    ``-m dynamo.frontend`` unless authored.
    """

    if spec.get("roles"):
        raise FrontendParseError(
            "P4: the frontend component uses roles; live retune needs a "
            "single-pod frontend"
        )
    main = _main_container(spec)
    command = [str(part) for part in main.get("command") or []]
    args = [str(part) for part in main.get("args") or []]
    if command:
        return tuple(command + args)
    return (*DEFAULT_FRONTEND_COMMAND, *(args or DEFAULT_FRONTEND_ARGS))


def _frontend_args(argv: Sequence[str]) -> list[str]:
    """The arguments after ``python -m dynamo.frontend``; anything else fails."""

    if not argv:
        raise FrontendParseError("P4: the frontend container has an empty command")
    program = posixpath.basename(argv[0])
    if program in _SHELLS or (len(argv) > 1 and argv[1] == "-c"):
        raise FrontendParseError(
            f"P4: the frontend command is shell-wrapped ({' '.join(argv[:2])} ...); "
            "write it as command: [python3] and args: [-m, dynamo.frontend, ...]"
        )
    if not program.startswith("python"):
        raise FrontendParseError(
            f"P4: the frontend command starts with {argv[0]!r}, not a python "
            f"interpreter running -m {FRONTEND_MODULE}"
        )
    index = 1
    while index < len(argv):
        token = argv[index]
        if token in _PYTHON_FLAGS:
            index += 1
            continue
        if token in _PYTHON_FLAGS_WITH_VALUE:
            index += 2
            continue
        if token == "-m" and index + 1 < len(argv):
            module, rest = argv[index + 1], index + 2
        elif token.startswith("-m") and len(token) > 2:
            module, rest = token[2:], index + 1
        else:
            break
        if module == FRONTEND_MODULE:
            return list(argv[rest:])
        break
    raise FrontendParseError(
        f"P4: the frontend command {' '.join(argv[:4])!r} does not run "
        f"python -m {FRONTEND_MODULE}"
    )


def _component_env(dgd: Mapping[str, Any], spec: Mapping[str, Any]) -> _Env:
    """Merge a component's env the way the operator renders it.

    The operator injects ``DYN_PARENT_DGD_K8S_*`` and ``POD_*``, MergeEnvs
    overlays graph ``env`` and the main container's ``env`` and sorts the
    result by name; Kubernetes expands ``$(VAR)`` against earlier names.
    """

    metadata = dgd.get("metadata") or {}
    entries: dict[str, Mapping[str, Any]] = {
        ENV_PARENT_DGD_NAME: {
            "name": ENV_PARENT_DGD_NAME,
            "value": str(metadata.get("name", "")),
        },
        ENV_PARENT_DGD_NAMESPACE: {
            "name": ENV_PARENT_DGD_NAMESPACE,
            "value": str(metadata.get("namespace", "")),
        },
    }
    for name in _OPERATOR_FIELD_REF_ENV:
        entries[name] = {"name": name, "valueFrom": {"fieldRef": {}}}
    main = _main_container(spec)
    for entry in [*_graph_env(dgd), *(main.get("env") or [])]:
        if isinstance(entry, Mapping) and entry.get("name"):
            entries[str(entry["name"])] = entry

    values: dict[str, str] = {}
    opaque: set[str] = set()
    for name in sorted(entries):
        entry = entries[name]
        expanded = (
            None
            if entry.get("valueFrom") is not None
            else _expand(str(entry.get("value") or ""), values, opaque)
        )
        if expanded is None:
            opaque.add(name)
        else:
            values[name] = expanded
    env_from = tuple(
        source for source in main.get("envFrom") or [] if isinstance(source, Mapping)
    )
    return _Env(
        entries=entries,
        values=values,
        opaque=frozenset(opaque),
        env_from=env_from,
    )


def _expand(
    value: str,
    values: Mapping[str, str],
    opaque: Iterable[str],
    *,
    mark_opaque: bool = False,
) -> Optional[str]:
    """Kubernetes ``$(VAR)`` expansion.

    ``$$`` escapes a ``$`` and a reference to an undefined name stays as
    written. A reference to an opaque name makes the result ``None``, or,
    with ``mark_opaque``, is replaced by a marker.
    """

    hidden = set(opaque)
    out: list[str] = []
    index = 0
    while index < len(value):
        if value.startswith("$$", index):
            out.append("$")
            index += 2
            continue
        if value.startswith("$(", index):
            close = value.find(")", index + 2)
            if close != -1:
                name = value[index + 2 : close]
                if name in hidden:
                    if not mark_opaque:
                        return None
                    out.append(f"{_OPAQUE}{name}\x00")
                else:
                    out.append(values.get(name, value[index : close + 1]))
                index = close + 1
                continue
        out.append(value[index])
        index += 1
    return "".join(out)


def _expand_arg(arg: str, values: Mapping[str, str], opaque: Iterable[str]) -> str:
    """Expand one argument, marking opaque references for the parser."""

    expanded = _expand(arg, values, opaque, mark_opaque=True)
    assert expanded is not None
    return expanded


def _opaque_flag_position(unknown: Sequence[str]) -> Optional[str]:
    """A left-over argument whose unreadable env could stand for a flag.

    The frontend parser consumes the value of every flag it knows, so a
    marked argument it leaves over sits where a flag could, unless it
    follows a left-over option (a flag the frontend hands to its chat
    processor, and this its value). A left-over option whose name is
    marked always could.
    """

    previous: Optional[str] = None
    for token in unknown:
        if token.startswith("-"):
            if _OPAQUE in token.split("=", 1)[0]:
                return token
        elif token.startswith(_OPAQUE) and not (
            previous is not None
            and previous.startswith("-")
            and "=" not in previous
            and _OPAQUE not in previous
        ):
            return token
        previous = token
    return None


def _opaque_reference(text: str) -> Optional[str]:
    """``$(NAME)`` for the first opaque marker in ``text``, or ``None``.

    argparse quotes a rejected value with ``repr``, so the marker may appear
    escaped.
    """

    for marker, end_marker in ((_OPAQUE, "\x00"), (_OPAQUE_REPR, "\\x00")):
        index = text.find(marker)
        if index != -1:
            start = index + len(marker)
            end = text.find(end_marker, start)
            return f"$({text[start:end] if end != -1 else text[start:]})"
    return None


def _env_from_may_set_router_env(source: Mapping[str, Any]) -> bool:
    """Whether an envFrom source could define a ``DYN_*`` name."""

    prefix = str(source.get("prefix") or "")
    return "DYN_".startswith(prefix) or prefix.startswith("DYN_")


def _describe_env_from(source: Mapping[str, Any]) -> str:
    for kind in ("configMapRef", "secretRef"):
        reference = source.get(kind)
        if isinstance(reference, Mapping):
            return f"{kind} {reference.get('name')!r}"
    return canonical_json(dict(source))


# ---------------------------------------------------- frontend arg parser


def _parse_error_message(message: str) -> str:
    reference = _opaque_reference(message)
    if reference is None:
        return f"P4: cannot parse the frontend args: {message.strip()}"
    return (
        f"P4: a frontend router arg references {reference}, which the planner "
        "cannot read"
    )


def _worker_parse_error_message(message: str) -> str:
    reference = _opaque_reference(message)
    if reference is None:
        return f"has router flags the planner cannot parse ({message.strip()})"
    return (
        f"has a router flag that references {reference}, which the planner "
        "cannot read"
    )


class _ArgsParser(argparse.ArgumentParser):
    """An ``ArgumentParser`` that raises instead of printing and exiting.

    A value carrying the opaque marker skips type conversion and the
    choices check unless its destination is in ``relevant``, so an
    unreadable value of a flag that does not matter here parses, while one
    of a flag that does fails with a message naming the reference.
    """

    def __init__(
        self,
        *,
        relevant: frozenset[str] = _RELEVANT_DESTS,
        describe: Callable[[str], str] = _parse_error_message,
    ) -> None:
        super().__init__(add_help=False)
        self._relevant = relevant
        self._describe = describe

    def error(self, message: str):  # type: ignore[override]
        raise FrontendParseError(self._describe(message))

    def exit(self, status: int = 0, message: Optional[str] = None):  # type: ignore[override]
        raise FrontendParseError(
            self._describe(message or f"argument parsing exited with {status}")
        )

    def _opaque_irrelevant(self, action: argparse.Action, value: Any) -> bool:
        return (
            isinstance(value, str)
            and _OPAQUE in value
            and action.dest not in self._relevant
        )

    def _get_value(self, action, arg_string):  # type: ignore[no-untyped-def]
        if self._opaque_irrelevant(action, arg_string):
            return arg_string
        return super()._get_value(action, arg_string)

    def _check_value(self, action, value):  # type: ignore[no-untyped-def]
        if self._opaque_irrelevant(action, value):
            return None
        return super()._check_value(action, value)


def _frontend_parser(
    env: Mapping[str, str],
) -> tuple[argparse.ArgumentParser, dict[str, str]]:
    """The frontend's argument parser with every env default read from ``env``.

    ``build_frontend_parser`` registers the frontend's arguments without
    reading the planner's env or filesystem and without the frontend's
    notices. Returns the parser and the env name -> dest map of everything
    it registered.
    """

    try:
        return build_frontend_parser(env, _ArgsParser())
    except (ValueError, TypeError, argparse.ArgumentTypeError) as error:
        # The frontend itself fails to start on such env.
        raise FrontendParseError(f"P4: invalid frontend env: {error}") from None


def _worker_router_parser(
    env: Mapping[str, str],
) -> tuple[argparse.ArgumentParser, dict[str, str]]:
    """A worker's router-advertisement parser with env defaults from ``env``.

    Registers ``router_args.add_worker_router_arguments``, which every
    backend (vLLM, SGLang, TensorRT-LLM, mocker) uses, against ``env``. Only
    ``router_mode`` is relevant: it decides whether the worker advertises a
    router configuration at all.
    """

    arg_env = ArgEnv(environ=dict(env))
    parser = _ArgsParser(
        relevant=frozenset({_ROUTER_MODE_DEST}),
        describe=_worker_parse_error_message,
    )
    try:
        add_worker_router_arguments(parser, env=arg_env)
    except (ValueError, TypeError, argparse.ArgumentTypeError) as error:
        # The worker itself fails to start on such env.
        raise FrontendParseError(f"has invalid router env ({error})") from None
    return parser, arg_env.env_dests


# --------------------------------------------------------- premise checks


def _startup_errors(
    startup: FrontendStartup, retune: RetuneConfig, planner_mode: str
) -> list[str]:
    """P4 on a parsed startup: KV mode, defaults, the controller's checks."""

    if startup.router_mode != "kv_router":
        return [
            f"P4: the frontend routes with {startup.router_mode!r}; live retune "
            "needs --router-mode kv"
        ]
    try:
        searched = set(_router_domain_keys(retune))
    except ValueError as error:
        return [f"P4: cannot expand retune.router: {error}"]
    config = startup.router_config
    errors: list[str] = []
    load_model = config.get(_LOAD_MODEL, "none")
    if load_model != "none":
        errors.append(
            f"P4: the frontend's {_LOAD_MODEL} is {load_model!r}; live retune "
            "needs 'none'"
        )
    if config.get(_CONDITIONAL_DISAGG):
        errors.append(
            "P4: the frontend enables conditional disaggregation, which replay "
            "and live routers resolve differently"
        )
    for key in sorted(config):
        if (
            key in searched
            or key in TOLERATED_ROUTER_FIELDS
            or key in (_LOAD_MODEL, _CONDITIONAL_DISAGG)
        ):
            continue
        errors.append(
            f"P4: the frontend sets {key}={config[key]!r}; only the searched "
            f"router knobs ({', '.join(sorted(searched)) or 'none'}) may differ "
            "from their defaults"
        )
    if errors:
        return errors
    return _controller_errors(retune, config, worker_types(planner_mode))


def _worker_router_errors(
    dgd: Mapping[str, Any],
    components: Sequence[Mapping[str, Any]],
    warnings: list[str],
) -> list[str]:
    """P4 on the workers: none advertises its own router configuration.

    A worker that resolves a router mode (``--router-mode`` or
    ``DYN_ROUTER_MODE``, from its container or the graph env) puts a
    ``RouterConfig`` in its model card, and the frontend routes that worker
    set with it instead of its own flags; replay would model the frontend's.
    """

    errors: list[str] = []
    env_from: list[str] = []
    for component in components:
        if _type(component) not in WORKER_TYPES:
            continue
        name = str(component.get("name", ""))
        env = _component_env(dgd, component)
        argv, shell = _worker_argv(component, env)
        detail = _worker_router_cached(
            canonical_json(
                {
                    "argv": argv,
                    "shell": shell,
                    "values": dict(env.values),
                    "opaque": sorted(env.opaque),
                }
            )
        )
        if detail is not None:
            errors.append(
                f"P4: worker component {name} {detail}; a worker's router "
                "configuration replaces the frontend's for its workers, so set "
                f"--router-mode and {ENV_ROUTER_MODE} on the frontend only (a "
                "graph-level env entry reaches every worker)"
            )
        if any(_env_from_may_set_router_env(source) for source in env.env_from):
            env_from.append(name)
    if env_from:
        warnings.append(
            f"P4: worker components {', '.join(sorted(env_from))} use envFrom; "
            "the planner cannot read it and assumes it sets no router env (a "
            f"{ENV_ROUTER_MODE} there would make the worker advertise its own "
            "router configuration)"
        )
    return errors


@lru_cache(maxsize=64)
def _worker_router_cached(key: str) -> Optional[str]:
    """Why one worker advertises a router configuration, or ``None``."""

    inputs = json.loads(key)
    try:
        return _worker_router_mode(
            inputs["argv"],
            bool(inputs["shell"]),
            inputs["values"],
            frozenset(inputs["opaque"]),
        )
    except FrontendParseError as error:
        return str(error)
    except Exception as error:  # noqa: BLE001 - fail closed, never crash the planner
        logger.exception("live retune could not parse a worker's router flags")
        return (
            "could not be checked for a router configuration "
            f"({type(error).__name__}: {error})"
        )


def _worker_router_mode(
    argv: Sequence[str],
    shell: bool,
    values: Mapping[str, str],
    opaque: frozenset[str],
) -> Optional[str]:
    parser, registered = _worker_router_parser(values)
    mode_env = sorted(
        name for name, dest in registered.items() if dest == _ROUTER_MODE_DEST
    )
    hidden = [name for name in mode_env if name in opaque]
    if hidden:
        return (
            f"sets {', '.join(hidden)} through valueFrom (or a reference to such "
            "a value), so the planner cannot tell whether it advertises a router "
            "configuration"
        )
    # A shell script can also export the env inline.
    inline = sorted(
        {token.split("=", 1)[0] for token in argv if shell and "=" in token}
        & set(mode_env)
    )
    if inline:
        return f"sets {', '.join(inline)} in its shell command"
    namespace, _ = parser.parse_known_args(list(argv))
    mode = getattr(namespace, _ROUTER_MODE_DEST, None)
    if mode is None:
        return None
    return (
        f"advertises its own router configuration (router mode {mode!r} from "
        f"--router-mode or {ENV_ROUTER_MODE})"
    )


def _worker_argv(component: Mapping[str, Any], env: _Env) -> tuple[list[str], bool]:
    """A worker's argv after ``$(VAR)`` expansion, and whether a shell runs it.

    Unreadable references stay marked. A shell-wrapped command is split
    into the script's words, so the flags of the command it runs are seen.
    """

    main = _main_container(component)
    parts: list[str] = []
    for part in [*(main.get("command") or []), *(main.get("args") or [])]:
        expanded = _expand(str(part), env.values, env.opaque, mark_opaque=True)
        assert expanded is not None
        parts.append(expanded)
    shell = bool(parts) and (
        posixpath.basename(parts[0]) in _SHELLS or parts[1:2] == ["-c"]
    )
    if not shell:
        return parts, False
    words: list[str] = []
    for part in parts:
        try:
            words.extend(shlex.split(part))
        except ValueError:
            words.extend(part.split())
    return words, True


def _router_domain_keys(retune: RetuneConfig) -> list[str]:
    if retune.router is None:
        return []
    return list(expand_router_domain(retune))


def _controller_errors(
    retune: RetuneConfig, router_config: Mapping[str, Any], wts: tuple[str, ...]
) -> list[str]:
    """The startup-dependent checks of ``RetuneController._router_domain``.

    A startup failing them would make the controller constructor raise, so
    P4 rejects it before a controller is built.
    """

    if retune.router is None:
        return []
    startup_json = canonical_json(dict(router_config))
    try:
        startup_maps = _policy_maps(startup_json, wts)
        load_model = json.loads(KvRouterConfig.from_json(startup_json).to_json())[
            _LOAD_MODEL
        ]
    except ValueError as error:
        return [
            f"P4: the frontend startup is not a valid router configuration: {error}"
        ]
    # A worker type that starts with no overlap credit can never get it.
    zero_credit = sorted(
        wt
        for wt, params in startup_maps.items()
        if wt != "decode" and params[OVERLAP_SCORE_CREDIT] == 0
    )
    try:
        domain = expand_router_domain(
            retune,
            startup_load_model=load_model,
            zero_startup_credit=bool(zero_credit),
        )
    except ValueError as error:
        return [f"P4: retune.router does not fit the frontend startup: {error}"]
    for point in grid_points(domain, {}):
        try:
            maps = _policy_maps(canonical_json({**router_config, **point.router}), wts)
        except ValueError as error:
            return [
                f"P4: retune.router candidate {point.router} is not a valid "
                f"router configuration for this frontend: {error}"
            ]
        for wt in zero_credit:
            if maps[wt][OVERLAP_SCORE_CREDIT] != 0:
                return [
                    f"P4: retune.router candidate {point.router} gives the {wt} "
                    "router overlap_score_credit > 0, but the frontend starts it "
                    "at 0 (a live router rejects that update)"
                ]
    return []


def _startup_config_mismatch(
    declared: Mapping[str, Any], startup: FrontendStartup
) -> Optional[str]:
    """``retune.dgdr.startup_router_config`` must mean the parsed startup."""

    try:
        declared_full = json.loads(
            KvRouterConfig.from_json(json.dumps(dict(declared))).to_json()
        )
    except ValueError as error:
        return (
            "P4: retune.dgdr.startup_router_config is not a KvRouterConfig "
            f"JSON: {error}"
        )
    parsed_full = json.loads(
        KvRouterConfig.from_json(json.dumps(dict(startup.router_config))).to_json()
    )
    keys = sorted(
        key
        for key in set(declared_full) | set(parsed_full)
        if declared_full.get(key) != parsed_full.get(key)
    )
    if not keys:
        return None
    diffs = "; ".join(
        f"{key}: declared {declared_full.get(key)!r}, frontend {parsed_full.get(key)!r}"
        for key in keys
    )
    return f"P4: retune.dgdr.startup_router_config differs from the frontend ({diffs})"


def _planner_replica_errors(
    components: Sequence[Mapping[str, Any]], warnings: list[str]
) -> list[str]:
    planners = [c for c in components if _type(c) == PLANNER_TYPE]
    if not planners:
        warnings.append(
            "P9: the DGD has no planner component; its replicas cannot be checked"
        )
    return [
        f"P9: planner component {c.get('name')} has {_replicas(c)} replicas; "
        "live retune needs 1"
        for c in planners
        if _replicas(c) != 1
    ]


def _trace_recipe_errors(
    analysis: _Analysis,
    *,
    dgd_name: str,
    claim_name: str,
    segments_dir: str,
    warnings: list[str],
) -> list[str]:
    """P10: the frontend writes request_end segments onto the claim."""

    assert analysis.spec is not None and analysis.env is not None
    entries = analysis.env.entries
    errors: list[str] = []

    def literal(name: str) -> Optional[str]:
        entry = entries.get(name)
        if entry is None:
            return None
        if entry.get("valueFrom") is not None:
            errors.append(f"P10: frontend env {name} must be a literal value")
            return None
        return str(entry.get("value") or "")

    trace = literal(ENV_TRACE)
    if trace is None or trace.strip().lower() not in _TRUTHY:
        errors.append(f"P10: the frontend must set {ENV_TRACE}=1")

    sinks = _csv(literal(ENV_TRACE_SINKS))
    if "file" not in sinks or {"jsonl", "jsonl_gz"} & set(sinks):
        errors.append(f"P10: the frontend must set {ENV_TRACE_SINKS}=file")
    elif set(sinks) != {"file"}:
        extra = ", ".join(sorted(set(sinks) - {"file"}))
        warnings.append(f"P10: {ENV_TRACE_SINKS} also enables {extra}")

    file_format = literal(ENV_TRACE_FORMAT)
    if file_format is not None and file_format.strip().lower() not in _GZ_FORMATS:
        errors.append(f"P10: {ENV_TRACE_FORMAT} must be jsonl_gz or unset")

    if set(_csv(literal(ENV_TRACE_RECORDS))) != {"request_end"}:
        errors.append(f"P10: the frontend must set {ENV_TRACE_RECORDS}=request_end")

    pod_name = entries.get(ENV_POD_NAME) or {}
    field_ref = (pod_name.get("valueFrom") or {}).get("fieldRef") or {}
    if field_ref.get("fieldPath") != "metadata.name":
        errors.append(
            f"P10: the frontend must set {ENV_POD_NAME} from fieldRef metadata.name"
        )
    parent = entries.get(ENV_PARENT_DGD_NAME) or {}
    if parent.get("valueFrom") is not None or parent.get("value") != dgd_name:
        errors.append(
            f"P10: the frontend overrides {ENV_PARENT_DGD_NAME}; it must stay the "
            f"DGD name {dgd_name!r}"
        )

    mounts, mount_errors = _claim_mounts(analysis.spec, claim_name)
    errors.extend(mount_errors)
    path = literal(ENV_TRACE_PATH)
    if path is None:
        errors.append(f"P10: the frontend must set {ENV_TRACE_PATH}")
    elif mounts:
        segments = segments_dir.strip("/")
        prefixes = [
            _join_dir(mount, segments, dgd_dir, f"$({ENV_POD_NAME})")
            for mount in mounts
            for dgd_dir in (f"$({ENV_PARENT_DGD_NAME})", dgd_name)
        ]
        # Directly in the pod's directory: the reconciler prunes only the
        # segments at its top level.
        if not any(
            path.startswith(prefix)
            and len(path) > len(prefix)
            and "/" not in path[len(prefix) :]
            for prefix in prefixes
        ):
            errors.append(
                f"P10: {ENV_TRACE_PATH}={path!r} must be a file prefix directly "
                f"under {prefixes[0]!r} (no further '/')"
            )
    return errors


def _claim_mounts(
    spec: Mapping[str, Any], claim_name: str
) -> tuple[list[str], list[str]]:
    """Writable main-container mount paths of the claim's root, and errors."""

    pod_spec = _pod_spec(spec)
    volumes = {
        str(volume.get("name"))
        for volume in pod_spec.get("volumes") or []
        if isinstance(volume, Mapping)
        and (volume.get("persistentVolumeClaim") or {}).get("claimName") == claim_name
    }
    if not volumes:
        return [], [f"P10: the frontend pod has no volume for claim {claim_name!r}"]
    candidates = [
        mount
        for mount in _main_container(spec).get("volumeMounts") or []
        if isinstance(mount, Mapping) and mount.get("name") in volumes
    ]
    if not candidates:
        return [], [
            f"P10: the frontend main container does not mount claim {claim_name!r}"
        ]
    mounts = [
        str(mount["mountPath"])
        for mount in candidates
        if mount.get("mountPath")
        and not mount.get("readOnly")
        and not mount.get("subPath")
        and not mount.get("subPathExpr")
    ]
    if not mounts:
        return [], [
            f"P10: the frontend mounts claim {claim_name!r} read-only or through a "
            "subPath; mount the claim root writable"
        ]
    return mounts, []


def _join_dir(mount: str, *parts: str) -> str:
    return "/".join([mount.rstrip("/"), *(part for part in parts if part)]) + "/"


@lru_cache(maxsize=8)
def _model_warning(model: str) -> Optional[str]:
    """P6 for ``model``; cached, so a broken install logs once per model."""

    try:
        # Optional dependency of a warn-only premise: without it, P6 is not
        # checked.
        from aisimulate_core.sdk.common import DefaultHFModels
    except ImportError:
        return None
    except Exception as error:  # noqa: BLE001 - evaluate_premises never raises
        logger.exception("live retune could not load AISimulate's default HF models")
        return (
            f"P6: cannot check retune.dgdr.model {model!r} against AISimulate's "
            f"default HF models ({type(error).__name__}: {error}); each DGDR Run "
            "must be able to download it"
        )
    if model in DefaultHFModels:
        return None
    return (
        f"P6: retune.dgdr.model {model!r} is not an AISimulate default HF model; "
        "each DGDR Run must be able to download it"
    )


# ------------------------------------------------------------ DGD helpers


def _components(dgd: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    components = (dgd.get("spec") or {}).get("components")
    if isinstance(components, list):
        return [c for c in components if isinstance(c, Mapping)]
    return []


def _graph_env(dgd: Mapping[str, Any]) -> list[Any]:
    return list((dgd.get("spec") or {}).get("env") or [])


def _pod_spec(spec: Mapping[str, Any]) -> Mapping[str, Any]:
    template = spec.get("podTemplate")
    pod_spec = template.get("spec") if isinstance(template, Mapping) else None
    return pod_spec if isinstance(pod_spec, Mapping) else {}


def _main_container(spec: Mapping[str, Any]) -> Mapping[str, Any]:
    for container in _pod_spec(spec).get("containers") or []:
        if isinstance(container, Mapping) and container.get("name") == MAIN_CONTAINER:
            return container
    return {}


def _sidecar_components(components: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted(str(c.get("name")) for c in components if c.get("frontendSidecar"))


def _type(component: Mapping[str, Any]) -> str:
    return str(component.get("type") or "")


def _replicas(component: Mapping[str, Any]) -> int:
    """Spec replicas; an omitted value is the workload default of 1."""

    replicas = component.get("replicas")
    try:
        return 1 if replicas is None else int(replicas)
    except (TypeError, ValueError):
        return 1


def _scaling_adapter(component: Mapping[str, Any]) -> bool:
    return component.get("scalingAdapter") is not None


def _dgd_name(dgd: Mapping[str, Any]) -> str:
    return str((dgd.get("metadata") or {}).get("name", ""))


def _projected_status(dgd: Mapping[str, Any]) -> Mapping[str, Any]:
    """``status.components`` while the status is current, else empty."""

    status = dgd.get("status") or {}
    generation = (dgd.get("metadata") or {}).get("generation")
    current = generation is not None and generation == status.get("observedGeneration")
    return (status.get("components") or {}) if current else {}


def _model_flags(
    component: Mapping[str, Any], env: _Env
) -> list[tuple[str, Optional[str]]]:
    """Each model flag in the main container's command and args, in order.

    Values are ``$(VAR)``-expanded against the component's env; a value
    that still references env (or looks like a flag) is ``None``.
    """

    main = _main_container(component)
    tokens: list[str] = []
    for part in [*(main.get("command") or []), *(main.get("args") or [])]:
        text = _expand(str(part), env.values, env.opaque, mark_opaque=True)
        assert text is not None
        try:
            tokens.extend(shlex.split(text))
        except ValueError:
            tokens.append(text)
    found: list[tuple[str, Optional[str]]] = []
    for index, token in enumerate(tokens):
        for flag in _MODEL_FLAGS:
            if token == flag and index + 1 < len(tokens):
                value = tokens[index + 1]
            elif token.startswith(flag + "="):
                value = token[len(flag) + 1 :]
            else:
                continue
            resolved = (
                value
                and "$" not in value
                and _OPAQUE not in value
                and not value.startswith("-")
            )
            found.append((flag, value if resolved else None))
    return found


def _flag_model(flags: Sequence[tuple[str, Optional[str]]]) -> Optional[str]:
    """The operator's served model: the first flag of ``_MODEL_FLAGS`` given."""

    for flag in _MODEL_FLAGS:
        for name, value in flags:
            if name == flag:
                return value
    return None


def _worker_model_names(
    dgd: Mapping[str, Any], components: Sequence[Mapping[str, Any]]
) -> set[str]:
    """Every model name or path the workers give, and their served names."""

    projected = _projected_status(dgd)
    names: set[str] = set()
    for component in components:
        if _type(component) not in WORKER_TYPES:
            continue
        served = (projected.get(str(component.get("name", ""))) or {}).get(
            "servedModelName"
        )
        if served:
            names.add(str(served))
        for _, value in _model_flags(component, _component_env(dgd, component)):
            if value:
                names.update(part for part in re.split(r"[\s,]+", value) if part)
    return names


def _csv(value: Optional[str]) -> list[str]:
    if not value:
        return []
    return [part.strip().lower() for part in value.split(",") if part.strip()]


def _unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))
