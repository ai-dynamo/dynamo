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
  ``--router-policy-config`` file, router env the planner cannot read and
  an argument whose unreadable ``$(VAR)`` could set a router flag make the
  frontend unparseable.
- Env set through ``valueFrom`` is unreadable. A ConfigMap, Secret or
  annotation value could also be or create an option ("dangerous"), as
  could a ``$(VAR)`` the env leaves undefined that an ``envFrom`` source
  may define (its ``prefix`` plus a ConfigMap or Secret key). Any other
  ``fieldRef`` (the operator's ``POD_*`` among them), a
  ``resourceFieldRef`` and the operator env of ``_operator_inert_env``
  (``DYN_NAMESPACE`` among them) never start with ``-`` or hold
  whitespace, so they stay unreadable values but never stand for an option
  ("inert"). ``envFrom`` sources are otherwise assumed to set no router
  env; ``evaluate_premises`` warns about them, and P14 compares the
  routers' policy parameters with the parsed startup.
- ``evaluate_premises`` checks P4 and P7-P10 and warns on P6 and P7; P14
  is the router actuator's. P4 also parses every worker component's router
  flags and env the same way (``_worker_router_errors``), each role's pod
  template for a component that has only those, and skips a component
  with a native ``runtime`` sidecar, whose main runs the engine: a worker
  that resolves a router mode advertises its own router configuration,
  which replaces the frontend's for its worker set, and one where
  dangerous env could stand for an option fails too. ``p4_ok`` alone
  decides whether a retune controller can be built.

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
from dynamo.common.environment import compose_environment, ordered_environment_enabled
from dynamo.common.operator_environment import owned_environment
from dynamo.frontend.frontend_args import build_frontend_parser
from dynamo.llm import KvRouterConfig
from dynamo.planner.retune.controller import startup_router_domain
from dynamo.planner.retune.domains import expand_router_domain
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
# The operator's worker invocation (component_common.go getCommonContainer),
# kept by graph.go's merge under an empty authored command.
DEFAULT_WORKER_COMMAND = ("/bin/sh", "-c")
# consts.go RuntimeContainerName: an init container of this name makes the
# component a native sidecar, whose main runs the engine.
RUNTIME_CONTAINER = "runtime"
# runtime_status.go ResolveComponentRuntimeStatus, in its priority order.
_MODEL_FLAGS = ("--served-model-name", "--model-name", "--model", "--model-path")
# Worker options known to take at least one value, so an unreadable value
# right after one is a value: were it itself an option, the listed one would
# lack its value and the worker would fail at startup. The worker's router
# parser knows only the router flags, not the arity of engine or runtime
# flags (--enforce-eager takes none, --connector a variable number), so only
# these exempt a value. Checked arities: vLLM --model (one) and
# --served-model-name (nargs "+", whose first value this is);
# SGLang --model-path and --served-model-name (one); TensorRT-LLM --model,
# --model-path and --served-model-name (one); the mocker's --model-path and
# --model-name (one); --speculative-config (one JSON value). Before adding
# an option, check its nargs and action in every backend that accepts it.
_WORKER_VALUE_OPTIONS = frozenset({"--speculative-config", *_MODEL_FLAGS})
# Left-over frontend options known to take exactly one value, in the sglang
# chat processor and in vLLM's. Other left-over options go to the chat
# processor too, where some take none (vLLM's --enable-auto-tool-choice).
_FRONTEND_VALUE_OPTIONS = frozenset(
    {"--chat-template", "--reasoning-parser", "--tool-call-parser"}
)

# Env the operator injects into every component (component_common.go
# getCommonContainer); user env of the same name overrides it.
ENV_PARENT_DGD_NAME = "DYN_PARENT_DGD_K8S_NAME"
ENV_PARENT_DGD_NAMESPACE = "DYN_PARENT_DGD_K8S_NAMESPACE"
_OPERATOR_FIELD_REF_ENV = {
    "POD_NAME": "metadata.name",
    "POD_NAMESPACE": "metadata.namespace",
    "POD_UID": "metadata.uid",
}
# Operator env whose value the planner does not compute but which never
# starts with "-" or holds whitespace: getCommonContainer's Dynamo namespace
# and component type; on a worker, component_worker.go GetBaseContainer's
# system server, health check, NIXL telemetry and forward-pass metric
# settings; on a multinode component, graph.go addMultinodeTopologyEnvVars'
# leader hostname and rank. Container env wins over envFrom, so no envFrom
# source defines them. Names the operator sets only in some configurations
# (NATS_SERVER, ETCD_ENDPOINTS, DYN_DISCOVERY_BACKEND, CONTAINER_NAME, the
# TLS paths) are not modeled.
_OPERATOR_INERT_ENV = ("DYN_COMPONENT", "DYN_NAMESPACE")
_OPERATOR_WORKER_INERT_ENV = (
    "DYN_FORWARDPASS_METRIC_PORT",
    "DYN_HEALTH_CHECK_ENABLED",
    "DYN_SYSTEM_ENABLED",
    "DYN_SYSTEM_PORT",
    "DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS",
    "NIXL_TELEMETRY_ENABLE",
    "NIXL_TELEMETRY_EXPORTER",
    "NIXL_TELEMETRY_PROMETHEUS_PORT",
)
_OPERATOR_MULTINODE_INERT_ENV = ("DYNAMO_LEADER_ADDRESS", "DYNAMO_RANK")

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
_POD_NAME_VALUE = "\x00fieldRef:metadata.name\x00"

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
_SHELLS = frozenset({"sh", "bash", "dash", "zsh", "ash", "ksh"})
# Shell option letters that take the next argument (-o pipefail, -O extglob),
# and bash's long options that do.
_SHELL_OPERAND_LETTERS = frozenset("oO")
_SHELL_LONG_OPTIONS_WITH_VALUE = frozenset({"--init-file", "--rcfile"})
# Python options that may precede ``-m``; the second set takes a value.
_PYTHON_FLAGS = frozenset({"-u", "-B", "-E", "-I", "-O", "-OO", "-P", "-q", "-s", "-S"})
_PYTHON_FLAGS_WITH_VALUE = frozenset({"-X", "-W"})
# Python's -m, alone or closing a cluster of flags (-um), with its module
# attached or in the next word.
_PYTHON_MODULE_OPTION = re.compile(r"-[A-Za-z]*?m(.*)")
# Marks an argument part that references env the planner cannot read, as
# "<marker><reference>\x00": _OPAQUE where the value could be or create an
# option, _INERT where it cannot.
_UNREADABLE = "\x00unreadable"
_OPAQUE = _UNREADABLE + ":"
_INERT = _UNREADABLE + "-inert:"
# A marker, also as argparse quotes it with repr.
_MARKER = re.compile(r"(?:\x00|\\x00)unreadable(?:-inert)?:(.*?)(?:\x00|\\x00|$)")
# The keys a ConfigMap or Secret may hold, so what an envFrom source defines.
_ENV_FROM_KEY = re.compile(r"[-._a-zA-Z0-9]+")

# Shell syntax the worker script scan reads.
_SHELL_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SHELL_ASSIGNMENT = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\+?=")
_SHELL_BLANKS = frozenset(" \t")
_SHELL_OPERATORS = frozenset("\n;&|()<>")
# Reserved words that may precede a command's name.
_SHELL_RESERVED = frozenset(
    {"!", "{", "do", "elif", "else", "if", "then", "time", "until", "while"}
)
_SHELL_DECLARATIONS = frozenset({"declare", "export", "local", "readonly", "typeset"})
# Commands that parse a quoted word again or turn it into env, so that one
# word can still reach the worker's options or env.
_SHELL_REPARSING = frozenset({".", "env", "eval", "export", "source"})
# Commands that run their first operand as the command.
_SHELL_LAUNCHERS = frozenset({"builtin", "command", "exec", "nohup"})


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
    container's command and args (the operator's order; of each role's pod
    template, for a component with role templates only), after Kubernetes
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
            _component_model_flags(dgd, component)
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
    ``opaque`` the names it cannot (``valueFrom``, or a reference to
    unreadable env), and ``inert`` those of them whose value can never be or
    create an option.
    ``replay`` retains the composed order for semantic origins so P10 can
    verify runtime source identity at the point the trace path is expanded.
    """

    entries: Mapping[str, Mapping[str, Any]]
    values: Mapping[str, str]
    opaque: frozenset[str]
    inert: frozenset[str]
    env_from: tuple[Mapping[str, Any], ...]
    replay: Optional[tuple[Mapping[str, Any], ...]]

    @property
    def env_from_prefixes(self) -> tuple[str, ...]:
        return _env_from_prefixes(self.env_from)

    def expand(self, value: str) -> str:
        return _expand(
            value,
            self.values,
            self.opaque,
            inert=self.inert,
            env_from_prefixes=self.env_from_prefixes,
        )


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
    try:
        env = _component_env(dgd, spec)
    except FrontendParseError as error:
        return _Analysis(component=name, spec=spec, error=str(error))
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
                "inert": sorted(env.inert),
                "env_from_prefixes": list(env.env_from_prefixes),
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
            inert=frozenset(inputs["inert"]),
            env_from_prefixes=tuple(inputs["env_from_prefixes"]),
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
    *,
    inert: frozenset[str] = frozenset(),
    env_from_prefixes: tuple[str, ...] = (),
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
            "(or references such a value, or a $(VAR) an envFrom source may "
            "define); the planner cannot read it, so set it with a literal value"
        )

    # An unreadable $(VAR) becomes a marker the parser carries as a value: a
    # non-router flag keeps it unconverted, a router flag fails the parse,
    # and a dangerous one left over where a flag could stand fails below.
    expanded = [
        _expand(arg, values, opaque, inert=inert, env_from_prefixes=env_from_prefixes)
        for arg in args
    ]
    namespace, unknown = parser.parse_known_args(expanded)
    hidden_flag = _opaque_flag_position(unknown, value_options=_FRONTEND_VALUE_OPTIONS)
    if hidden_flag is not None:
        original = dict(zip(expanded, args)).get(hidden_flag, hidden_flag)
        raise FrontendParseError(
            f"P4: frontend arg {original!r} references env the planner cannot "
            "read and is neither attached to an option nor the value of a "
            "frontend flag or of one known to take a value "
            f"({', '.join(sorted(_FRONTEND_VALUE_OPTIONS))}), so it could set a "
            "router flag"
        )
    unresolved = sorted(
        dest
        for dest, value in vars(namespace).items()
        if dest in _RELEVANT_DESTS and _UNREADABLE in str(value)
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

    The operator injects ``DYN_PARENT_DGD_K8S_*``, ``POD_*`` and the inert
    names of ``_operator_inert_env``, except into a native-sidecar main,
    which gets no base container; MergeEnvs overlays graph ``env`` and the
    main container's ``env``. Origins before 1.8 sort by name; newer origins
    compose sources without changing authored list order. Kubernetes expands
    ``$(VAR)`` against earlier names and every ``envFrom`` key.
    """

    metadata = dgd.get("metadata") or {}
    entries: dict[str, Mapping[str, Any]] = {}
    operator_inert: set[str] = set()
    if not _native_sidecar(spec):
        entries[ENV_PARENT_DGD_NAME] = {
            "name": ENV_PARENT_DGD_NAME,
            "value": str(metadata.get("name", "")),
        }
        entries[ENV_PARENT_DGD_NAMESPACE] = {
            "name": ENV_PARENT_DGD_NAMESPACE,
            "value": str(metadata.get("namespace", "")),
        }
        for name, path in _OPERATOR_FIELD_REF_ENV.items():
            entries[name] = {
                "name": name,
                "valueFrom": {"fieldRef": {"fieldPath": path}},
            }
        operator_inert.update(_operator_inert_env(spec))
    main = _main_container(spec)
    layers = [list(entries.values())]
    for source in [_graph_env(dgd), main.get("env") or []]:
        layer = [
            entry
            for entry in source
            if isinstance(entry, Mapping) and entry.get("name")
        ]
        layers.append(layer)
        for entry in layer:
            entries[str(entry["name"])] = entry
            operator_inert.discard(str(entry["name"]))
    env_from = tuple(
        source for source in main.get("envFrom") or [] if isinstance(source, Mapping)
    )
    prefixes = _env_from_prefixes(env_from)

    values: dict[str, str] = {}
    opaque: set[str] = set()
    inert: set[str] = set()
    ordered = ordered_environment_enabled(metadata.get("annotations") or {})
    if ordered:
        try:
            bindings, omitted = owned_environment(dgd, spec)
        except ValueError as error:
            raise FrontendParseError(
                f"P4: invalid component backend: {error}"
            ) from None
        layers = [
            [entry for entry in layer if entry["name"] not in omitted]
            for layer in layers
        ]
        for name in omitted:
            entries.pop(name, None)
            operator_inert.discard(name)
        owned: list[Mapping[str, Any]] = []
        for name, value in bindings.items():
            entry = {"name": name} if value is None else {"name": name, "value": value}
            owned.append(entry)
            entries[name] = entry
            if value is None:
                operator_inert.add(name)
            else:
                operator_inert.discard(name)
        layers.append(owned)
        try:
            replay = compose_environment(*layers)
        except ValueError as error:
            raise FrontendParseError(
                f"P4: invalid component environment: {error}"
            ) from None
        opaque.update(operator_inert)
        inert.update(operator_inert)
    else:
        replay = [
            entries.get(name, {"name": name})
            for name in sorted({*entries, *operator_inert})
        ]
    for entry in replay:
        name = str(entry["name"])
        if name in operator_inert:
            opaque.add(name)
            inert.add(name)
            continue
        source = entry.get("valueFrom")
        if source is not None:
            if ordered:
                values.pop(name, None)
                inert.discard(name)
            opaque.add(name)
            if _inert_source(source):
                inert.add(name)
            continue
        expanded = _expand(
            str(entry.get("value") or ""),
            values,
            opaque,
            inert=inert,
            env_from_prefixes=prefixes,
        )

        # A repeated definition expands against the previous value before replacing it.
        if ordered:
            values.pop(name, None)
            opaque.discard(name)
            inert.discard(name)
        if _UNREADABLE not in expanded:
            values[name] = expanded
            continue
        opaque.add(name)
        # Literal text around inert values may still make an option.
        if _OPAQUE not in expanded and not (
            expanded.startswith("-") or any(char.isspace() for char in expanded)
        ):
            inert.add(name)
    return _Env(
        entries=entries,
        values=values,
        opaque=frozenset(opaque),
        inert=frozenset(inert),
        env_from=env_from,
        replay=tuple(replay) if ordered else None,
    )


def _operator_inert_env(spec: Mapping[str, Any]) -> list[str]:
    """The ``_OPERATOR_*INERT_ENV`` names the operator sets on a component.

    The worker names need an explicit worker type (component_common.go
    ComponentDefaultsFactory gives an untyped component the common base
    container only), the multinode ones a ``multinode`` component, whose
    ``nodeCount`` defaults to 2.
    """

    names = list(_OPERATOR_INERT_ENV)
    if _type(spec) and _type(spec) in WORKER_TYPES:
        names.extend(_OPERATOR_WORKER_INERT_ENV)
    if isinstance(spec.get("multinode"), Mapping):
        names.extend(_OPERATOR_MULTINODE_INERT_ENV)
    return names


def _inert_source(source: Any) -> bool:
    """Whether a ``valueFrom`` value can never be or create an option.

    A fieldRef other than an annotation (a name, namespace, uid, label, IP,
    node or service account name) and a resourceFieldRef (a quantity) never
    start with ``-`` or hold whitespace; a ConfigMap or Secret key and an
    annotation can hold anything.
    """

    if not isinstance(source, Mapping):
        return False
    if isinstance(source.get("resourceFieldRef"), Mapping):
        return True
    field_ref = source.get("fieldRef")
    if not isinstance(field_ref, Mapping):
        return False
    return not str(field_ref.get("fieldPath") or "").startswith("metadata.annotations")


def _env_from_prefixes(env_from: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(sorted({str(source.get("prefix") or "") for source in env_from}))


def _env_from_may_define(name: str, prefixes: Iterable[str]) -> bool:
    """Whether an ``envFrom`` source with one of ``prefixes`` could define ``name``."""

    return any(
        name.startswith(prefix) and _ENV_FROM_KEY.fullmatch(name[len(prefix) :])
        for prefix in prefixes
    )


def _expand(
    value: str,
    values: Mapping[str, str],
    opaque: Iterable[str],
    *,
    inert: Iterable[str] = (),
    env_from_prefixes: Iterable[str] = (),
) -> str:
    """Kubernetes ``$(VAR)`` expansion, marking what the planner cannot read.

    ``$$`` escapes a ``$``. A reference to an ``opaque`` name, or to an
    undefined one an ``envFrom`` source may define, is replaced by a marker:
    ``_INERT`` for an ``inert`` name, ``_OPAQUE`` otherwise. Any other
    undefined reference stays as written.
    """

    hidden, harmless = set(opaque), set(inert)
    prefixes = tuple(env_from_prefixes)
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
                if name in harmless:
                    out.append(f"{_INERT}$({name})\x00")
                elif name in hidden or (
                    name not in values and _env_from_may_define(name, prefixes)
                ):
                    out.append(f"{_OPAQUE}$({name})\x00")
                else:
                    out.append(values.get(name, value[index : close + 1]))
                index = close + 1
                continue
        out.append(value[index])
        index += 1
    return "".join(out)


def _opaque_flag_position(
    unknown: Sequence[str], *, value_options: frozenset[str]
) -> Optional[str]:
    """A left-over argument whose dangerous env could stand for a flag.

    ``unknown`` is what the frontend parser, or a worker's router parser,
    leaves over in exec form. The parser consumes the value of every flag it
    knows, so a dangerous argument it leaves over sits where a flag could,
    unless it is the value of a left-over option, and a left-over option
    whose name is unreadable always could. Inert env never stands for a
    flag, and ``--option=$(VAR)`` is the option's value whatever it holds.

    A reference expands to exactly one argument, so right after one of
    ``value_options``, the left-over options known to take a value, it is
    that value: one that starts with ``-`` makes that option's own parser
    fail at startup. After any other left-over option it could be parsed as
    a flag: the frontend hands its left-over options to its chat processor
    and a worker to its engine, and both have flags that take no value
    (vLLM's ``--enable-auto-tool-choice``, ``--enforce-eager``) or
    optional values (``--connector``).
    """

    previous: Optional[str] = None
    for token in unknown:
        if token.startswith("-"):
            if _UNREADABLE in token.split("=", 1)[0]:
                return token
        elif token.startswith(_OPAQUE) and previous not in value_options:
            return token
        previous = token
    return None


def _env_from_note(
    reference: Optional[str], opaque: frozenset[str], prefixes: Sequence[str]
) -> str:
    """Says why a reference to env (``$(NAME)``, ``$NAME``) is unreadable.

    Empty unless the name is not opaque env and an envFrom source may
    define it. A script variable's reference, which names its source
    (``$NAME(=<source>)``), never gets the note.
    """

    name = (reference or "").strip("$()")
    if name in opaque or not _env_from_may_define(name, prefixes):
        return ""
    return " (the container's env leaves it to an envFrom source)"


def _opaque_reference(text: str) -> Optional[str]:
    """The reference (``$(NAME)``, ``$NAME``) of the first marker in ``text``.

    argparse quotes a rejected value with ``repr``, so the marker may appear
    escaped.
    """

    match = _MARKER.search(text)
    return None if match is None else match.group(1)


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

    A value carrying an unreadable marker skips type conversion and the
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
            and _UNREADABLE in value
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
    searched = set(expand_router_domain(retune))
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
            f"router knobs ({', '.join(sorted(searched))}) may differ from "
            "their defaults"
        )
    if errors:
        return errors
    # The controller constructor's own checks, so that P4 passes only for a
    # startup a controller can be built on.
    try:
        startup_router_domain(retune, config, worker_types(planner_mode))
    except ValueError as error:
        return [f"P4: retune.router does not fit the frontend startup: {error}"]
    return []


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
    A worker whose mode the planner cannot rule out fails too: one where
    dangerous env could stand for an option. In exec form that is a
    reference neither attached to its option (``--option=$(VAR)``) nor
    after one of ``_WORKER_VALUE_OPTIONS``; in shell form any dangerous
    ``$(VAR)``, which Kubernetes substitutes before the shell parses, and
    the script references ``_shell_expand`` marks.

    A component with role pod templates and no component ``podTemplate``
    is checked role by role, each role's template as its pods run it. A
    component with a native ``runtime`` sidecar is skipped: its main runs
    the engine with its image entrypoint, and the sidecar registers no
    router configuration.
    """

    errors: list[str] = []
    env_from: list[str] = []
    for component in components:
        if _type(component) not in WORKER_TYPES or _native_sidecar(component):
            continue
        name = str(component.get("name", ""))
        try:
            roles = [
                (role, spec, _component_env(dgd, spec))
                for role, spec in _role_specs(component)
            ]
        except FrontendParseError as error:
            errors.append(f"P4: worker component {name}: {error}")
            continue
        for role, spec, env in roles:
            argv, shell = _worker_argv(spec, env)
            detail = _worker_router_cached(
                canonical_json(
                    {
                        "argv": argv,
                        "shell": shell,
                        "values": dict(env.values),
                        "opaque": sorted(env.opaque),
                        "env_from_prefixes": list(env.env_from_prefixes),
                    }
                )
            )
            if detail is not None:
                label = name if role is None else f"{name} (role {role})"
                errors.append(
                    f"P4: worker component {label} {detail}; a worker's router "
                    "configuration replaces the frontend's for its workers, so set "
                    f"--router-mode and {ENV_ROUTER_MODE} on the frontend only (a "
                    "graph-level env entry reaches every worker)"
                )
                break
        if any(
            _env_from_may_set_router_env(source)
            for _, _, env in roles
            for source in env.env_from
        ):
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
            tuple(inputs["env_from_prefixes"]),
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
    env_from_prefixes: Sequence[str],
) -> Optional[str]:
    parser, registered = _worker_router_parser(values)
    mode_env = sorted(
        name for name, dest in registered.items() if dest == _ROUTER_MODE_DEST
    )
    hidden = [name for name in mode_env if name in opaque]
    if hidden:
        return (
            f"sets {', '.join(hidden)} through valueFrom (or a reference to such "
            "a value, or a $(VAR) an envFrom source may define), so the planner "
            "cannot tell whether it advertises a router configuration"
        )
    # A shell script can also export the env inline.
    inline = sorted(
        {token.split("=", 1)[0] for token in argv if shell and "=" in token}
        & set(mode_env)
    )
    if inline:
        return f"sets {', '.join(inline)} in its shell command"
    namespace, remainder = parser.parse_known_args(list(argv))
    mode = getattr(namespace, _ROUTER_MODE_DEST, None)
    if mode is not None:
        return (
            f"advertises its own router configuration (router mode {mode!r} from "
            f"--router-mode or {ENV_ROUTER_MODE})"
        )
    if shell:
        # Kubernetes substitutes $(VAR) into the script before the shell
        # parses it, so a dangerous one can become several words anywhere;
        # _shell_expand marks only the shell's own references that can.
        opaque_word = next((word for word in argv if _OPAQUE in word), None)
        if opaque_word is None:
            return None
        reference = _opaque_reference(opaque_word) or ""
        note = _env_from_note(reference, opaque, env_from_prefixes)
        if reference.startswith("$("):
            return (
                f"has a shell script that references {reference}, which "
                "Kubernetes substitutes into the script before the shell parses "
                "it, so its value could become an option wherever it stands, and "
                f"the planner cannot read it{note}, so it could select a router "
                "mode; give the value literally or use the shell's own reference "
                f'attached to its option (--option="${reference[2:-1]}")'
            )
        return (
            f"has a shell script that references {_describe_reference(reference)} "
            f"where it could become an option, and the planner cannot read it{note}, "
            "so it could select a router mode; give the value literally, quote the "
            'reference ("$VAR") outside the worker\'s arguments, or attach it to '
            'its option (--option="$VAR")'
        )
    hidden_flag = _opaque_flag_position(remainder, value_options=_WORKER_VALUE_OPTIONS)
    if hidden_flag is None:
        return None
    reference = _opaque_reference(hidden_flag) or ""
    note = _env_from_note(reference, opaque, env_from_prefixes)
    if hidden_flag.startswith("-"):
        return (
            f"has an option whose name references {reference}, which the planner "
            f"cannot read{note}, so it could select a router mode; spell the "
            "option's name literally"
        )
    return (
        f"has an argument that references {reference}, which the planner cannot "
        f"read{note} and which is neither attached to an option nor the value of "
        "one known to take a value "
        f"({', '.join(sorted(_WORKER_VALUE_OPTIONS))}), so it could select a "
        "router mode; give the value literally or attach it to its option "
        "(--option=$(VAR))"
    )


def _describe_reference(reference: str) -> str:
    """A shell reference as a message shows it, with its source when it has one.

    ``_ShellScan`` writes a script variable assigned from dangerous env as
    ``$NAME(=<source>)``, since a marker holds no whitespace.
    """

    match = re.fullmatch(r"(\$[A-Za-z_][A-Za-z0-9_]*)\(=(.*)\)", reference)
    if match is None:
        return reference
    return f"{match.group(1)} (assigned from {match.group(2)})"


def _worker_argv(component: Mapping[str, Any], env: _Env) -> tuple[list[str], bool]:
    """A worker's argv after ``$(VAR)`` expansion, and whether a shell runs it.

    An empty command is the operator's ``/bin/sh -c``. A shell runs the
    command when it starts with one, or when a shell's ``-c`` follows a
    launcher (``env bash -c``, ``tini -- sh -c``); then its script's own
    references are resolved (``_shell_expand``) and every part is split
    into words, so the flags of the commands it runs are seen. Unreadable
    ``$(VAR)`` references stay marked. A standalone ``--`` is dropped: a
    launcher consumes it, while the router parser would read every later
    flag as a positional.
    """

    main = _main_container(component)
    command = list(main.get("command") or []) or list(DEFAULT_WORKER_COMMAND)
    parts = [env.expand(str(part)) for part in [*command, *(main.get("args") or [])]]
    script = _shell_script_index(parts)
    shell = script is not None or (
        bool(parts)
        and (posixpath.basename(parts[0]) in _SHELLS or parts[1:2] == ["-c"])
    )
    words = parts
    if shell:
        words = []
        for index, part in enumerate(parts):
            if index == script:
                part = _shell_expand(part, env)
            try:
                words.extend(shlex.split(part))
            except ValueError:
                words.extend(part.split())
    return [word for word in words if word != "--"], shell


def _native_sidecar(component: Mapping[str, Any]) -> bool:
    """Whether the operator runs the component's Dynamo runtime as a native sidecar.

    v1beta1_helpers.go GetComponentDynamoSidecar: an init container named
    ``runtime`` in the pod template, or in the leader role's.
    """

    templates = [component]
    for role in component.get("roles") or []:
        if isinstance(role, Mapping) and role.get("name") == "leader":
            templates.append(role)
    return any(
        isinstance(container, Mapping) and container.get("name") == RUNTIME_CONTAINER
        for template in templates
        for container in _pod_spec(template).get("initContainers") or []
    )


def _role_specs(
    component: Mapping[str, Any],
) -> list[tuple[Optional[str], Mapping[str, Any]]]:
    """The component as each of its pods runs it, with the role's name.

    v1beta1_helpers.go EffectiveComponentForRole: a component ``podTemplate``,
    or no role ``podTemplate``, serves every pod (role ``None`` here);
    otherwise each role with a ``podTemplate`` runs that template. The
    roles stay in each copy, so ``_native_sidecar`` still reads the
    leader's.
    """

    roles = [
        role
        for role in component.get("roles") or []
        if isinstance(role, Mapping) and isinstance(role.get("podTemplate"), Mapping)
    ]
    if isinstance(component.get("podTemplate"), Mapping) or not roles:
        return [(None, component)]
    return [
        (str(role.get("name", "")), {**component, "podTemplate": role["podTemplate"]})
        for role in roles
    ]


def _shell_script_index(parts: Sequence[str]) -> Optional[int]:
    """The index of the script a shell's ``-c`` runs in ``parts``, if any.

    It is ``len(parts)`` when ``parts`` end before the script.
    """

    for index, part in enumerate(parts):
        if posixpath.basename(part) in _SHELLS:
            script = _shell_options_end(parts, index + 1)
            if script is not None:
                return script
    return None


def _shell_options_end(parts: Sequence[str], index: int) -> Optional[int]:
    """Where the options a shell reads from ``index`` on end, if they include ``-c``.

    The shell reads ``-`` and ``+`` clusters up to ``--``, ``-`` or a word
    that is none; each ``o`` or ``O`` in a cluster takes the next argument
    (``-euo pipefail``). With ``-c`` the script is the first word after them.
    """

    command = False
    while index < len(parts):
        option = parts[index]
        if option[:1] not in ("-", "+"):
            break
        index += 1
        if option in ("-", "--"):
            break
        if option.startswith("--"):
            if option in _SHELL_LONG_OPTIONS_WITH_VALUE:
                index += 1
            continue
        command |= option.startswith("-") and "c" in option
        index += sum(letter in _SHELL_OPERAND_LETTERS for letter in option[1:])
    return index if command else None


def _shell_expand(script: str, env: _Env) -> str:
    """A worker script with the shell's own parameter references resolved.

    Runs on the text the shell parses, after Kubernetes ``$(VAR)``
    substitution, and keeps it apart from each ``$NAME``, ``${NAME}`` and
    ``${NAME<op>word}`` outside single quotes, comments and heredocs:

    - literal env becomes its value as the shell passes it on (split into
      words when unquoted), so the router parser sees an option it holds;
    - dangerous env (a ConfigMap, Secret or annotation ``valueFrom``, or a
      name the container's env leaves to an ``envFrom`` source, which with
      an unprefixed source includes image env such as ``PATH``) becomes
      ``_OPAQUE`` where its value could be or create an option of the
      worker: unquoted, which the shell splits into words; anywhere in an
      argument of ``eval`` or of a shell given ``-c`` (also behind a
      launcher), which parse it again; or quoted as a whole argument of the
      command that runs the worker (``-m dynamo.<x>``) or of one that turns
      it into code or env (source, env, export, a shell, named by path or
      behind exec, nohup, command or builtin), unless it follows one of
      ``_WORKER_VALUE_OPTIONS``. A redirection target other than a
      here-string names a file and never counts. An assignment neither
      splits nor passes it on, but leaves its variable dangerous, and a
      reference to that variable names the env it was assigned from;
    - other unreadable env becomes ``_INERT``;
    - script variables, special parameters and image env that no
      ``envFrom`` source may define stay as written;
    - a default or alternate word is scanned in their place wherever the
      shell may use it.

    The scan does not follow control flow, so an assignment may not run:
    it leaves literal env and explicit dangerous env of its name, and a
    dangerous earlier assignment, in place. Command substitution, backticks
    and arithmetic are neither evaluated nor rejected, and neither are
    scripts the worker sources or runs, literal text that ``eval`` or a
    nested shell parses again, text that a shell or ``xargs`` reads from a
    heredoc, a here-string or a pipe, or a script ``su -c`` runs; a script
    variable counts as dangerous only when the script assigns it from
    dangerous env directly (not through an array, positional parameters, a
    function argument or a command's output).
    """

    return _ShellScan(script, env).run()


class _ShellScan:
    """One pass of ``_shell_expand`` over a script."""

    def __init__(self, script: str, env: _Env) -> None:
        self.script = script
        self.env = env
        self.out: list[str] = []
        self.index = 0
        self.quoted = False
        # The current word as far as the scan knows it: an expansion it does
        # not evaluate adds nothing, since it may be empty.
        self.word = ""
        self.in_word = False
        self.assignment = False
        self.dangerous = False
        # The known words of the current simple command, and its name.
        self.words: list[str] = []
        self.command: Optional[str] = None
        self.command_dangerous = False
        self.redirect = False
        self.here_string = False
        self.heredocs: list[tuple[str, bool]] = []
        # Script variables, and whether each was assigned dangerous env.
        self.variables: dict[str, bool] = {}
        # The first dangerous reference in the current word, and the one
        # each dangerous script variable was assigned from.
        self.source: Optional[str] = None
        self.sources: dict[str, str] = {}

    def run(self) -> str:
        while self.index < len(self.script):
            # A default or alternate word is spliced into the script.
            script = self.script
            char = script[self.index]
            if char == "\x00":
                end = script.find("\x00", self.index + 1)
                end = len(script) if end == -1 else end + 1
                if script.startswith(_OPAQUE, self.index):
                    self.dangerous = True
                    self.source = self.source or _opaque_reference(
                        script[self.index : end]
                    )
                self.emit(end, "")
            elif not self.quoted and char in _SHELL_BLANKS:
                self.end_word()
                self.copy(self.index + 1)
            elif not self.quoted and char in _SHELL_OPERATORS:
                self.operator()
            elif not self.quoted and char == "#" and not self.in_word:
                end = script.find("\n", self.index)
                self.copy(len(script) if end == -1 else end)
            elif char == "\\":
                self.escape()
            elif char == "'" and not self.quoted:
                end = script.find("'", self.index + 1)
                end = len(script) if end == -1 else end + 1
                self.emit(end, script[self.index + 1 : end - 1])
            elif char == '"':
                self.emit(self.index + 1, "")
                self.quoted = not self.quoted
            elif char == "`":
                self.emit(_backtick_end(script, self.index + 1), "")
            elif char == "$":
                self.dollar()
            else:
                self.emit(self.index + 1, char)
        self.end_word()
        self.end_command()
        return "".join(self.out)

    def copy(self, end: int) -> None:
        self.out.append(self.script[self.index : end])
        self.index = end

    def emit(self, end: int, known: str, text: Optional[str] = None) -> None:
        """Add the script up to ``end`` (or ``text`` for it) to the current word."""

        if not self.in_word:
            self.in_word = True
            self.assignment = self.command is None and bool(
                _SHELL_ASSIGNMENT.match(self.script, self.index)
            )
        self.out.append(self.script[self.index : end] if text is None else text)
        self.word += known
        self.index = end

    def end_word(self) -> None:
        if not self.in_word:
            return
        word = self.word
        self.command_dangerous |= self.dangerous
        if self.redirect:
            self.redirect = self.here_string = False
        else:
            match = _SHELL_ASSIGNMENT.match(word)
            if match and (self.assignment or self.command in _SHELL_DECLARATIONS):
                self.assign(match.group(1), self.dangerous)
            elif _SHELL_NAME.fullmatch(word) and (
                self.command == "read"
                or (self.command == "for" and self.words == ["for"])
            ):
                self.assign(word, False)
            self.words.append(word)
            if (
                self.command is None
                and not self.assignment
                and word not in _SHELL_RESERVED
            ):
                self.command = word
        self.word, self.in_word, self.assignment, self.dangerous = (
            "",
            False,
            False,
            False,
        )
        self.source = None

    def end_command(self) -> None:
        if self.command == "for" and len(self.words) > 1 and self.command_dangerous:
            self.variables[self.words[1]] = True
        self.words, self.command = [], None
        self.command_dangerous = self.redirect = self.here_string = False

    def assign(self, name: str, dangerous: bool) -> None:
        # The scan does not follow control flow, so the assignment may not
        # run: it never clears explicit dangerous env or an earlier
        # dangerous assignment.
        env = self.env
        self.variables[name] = (
            dangerous
            or self.variables.get(name, False)
            or (name in env.opaque and name not in env.inert)
        )
        if dangerous and self.source is not None:
            self.sources.setdefault(name, self.source)

    def operator(self) -> None:
        script, index = self.script, self.index
        self.end_word()
        if script.startswith("<<", index) and not script.startswith("<<<", index):
            heredoc, end = _shell_heredoc(script, index)
            if heredoc is not None:
                self.heredocs.append(heredoc)
            self.copy(end)
            return
        if script[index] in "<>" or script.startswith("&>", index):
            # A redirection, whose target is never an argument of the
            # command, unless a here-string feeds it the target's text.
            end = index + 1
            while end < len(script) and script[end] in "<>&|":
                end += 1
            self.here_string = script.startswith("<<<", index)
            self.copy(end)
            self.redirect = True
            return
        self.end_command()
        self.copy(index + 1)
        if script[index] == "\n":
            for delimiter, strip in self.heredocs:
                self.copy(_shell_heredoc_end(script, self.index, delimiter, strip))
            self.heredocs.clear()

    def escape(self) -> None:
        index = self.index
        following = self.script[index + 1 : index + 2]
        if following == "\n":
            # A line continuation joins the lines.
            self.index += 2
        elif self.quoted and following not in ('"', "\\", "$", "`"):
            self.emit(index + 1, "\\")
        else:
            self.emit(index + 2, following)

    def dollar(self) -> None:
        script, index = self.script, self.index
        following = script[index + 1 : index + 2]
        name = _SHELL_NAME.match(script, index + 1)
        if following == "(":
            # Command substitution or arithmetic, not evaluated.
            self.emit(_shell_substitution_end(script, index + 2), "")
        elif following == "{":
            self.braced()
        elif name is not None:
            self.reference(name.group(0), name.end())
        elif following and following in "@*#?$!-0123456789":
            self.emit(index + 2, "")
        else:
            self.emit(index + 1, "$")

    def braced(self) -> None:
        script, index = self.script, self.index
        close = _shell_brace_end(script, index + 2)
        if close == -1:
            self.emit(len(script), "")
            return
        body = script[index + 2 : close]
        name = _SHELL_NAME.match(body)
        if name is None:
            # A length, an indirection or a positional parameter.
            self.emit(close + 1, "")
            return
        operator = body[name.end() :]
        word_operator = operator.lstrip(":")[:1]
        if word_operator in ("-", "=", "+"):
            word = operator[operator.index(word_operator) + 1 :]
            kind = self.kind(name.group(0))
            value = self.env.values.get(name.group(0), "")
            known_set = kind == "readable" and (
                bool(value) or not operator.startswith(":")
            )
            # The shell expands the word where it replaces the value: an
            # alternate one where the value may be set, a default one where
            # it may be unset and is not unreadable env.
            if word_operator == "+":
                if known_set or kind != "readable":
                    self.script = script[:index] + word + script[close + 1 :]
                else:
                    self.index = close + 1
                return
            if not known_set and kind not in ("dangerous", "inert"):
                self.script = script[:index] + word + script[close + 1 :]
                return
        self.reference(name.group(0), close + 1, operator)

    def reference(self, name: str, end: int, operator: str = "") -> None:
        kind = self.kind(name)
        if kind == "readable":
            value = self.env.values[name]
            # The value, unless a default word replaces an empty one or an
            # operator derives another value from it.
            if operator[:1] in ("", "-", "=", "?") or (
                operator[:2] in (":-", ":=", ":?") and value
            ):
                self.literal(value, end)
            else:
                self.emit(end, "")
        elif kind == "unknown":
            self.emit(end, "")
        else:
            dangerous = kind == "dangerous"
            self.dangerous |= dangerous
            # A marker holds no whitespace: _worker_argv splits the script.
            shown = f"${name}"
            if self.variables.get(name) and name in self.sources:
                shown += f"(={self.sources[name]})"
            if dangerous:
                self.source = self.source or shown
            marker = _OPAQUE if dangerous and self.option_position() else _INERT
            self.emit(end, "", f"{marker}{shown}\x00")

    def kind(self, name: str) -> str:
        env = self.env
        assigned = self.variables.get(name)
        if assigned:
            return "dangerous"
        if name in env.values:
            # The script's own assignment may not run either.
            return "readable"
        if assigned is not None:
            return "unknown"
        if name in env.inert:
            return "inert"
        if name in env.opaque or _env_from_may_define(name, env.env_from_prefixes):
            return "dangerous"
        return "unknown"

    def literal(self, value: str, end: int) -> None:
        if self.quoted:
            self.emit(end, value, value.replace("\\", "\\\\").replace('"', '\\"'))
        elif self.in_word and self.assignment:
            self.emit(end, value, shlex.quote(value))
        else:
            # The shell splits an unquoted value into words.
            pieces = value.split()
            for position, piece in enumerate(pieces):
                if position or value[:1].isspace():
                    self.end_word()
                    self.out.append(" ")
                self.emit(self.index, piece, shlex.quote(piece))
            if value[-1:].isspace():
                self.end_word()
                self.out.append(" ")
            self.index = end

    def option_position(self) -> bool:
        """Whether a dangerous value here could be or create an option of the worker."""

        if self.in_word and self.assignment:
            return False
        if self.redirect and not self.here_string:
            # A redirection target names a file or descriptor.
            return False
        if not self.quoted or self.reparses():
            return True
        lead = self.word if self.in_word else ""
        if lead:
            # Attached to its option, or inside one's name.
            return lead.startswith("-") and "=" not in lead
        if self.words and self.words[-1] in _WORKER_VALUE_OPTIONS:
            return False
        if self.command is None:
            # The command's name: one word, a program to run.
            return False
        launched = self.launched()
        if launched in _SHELL_REPARSING or launched in _SHELLS:
            return True
        words = self.words
        for word, following in zip(words, [*words[1:], ""]):
            module = _PYTHON_MODULE_OPTION.fullmatch(word)
            if module is not None and (module.group(1) or following).startswith(
                "dynamo."
            ):
                return True
        return False

    def launched(self) -> str:
        """The current command's name by basename, past ``_SHELL_LAUNCHERS``."""

        if self.command is None:
            return ""
        for word in self.words[self.words.index(self.command) :]:
            name = posixpath.basename(word)
            if not word.startswith("-") and name not in _SHELL_LAUNCHERS:
                return name
        return ""

    def reparses(self) -> bool:
        """Whether the current command parses its arguments again as a script.

        ``eval`` does, also behind ``_SHELL_LAUNCHERS``, and so does a shell
        given ``-c``, also behind a launcher (``exec``, ``nohup``, ``env``),
        whose positional parameters its script may expand unquoted.
        """

        return self.launched() == "eval" or _shell_script_index(self.words) is not None


def _backtick_end(script: str, index: int) -> int:
    """The index after the backtick closing one opened before ``index``."""

    while index < len(script):
        if script[index] == "\\":
            index += 2
        elif script[index] == "`":
            return index + 1
        else:
            index += 1
    return len(script)


def _shell_brace_end(script: str, index: int) -> int:
    """The index of the ``}`` closing a ``${`` whose body starts at ``index``, or -1."""

    depth = 0
    while index < len(script):
        if script[index] == "\\":
            index += 2
            continue
        if script.startswith("${", index):
            depth += 1
            index += 2
            continue
        if script[index] == "}":
            if depth == 0:
                return index
            depth -= 1
        index += 1
    return -1


def _shell_substitution_end(script: str, index: int) -> int:
    """The index after the ``)`` closing a ``$(`` whose body starts at ``index``."""

    depth = 0
    quote = ""
    heredocs: list[tuple[str, bool]] = []
    word_start = True
    while index < len(script):
        char = script[index]
        if quote == "'":
            quote = "" if char == "'" else quote
            index += 1
            continue
        if char == "\\":
            index += 2
            word_start = False
            continue
        if char == "`":
            index = _backtick_end(script, index + 1)
            continue
        if script.startswith("$(", index):
            index = _shell_substitution_end(script, index + 2)
            continue
        if quote == '"':
            quote = "" if char == '"' else quote
            index += 1
            continue
        if char in "'\"":
            quote = char
        elif char == "#" and word_start:
            end = script.find("\n", index)
            index = len(script) if end == -1 else end
            continue
        elif script.startswith("<<", index) and not script.startswith("<<<", index):
            heredoc, index = _shell_heredoc(script, index)
            if heredoc is not None:
                heredocs.append(heredoc)
            continue
        elif char == "\n" and heredocs:
            index += 1
            for delimiter, strip in heredocs:
                index = _shell_heredoc_end(script, index, delimiter, strip)
            heredocs.clear()
            word_start = True
            continue
        elif char == "(":
            depth += 1
        elif char == ")":
            if depth == 0:
                return index + 1
            depth -= 1
        word_start = char in _SHELL_BLANKS or char in _SHELL_OPERATORS
        index += 1
    return len(script)


def _shell_heredoc(script: str, index: int) -> tuple[Optional[tuple[str, bool]], int]:
    """The heredoc ``<<`` at ``index`` opens, and the index after its delimiter.

    The heredoc is its delimiter and whether ``<<-`` strips leading tabs.
    """

    index += 2
    strip = script.startswith("-", index)
    if strip:
        index += 1
    while index < len(script) and script[index] in _SHELL_BLANKS:
        index += 1
    start = index
    while (
        index < len(script)
        and script[index] not in _SHELL_BLANKS
        and script[index] not in _SHELL_OPERATORS
    ):
        index += 1
    delimiter = re.sub(r"[\"'\\]", "", script[start:index])
    return ((delimiter, strip) if delimiter else None), index


def _shell_heredoc_end(script: str, index: int, delimiter: str, strip: bool) -> int:
    """The index after a heredoc body starting at ``index`` and its delimiter line."""

    while index < len(script):
        end = script.find("\n", index)
        end = len(script) if end == -1 else end
        line = script[index:end]
        index = end + 1
        if (line.lstrip("\t") if strip else line) == delimiter:
            break
    return min(index, len(script))


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


def _trace_environment(
    replay: Sequence[Mapping[str, Any]], env_from_prefixes: Sequence[str]
) -> dict[str, str]:
    """Replay trace values with distinct identities for runtime sources."""

    values: dict[str, str] = {}
    for entry in replay:
        name = str(entry["name"])
        source = entry.get("valueFrom")
        if source is not None:
            field_ref = source.get("fieldRef") if isinstance(source, Mapping) else None
            if (
                isinstance(field_ref, Mapping)
                and field_ref.get("fieldPath") == "metadata.name"
            ):
                value = _POD_NAME_VALUE
            else:
                marker = _INERT if _inert_source(source) else _OPAQUE
                value = f"{marker}{name}\x00"
        else:
            value = _expand(
                str(entry.get("value") or ""),
                values,
                (),
                env_from_prefixes=env_from_prefixes,
            )
        values[name] = value
    return values


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
    replayed = (
        _trace_environment(analysis.env.replay, analysis.env.env_from_prefixes)
        if analysis.env.replay is not None
        else None
    )
    errors: list[str] = []

    def literal(name: str) -> Optional[str]:
        entry = entries.get(name)
        if entry is None:
            return None
        if entry.get("valueFrom") is not None:
            errors.append(f"P10: frontend env {name} must be a literal value")
            return None
        if replayed is not None:
            return replayed.get(name)
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
        dgd_dirs = (
            (dgd_name,)
            if replayed is not None
            else (f"$({ENV_PARENT_DGD_NAME})", dgd_name)
        )
        pod_name_value = (
            _POD_NAME_VALUE if replayed is not None else f"$({ENV_POD_NAME})"
        )
        prefixes = [
            _join_dir(mount, segments_dir, dgd_dir, pod_name_value)
            for mount in mounts
            for dgd_dir in dgd_dirs
        ]
        # Directly in the pod's directory: the reconciler prunes only the
        # segments at its top level.
        if not any(
            (replayed is None or _OPAQUE not in path)
            and path.startswith(prefix)
            and len(path) > len(prefix)
            and "/" not in path[len(prefix) :]
            for prefix in prefixes
        ):
            display_path = str((entries.get(ENV_TRACE_PATH) or {}).get("value") or "")
            display_prefix = (
                _join_dir(mounts[0], segments_dir, dgd_name, f"$({ENV_POD_NAME})")
                if replayed is not None
                else prefixes[0]
            )
            expansion = " after env expansion" if replayed is not None else ""
            errors.append(
                f"P10: {ENV_TRACE_PATH}={display_path!r} must be a file prefix directly "
                f"under {display_prefix!r}{expansion} (no further '/')"
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
        text = env.expand(str(part))
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
                and _UNREADABLE not in value
                and not value.startswith("-")
            )
            found.append((flag, value if resolved else None))
    return found


def _component_model_flags(
    dgd: Mapping[str, Any], component: Mapping[str, Any]
) -> list[tuple[str, Optional[str]]]:
    """``_model_flags`` of every pod template the component runs (``_role_specs``)."""

    try:
        return [
            flag
            for _, spec in _role_specs(component)
            for flag in _model_flags(spec, _component_env(dgd, spec))
        ]
    except FrontendParseError:
        # An invalid environment has no trustworthy served-model name; P4 reports it.
        return []


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
        for _, value in _component_model_flags(dgd, component):
            if value:
                names.update(part for part in re.split(r"[\s,]+", value) if part)
    return names


def _csv(value: Optional[str]) -> list[str]:
    if not value:
        return []
    return [part.strip().lower() for part in value.split(",") if part.strip()]


def _unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))
