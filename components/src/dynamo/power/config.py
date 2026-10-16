# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for Dynamo power control.

This module is the shared contract between the launcher (the srt-slurm
``power_control`` stage today, the Dynamo Operator later) and the two
processes in this package:

* :class:`PowerControlConfig` -- the ``power_control`` recipe block, parsed
  from a plain mapping with :meth:`PowerControlConfig.from_dict`.
* :class:`FomSpec` -- a figure-of-merit specification in any of the accepted
  forms (preset name, ``prom:...`` series, ``python:...`` provider), with the
  static part of EPD's signal contract enforced by :func:`validate_fom`.
* :class:`PoolLayout` / :class:`GpuGroup` -- the unit of control. A GPU group
  is the set of GPUs on one node that belong to one pool; a node that would
  belong to two pools is rejected by :func:`build_pool_layouts`.
* :class:`MetricsUrl` -- a ``/metrics`` endpoint with the role the launcher
  tagged it with (``frontend=<url>`` / ``worker=<url>``), which is what a
  FoM ``scope`` selects.
* :func:`epd_environment` -- the node_agent's flags mapped onto the
  environment variable names EPD really reads.
* :func:`stale_after_s_for`, :func:`validate_gpu_identity`,
  :func:`nupd_budget_binds` -- the small rules the two processes and the
  launcher must agree on, kept here so they are unit-testable without GPUs.

A GPU index, everywhere in this package, is the host NVML index, which must
equal the ``/dev/nvidiaN`` minor number: that is what srt-slurm's
``gpu_indices`` / ``Endpoint.gpus_on_node`` mean, what the node_agent puts in
``NVIDIA_VISIBLE_DEVICES`` and what the applier hands to ``nvidia-smi -i``.

Nothing here imports EPD, pynvml or anything outside the standard library, so
the launcher can validate a recipe without GPUs or EPD installed.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

MODES = ("nupd", "ubpc", "both")
APPLIERS = ("sudo-smi", "dcgmi", "nvml")
WARMUP_POLICIES = ("adaptive", "fixed")
POOL_SHORTHANDS = ("shared", "per-role")
FOM_PRESETS = (
    "output_tokens",
    "tokens_per_gpu",
    "per_user_speed_p25",
    "tokens_per_joule",
    "ctx_tokens",
    "gen_tokens",
    "inv_ttft_p50",
)
PROM_METRIC_KINDS = ("counter", "gauge")
PROM_AGGREGATIONS = ("sum", "avg")
# Which --metrics-url entries a ``prom:`` FoM scrapes: every one, the ones the
# launcher tagged ``frontend=``, or the ones it tagged ``worker=``.
FOM_SCOPES = ("all", "frontend", "pool_workers")
METRICS_URL_ROLES = ("frontend", "worker")
_SCOPE_ROLE = {"all": None, "frontend": "frontend", "pool_workers": "worker"}
# The CLI field ``scope=<...>`` is part of the prom grammar, so a Prometheus
# label literally named ``scope`` cannot be used as a filter.
FOM_SCOPE_FIELD = "scope"

DEFAULT_MODE = "nupd"
DEFAULT_INTERVAL_S = 30
DEFAULT_PERF_THRESHOLD = 0.03
DEFAULT_UBPC_BUFFERS = (("prefill", 2.0), ("decode", 1.1))
# The node_agent's --ubpc-buffer default (one pool, one buffer).
DEFAULT_NODE_UBPC_BUFFER = 1.1
DEFAULT_WARMUP_POLICY = "adaptive"
DEFAULT_APPLIER = "sudo-smi"
# The applier falls back to the uniform budget when the decision file is older
# than this. EPD rewrites the file once per controller cycle, so the window is
# tied to interval_s through stale_after_s_for(): about STALE_CYCLES cycles
# (design section 5) and never less than two, or the applier would flap
# between the budget and EPD's caps once per cycle.
DEFAULT_STALE_AFTER_S = 90
STALE_CYCLES = 3
MIN_STALE_CYCLES = 2
DEFAULT_POLL_S = 5
# EPD's packaged UBPC buffer, used when a role has no entry and no scalar was given.
EPD_DEFAULT_UBPC_BUFFER = 1.2

_METRIC_NAME_RE = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")
_LABEL_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
_DOTTED_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_POWER_CONTROL_KEYS = frozenset(
    {
        "enabled",
        "mode",
        "pools",
        "budget_w_per_gpu",
        "fom",
        "interval_s",
        "perf_threshold",
        "ubpc_buffer",
        "min_tgp_w",
        "max_tgp_w",
        "warmup_policy",
        "applier",
        "epd_source",
        "dynamo_power_source",
        "critical",
    }
)
_POOL_KEYS = frozenset(
    {"name", "roles", "nodes", "budget_w_per_gpu", "fom", "ubpc_buffer", "min_tgp_w"}
)


class PowerControlConfigError(ValueError):
    """A ``power_control`` block, FoM spec or pool layout that cannot be used."""


# --------------------------------------------------------------------------- FoM


@dataclass(frozen=True)
class FomSpec:
    """One figure-of-merit specification.

    ``source`` selects the form:

    * ``"preset"`` -- ``name`` is one of :data:`FOM_PRESETS`; the real metric
      names behind each preset live in :mod:`dynamo.power.fom`.
    * ``"prom"`` -- ``name`` is a Prometheus metric scraped from the
      ``--metrics-url`` entries that ``scope`` selects; ``metric_kind`` says
      whether it is differenced into a rate (``counter``) or taken as-is
      (``gauge``); ``aggregation`` combines the matching series (``sum`` or
      ``avg``); ``labels`` filters them.
    * ``"python"`` -- ``name`` is ``module:Class``; the class is instantiated
      without arguments and must expose ``get_perf() -> float``.

    ``scope`` (``prom`` only) is one of :data:`FOM_SCOPES`: ``all`` scrapes
    every endpoint, ``frontend`` only the endpoints the launcher tagged
    ``frontend=``, ``pool_workers`` only the ones tagged ``worker=``. The
    node_agent refuses a scope that selects no endpoint.
    """

    source: str
    name: str
    metric_kind: str | None = None
    aggregation: str | None = None
    labels: tuple[tuple[str, str], ...] = ()
    scope: str = "all"

    def to_cli(self) -> str:
        """Render the spec in the node_agent's ``--fom`` grammar (round-trips)."""
        if self.source == "preset":
            return self.name
        if self.source == "python":
            return f"python:{self.name}"
        parts = ["prom", self.name]
        if self.labels:
            parts.append(",".join(f"{key}={value}" for key, value in self.labels))
        if self.scope != "all":
            parts.append(f"{FOM_SCOPE_FIELD}={self.scope}")
        parts.append(str(self.metric_kind))
        parts.append(str(self.aggregation))
        return ":".join(parts)

    def describe(self) -> str:
        """Human-readable one-liner for logs."""
        if self.source == "preset":
            return f"preset {self.name}"
        if self.source == "python":
            return f"python provider {self.name}"
        labels = (
            "{" + ",".join(f'{key}="{value}"' for key, value in self.labels) + "}"
            if self.labels
            else ""
        )
        scope = f" on {self.scope}" if self.scope != "all" else ""
        return f"{self.metric_kind} {self.name}{labels} ({self.aggregation}){scope}"


def parse_fom(raw: object) -> FomSpec:
    """Parse and validate a FoM given as a string or a mapping.

    Accepted strings: a preset name,
    ``prom:<metric>[:<k>=<v>,...][:scope=<frontend|pool_workers>]:<counter|gauge>:<sum|avg>``
    or ``python:<module>:<Class>``. Accepted mappings: ``{preset: name}``,
    ``{provider: "module:Class"}`` or ``{metric: ..., kind: ..., agg: ...,
    labels: {...}, scope: ...}``.
    """
    if isinstance(raw, str):
        spec = _parse_fom_string(raw.strip())
    elif isinstance(raw, Mapping):
        spec = _parse_fom_mapping(raw)
    else:
        raise PowerControlConfigError(
            f"fom must be a string or a mapping, got {type(raw).__name__}"
        )
    validate_fom(spec)
    return spec


_PROM_GRAMMAR = (
    "'prom:<metric>[:<label>=<value>,...][:scope=<frontend|pool_workers>]"
    ":<counter|gauge>:<sum|avg>'"
)


def _parse_fom_string(text: str) -> FomSpec:
    if text in FOM_PRESETS:
        return FomSpec(source="preset", name=text)
    if text.startswith("prom:"):
        return _parse_prom_cli(text)
    if text.startswith("python:"):
        return FomSpec(source="python", name=text.partition(":")[2])
    raise PowerControlConfigError(
        f"unknown fom {text!r}: expected a preset ({', '.join(FOM_PRESETS)}), "
        f"{_PROM_GRAMMAR} or 'python:<module>:<Class>'"
    )


def _parse_prom_cli(text: str) -> FomSpec:
    # Metric names may themselves contain ':' (vllm:generation_tokens_total), so
    # the two trailing fields are fixed and the metric is everything before the
    # optional scope and label fields.
    parts = text.split(":")
    if len(parts) < 4:
        raise PowerControlConfigError(f"fom {text!r}: expected {_PROM_GRAMMAR}")
    aggregation = parts[-1]
    metric_kind = parts[-2]
    middle = parts[1:-2]
    scope = "all"
    if len(middle) > 1 and middle[-1].startswith(f"{FOM_SCOPE_FIELD}="):
        scope = middle[-1].partition("=")[2]
        middle = middle[:-1]
    labels: tuple[tuple[str, str], ...] = ()
    if len(middle) > 1 and "=" in middle[-1]:
        labels = _parse_labels(middle[-1])
        middle = middle[:-1]
    metric = ":".join(middle)
    if not metric:
        raise PowerControlConfigError(f"fom {text!r}: missing metric name")
    return FomSpec(
        source="prom",
        name=metric,
        metric_kind=metric_kind,
        aggregation=aggregation,
        labels=labels,
        scope=scope,
    )


def _parse_labels(text: str) -> tuple[tuple[str, str], ...]:
    pairs = []
    for item in text.split(","):
        key, separator, value = item.partition("=")
        if not separator or not key or not value:
            raise PowerControlConfigError(
                f"fom label {item!r}: expected '<name>=<value>'"
            )
        pairs.append((key, value))
    return tuple(pairs)


def _parse_fom_mapping(raw: Mapping) -> FomSpec:
    keys = set(raw)
    if "preset" in keys:
        _reject_extra_keys(keys, {"preset"}, "fom")
        return FomSpec(source="preset", name=str(raw["preset"]))
    if "provider" in keys:
        _reject_extra_keys(keys, {"provider"}, "fom")
        return FomSpec(source="python", name=str(raw["provider"]))
    if "metric" in keys:
        _reject_extra_keys(keys, {"metric", "kind", "agg", "labels", "scope"}, "fom")
        if "kind" not in raw:
            raise PowerControlConfigError(
                "fom mapping with 'metric' needs 'kind: counter|gauge'"
            )
        labels_raw = raw.get("labels") or {}
        if not isinstance(labels_raw, Mapping):
            raise PowerControlConfigError("fom 'labels' must be a mapping")
        labels = tuple((str(key), str(value)) for key, value in labels_raw.items())
        return FomSpec(
            source="prom",
            name=str(raw["metric"]),
            metric_kind=str(raw["kind"]),
            aggregation=str(raw.get("agg", "sum")),
            labels=labels,
            scope=str(raw.get("scope", "all")),
        )
    raise PowerControlConfigError(
        "fom mapping needs one of 'preset', 'provider' or 'metric'"
    )


def validate_fom(spec: FomSpec) -> None:
    """Check the static part of EPD's signal contract.

    EPD needs a number that is higher-is-better, a per-interval rate at pool
    scope, with ``0.0`` reserved for "no signal". Statically that means: a
    known preset (each one is built to satisfy the contract), or a Prometheus
    series whose kind says how it becomes a rate (counters are differenced,
    gauges taken as-is), whose aggregation covers every scraped endpoint and
    whose scope names a known endpoint set, or a loadable ``module:Class``.
    Direction (higher is better) cannot be checked here: invert latencies
    yourself or use the ``inv_ttft_p50`` / ``per_user_speed_p25`` presets.
    """
    if spec.source == "preset":
        if spec.name not in FOM_PRESETS:
            raise PowerControlConfigError(
                f"unknown fom preset {spec.name!r}; choose from {', '.join(FOM_PRESETS)}"
            )
        return
    if spec.source == "prom":
        if not _METRIC_NAME_RE.match(spec.name):
            raise PowerControlConfigError(
                f"fom metric {spec.name!r} is not a valid Prometheus metric name"
            )
        if spec.metric_kind not in PROM_METRIC_KINDS:
            raise PowerControlConfigError(
                f"fom metric kind {spec.metric_kind!r} must be one of "
                f"{', '.join(PROM_METRIC_KINDS)} (a histogram needs a preset or a "
                "python provider)"
            )
        if spec.aggregation not in PROM_AGGREGATIONS:
            raise PowerControlConfigError(
                f"fom aggregation {spec.aggregation!r} must be one of "
                f"{', '.join(PROM_AGGREGATIONS)}"
            )
        for key, _value in spec.labels:
            if key == FOM_SCOPE_FIELD:
                raise PowerControlConfigError(
                    f"fom label {FOM_SCOPE_FIELD!r} is reserved for the scope field "
                    f"({FOM_SCOPE_FIELD}=<{'|'.join(FOM_SCOPES[1:])}>)"
                )
            if not _LABEL_NAME_RE.match(key):
                raise PowerControlConfigError(
                    f"fom label name {key!r} is not a valid Prometheus label name"
                )
        if spec.scope not in FOM_SCOPES:
            raise PowerControlConfigError(
                f"fom scope {spec.scope!r} must be one of {', '.join(FOM_SCOPES)}"
            )
        return
    if spec.source == "python":
        module, separator, class_name = spec.name.partition(":")
        if (
            not separator
            or not _DOTTED_NAME_RE.match(module)
            or not _IDENTIFIER_RE.match(class_name)
        ):
            raise PowerControlConfigError(
                f"fom python provider {spec.name!r}: expected '<module>:<Class>'"
            )
        return
    raise PowerControlConfigError(f"unknown fom source {spec.source!r}")


# ------------------------------------------------------------------------- pools


@dataclass(frozen=True)
class PoolSpec:
    """One power pool: the nodes one EPD controller manages together.

    ``min_tgp_w`` is a per-GPU floor the *applier* enforces
    (``--min-w-per-gpu``: a requested cap below it is raised and reported as
    ``clamped``). EPD's own cap law has no floor: its ``MIN_TGP`` is read only
    by the history-based ensemble, which the node_agent keeps off.
    """

    name: str
    budget_w_per_gpu: int
    fom: FomSpec
    ubpc_buffer: float
    roles: tuple[str, ...] = ()
    nodes: tuple[str, ...] = ()
    min_tgp_w: int | None = None


@dataclass(frozen=True)
class PowerControlConfig:
    """The ``power_control`` block of an srt-slurm recipe (or a DGD).

    ``pools`` is ``"shared"``, ``"per-role"`` or an explicit tuple of
    :class:`PoolSpec`. The top-level ``budget_w_per_gpu`` and ``fom`` are the
    defaults every pool inherits; they are required unless every explicit
    pool sets its own. ``ubpc_buffer`` is stored as ``(role, buffer)`` pairs;
    a scalar is stored under the role ``"*"``.
    """

    enabled: bool = True
    mode: str = DEFAULT_MODE
    pools: str | tuple[PoolSpec, ...] = "shared"
    budget_w_per_gpu: int | None = None
    fom: FomSpec | None = None
    interval_s: int = DEFAULT_INTERVAL_S
    perf_threshold: float = DEFAULT_PERF_THRESHOLD
    ubpc_buffer: tuple[tuple[str, float], ...] = DEFAULT_UBPC_BUFFERS
    min_tgp_w: int | None = None
    max_tgp_w: int | None = None
    warmup_policy: str = DEFAULT_WARMUP_POLICY
    applier: str = DEFAULT_APPLIER
    epd_source: str | None = None
    dynamo_power_source: str | None = None
    critical: bool = True

    @property
    def use_ubpc(self) -> bool:
        """Whether EPD's UBPC calibration runs (``ubpc`` and ``both`` modes).

        ``ubpc`` and ``both`` configure EPD identically: NUPD redistributes
        whenever the budget binds, so the two differ only by the user choosing
        a budget at or above the max TGP for ``ubpc``.
        """
        return self.mode != "nupd"

    @property
    def stale_after_s(self) -> int:
        """The ``--stale-after-s`` the launcher renders for every applier."""
        return stale_after_s_for(self.interval_s)

    def buffer_for(self, roles: Sequence[str]) -> float:
        """UBPC buffer for a pool made of ``roles``.

        EPD has one buffer per controller, so a pool that spans several roles
        gets the largest buffer among them (the prefill value protects bursts).
        A scalar block value applies to every role; a role without an entry
        falls back to EPD's packaged default.
        """
        table = dict(self.ubpc_buffer)
        if "*" in table:
            return table["*"]
        candidates = [table[role] for role in roles if role in table]
        if not candidates:
            return EPD_DEFAULT_UBPC_BUFFER
        return max(candidates)

    @classmethod
    def from_dict(cls, raw: Mapping) -> "PowerControlConfig":
        """Parse and validate a ``power_control`` mapping (for example from YAML)."""
        if not isinstance(raw, Mapping):
            raise PowerControlConfigError(
                f"power_control must be a mapping, got {type(raw).__name__}"
            )
        _reject_extra_keys(set(raw), _POWER_CONTROL_KEYS, "power_control")

        mode = _one_of(raw.get("mode", DEFAULT_MODE), MODES, "power_control.mode")
        budget = _optional_positive_int(
            raw.get("budget_w_per_gpu"), "power_control.budget_w_per_gpu"
        )
        fom = parse_fom(raw["fom"]) if "fom" in raw else None
        buffers = _parse_ubpc_buffers(
            raw.get("ubpc_buffer", dict(DEFAULT_UBPC_BUFFERS))
        )
        min_tgp_w = _optional_positive_int(
            raw.get("min_tgp_w"), "power_control.min_tgp_w (null = no floor)"
        )
        _check_floor(min_tgp_w, budget, "power_control")
        max_tgp_w = _optional_positive_int(
            raw.get("max_tgp_w"), "power_control.max_tgp_w"
        )
        interval_s = _positive_int(
            raw.get("interval_s", DEFAULT_INTERVAL_S), "power_control.interval_s"
        )
        check_stale_window(stale_after_s_for(interval_s), interval_s)
        config = cls(
            enabled=_bool(raw.get("enabled", True), "power_control.enabled"),
            mode=mode,
            pools="shared",
            budget_w_per_gpu=budget,
            fom=fom,
            interval_s=interval_s,
            perf_threshold=_fraction(
                raw.get("perf_threshold", DEFAULT_PERF_THRESHOLD),
                "power_control.perf_threshold",
            ),
            ubpc_buffer=buffers,
            min_tgp_w=min_tgp_w,
            max_tgp_w=max_tgp_w,
            warmup_policy=_one_of(
                raw.get("warmup_policy", DEFAULT_WARMUP_POLICY),
                WARMUP_POLICIES,
                "power_control.warmup_policy",
            ),
            applier=_one_of(
                raw.get("applier", DEFAULT_APPLIER), APPLIERS, "power_control.applier"
            ),
            epd_source=_optional_str(raw.get("epd_source"), "power_control.epd_source"),
            dynamo_power_source=_optional_str(
                raw.get("dynamo_power_source"), "power_control.dynamo_power_source"
            ),
            critical=_bool(raw.get("critical", True), "power_control.critical"),
        )
        pools_raw = raw.get("pools", "shared")
        if isinstance(pools_raw, str):
            pools = _one_of(pools_raw, POOL_SHORTHANDS, "power_control.pools")
            if budget is None:
                raise PowerControlConfigError(
                    "power_control.budget_w_per_gpu is required with " f"pools: {pools}"
                )
            if fom is None:
                raise PowerControlConfigError(
                    "power_control.fom is required: name the objective EPD protects "
                    f"(a preset such as {FOM_PRESETS[1]!r}, a 'prom:' series or a "
                    "'python:' provider)"
                )
            return replace(config, pools=pools)
        if not isinstance(pools_raw, Sequence) or not pools_raw:
            raise PowerControlConfigError(
                "power_control.pools must be 'shared', 'per-role' or a non-empty list"
            )
        explicit = tuple(_parse_pool(config, entry) for entry in pools_raw)
        names = [pool.name for pool in explicit]
        if len(set(names)) != len(names):
            raise PowerControlConfigError(f"duplicate pool names in {names}")
        return replace(config, pools=explicit)


def _parse_pool(config: PowerControlConfig, raw: object) -> PoolSpec:
    if not isinstance(raw, Mapping):
        raise PowerControlConfigError(
            "each power_control.pools entry must be a mapping"
        )
    _reject_extra_keys(set(raw), _POOL_KEYS, "power_control.pools[]")
    if "name" not in raw or not str(raw["name"]).strip():
        raise PowerControlConfigError("each pool needs a non-empty 'name'")
    name = str(raw["name"])
    roles = tuple(
        str(role) for role in _string_list(raw.get("roles"), f"pool {name} roles")
    )
    nodes = tuple(
        str(node) for node in _string_list(raw.get("nodes"), f"pool {name} nodes")
    )
    if bool(roles) == bool(nodes):
        raise PowerControlConfigError(
            f"pool {name}: give exactly one of 'roles' or 'nodes'"
        )
    budget = _optional_positive_int(
        raw.get("budget_w_per_gpu"), f"pool {name} budget_w_per_gpu"
    )
    if budget is None:
        budget = config.budget_w_per_gpu
    if budget is None:
        raise PowerControlConfigError(
            f"pool {name}: budget_w_per_gpu is required (no top-level default)"
        )
    fom = parse_fom(raw["fom"]) if "fom" in raw else config.fom
    if fom is None:
        raise PowerControlConfigError(
            f"pool {name}: fom is required (no top-level default)"
        )
    if "ubpc_buffer" in raw:
        buffer = _positive_float(raw["ubpc_buffer"], f"pool {name} ubpc_buffer")
    else:
        buffer = config.buffer_for(roles)
    min_tgp_w = (
        _optional_positive_int(
            raw.get("min_tgp_w"), f"pool {name} min_tgp_w (null = no floor)"
        )
        if "min_tgp_w" in raw
        else config.min_tgp_w
    )
    _check_floor(min_tgp_w, budget, f"pool {name}")
    return PoolSpec(
        name=name,
        budget_w_per_gpu=budget,
        fom=fom,
        ubpc_buffer=buffer,
        roles=roles,
        nodes=nodes,
        min_tgp_w=min_tgp_w,
    )


def _check_floor(min_tgp_w: int | None, budget_w_per_gpu: int | None, where: str):
    if min_tgp_w is None or budget_w_per_gpu is None:
        return
    if min_tgp_w > budget_w_per_gpu:
        raise PowerControlConfigError(
            f"{where}: min_tgp_w {min_tgp_w} W is above budget_w_per_gpu "
            f"{budget_w_per_gpu} W; the applier's fail-safe cap (the budget) would "
            "sit below the floor"
        )


def _parse_ubpc_buffers(raw: object) -> tuple[tuple[str, float], ...]:
    if isinstance(raw, Mapping):
        if not raw:
            raise PowerControlConfigError("power_control.ubpc_buffer mapping is empty")
        return tuple(
            (str(role), _positive_float(value, f"power_control.ubpc_buffer.{role}"))
            for role, value in raw.items()
        )
    return (("*", _positive_float(raw, "power_control.ubpc_buffer")),)


# ------------------------------------------------------------- shared rules


def stale_after_s_for(interval_s: int) -> int:
    """The applier's ``--stale-after-s`` for a controller cycle of ``interval_s``.

    EPD rewrites ``power_decision.json`` once per cycle, so the staleness
    window must span several cycles: about :data:`STALE_CYCLES` (design
    section 5), and never below :data:`DEFAULT_STALE_AFTER_S`.
    """
    return max(DEFAULT_STALE_AFTER_S, STALE_CYCLES * int(interval_s))


def check_stale_window(stale_after_s: float, interval_s: int) -> None:
    """Refuse a staleness window shorter than :data:`MIN_STALE_CYCLES` cycles.

    Between two decision writes the file's timestamp does not move; a window
    shorter than two cycles would trip the applier's fail-safe every cycle and
    flap the caps between the budget and EPD's decision.
    """
    if stale_after_s < MIN_STALE_CYCLES * interval_s:
        raise PowerControlConfigError(
            f"stale_after_s {stale_after_s:g} s is shorter than {MIN_STALE_CYCLES} "
            f"controller cycles of interval_s {interval_s} s; the applier would "
            "flap between the uniform budget and EPD's caps"
        )


def validate_gpu_identity(
    device_count: int, minor_by_index: Mapping[int, int], gpu_indices: Sequence[int]
) -> None:
    """Check that the applier's GPU indices name the GPUs the node_agent senses.

    The node_agent identifies its group by NVML *minor numbers* (it may see a
    restricted device set), the applier by host NVML *indices* (``nvidia-smi
    -i``, ``nvmlDeviceGetHandleByIndex``, DCGM GPU ids). The two agree only
    when every index ``i`` of the group exists and ``/dev/nvidia<i>`` is NVML
    device ``i``; on a host where that does not hold the two sides would
    silently act on different GPUs, so the applier refuses to start.
    """
    for index in gpu_indices:
        if index >= device_count:
            raise PowerControlConfigError(
                f"GPU index {index} does not exist: this host has {device_count} "
                "NVML devices"
            )
        minor = minor_by_index.get(index)
        if minor is None:
            raise PowerControlConfigError(
                f"GPU index {index}: no NVML minor number was read for it"
            )
        if minor != index:
            raise PowerControlConfigError(
                f"NVML device {index} is /dev/nvidia{minor}: GPU indices and minor "
                "numbers disagree on this host, so the node_agent (minor numbers) "
                "and the applier (NVML indices) would act on different GPUs"
            )


def nupd_budget_binds(budget_w_per_gpu: float, max_tgp_w: float) -> bool:
    """Whether NUPD's redistribution can act: the budget is below the max TGP.

    At or above the max TGP every node already sits at its ceiling, so
    ``mode: nupd`` is refused and ``mode: ubpc`` differs from ``both`` only in
    this case (with a binding budget NUPD redistributes in both).
    """
    return budget_w_per_gpu < max_tgp_w


# ------------------------------------------------------------------ GPU groups


@dataclass(frozen=True)
class WorkerPlacement:
    """Where one worker's GPUs sit: the role -> node -> GPU-index map.

    srt-slurm fills this from ``Endpoint.gpus_on_node`` when present (multi-node
    workers such as TEP6 that span two servers) and from ``gpu_indices``
    otherwise. ``worker`` is only used in messages.
    """

    role: str
    node: str
    gpu_indices: tuple[int, ...]
    worker: str = ""


@dataclass(frozen=True)
class GpuGroup:
    """The unit of control: the GPUs on one node that belong to one pool."""

    pool: str
    node: str
    gpu_indices: tuple[int, ...]


@dataclass(frozen=True)
class PoolLayout:
    """A pool resolved onto GPU groups. ``groups[0]`` hosts the controller (rank 0)."""

    spec: PoolSpec
    groups: tuple[GpuGroup, ...]

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def nodes(self) -> tuple[str, ...]:
        return tuple(group.node for group in self.groups)

    @property
    def controller_node(self) -> str:
        return self.groups[0].node

    @property
    def gpu_count(self) -> int:
        """GPUs in use by the pool (the budget counts these, not nodes x 4)."""
        return sum(len(group.gpu_indices) for group in self.groups)

    def rank_of(self, node: str) -> int:
        """EPD rank of ``node`` inside this pool (its position in the roster)."""
        return self.nodes.index(node)

    def group_for(self, node: str) -> GpuGroup:
        return self.groups[self.rank_of(node)]


def resolve_pools(
    config: PowerControlConfig, placements: Sequence[WorkerPlacement]
) -> tuple[PoolSpec, ...]:
    """Expand ``shared`` / ``per-role`` into explicit :class:`PoolSpec` entries."""
    if not isinstance(config.pools, str):
        return config.pools
    if config.budget_w_per_gpu is None or config.fom is None:
        raise PowerControlConfigError(
            "power_control.budget_w_per_gpu and power_control.fom are required "
            f"with pools: {config.pools}"
        )
    roles = _unique(placement.role for placement in placements)
    if config.pools == "shared":
        return (
            PoolSpec(
                name="shared",
                budget_w_per_gpu=config.budget_w_per_gpu,
                fom=config.fom,
                ubpc_buffer=config.buffer_for(roles),
                roles=roles,
                min_tgp_w=config.min_tgp_w,
            ),
        )
    return tuple(
        PoolSpec(
            name=role,
            budget_w_per_gpu=config.budget_w_per_gpu,
            fom=config.fom,
            ubpc_buffer=config.buffer_for((role,)),
            roles=(role,),
            min_tgp_w=config.min_tgp_w,
        )
        for role in roles
    )


def build_pool_layouts(
    config: PowerControlConfig, placements: Sequence[WorkerPlacement]
) -> tuple[PoolLayout, ...]:
    """Map pools onto GPU groups and refuse layouts the design cannot express.

    Rules: every worker belongs to exactly one pool; two workers of one pool
    may share a node (their GPU indices are merged into one group, the TEP6
    case) but may not claim the same GPU twice; a node that two pools would
    share is rejected. Groups keep the order in which their nodes first
    appear, so the pool's first node is the controller.
    """
    if not placements:
        raise PowerControlConfigError("no worker placements: nothing to control")
    pools = resolve_pools(config, placements)
    groups: dict[tuple[str, str], list[int]] = {}
    node_owner: dict[str, str] = {}
    for placement in placements:
        label = placement.worker or f"{placement.role} worker"
        indices = _validated_gpu_indices(
            placement.gpu_indices, f"{label} on {placement.node}"
        )
        matching = [
            pool
            for pool in pools
            if placement.role in pool.roles or placement.node in pool.nodes
        ]
        if not matching:
            raise PowerControlConfigError(
                f"{label} on {placement.node} belongs to no pool; every worker must "
                "be covered by exactly one pool"
            )
        if len(matching) > 1:
            raise PowerControlConfigError(
                f"{label} on {placement.node} matches pools "
                f"{[pool.name for pool in matching]}; pools must not overlap"
            )
        pool = matching[0]
        owner = node_owner.setdefault(placement.node, pool.name)
        if owner != pool.name:
            raise PowerControlConfigError(
                f"node {placement.node} would be split between pools {owner!r} and "
                f"{pool.name!r}; a GPU group must belong to one pool"
            )
        claimed = groups.setdefault((pool.name, placement.node), [])
        overlap = sorted(set(claimed) & set(indices))
        if overlap:
            raise PowerControlConfigError(
                f"GPU indices {overlap} on {placement.node} are claimed twice "
                f"within pool {pool.name!r}"
            )
        claimed.extend(indices)
    layouts = []
    for pool in pools:
        pool_groups = tuple(
            GpuGroup(pool=pool.name, node=node, gpu_indices=tuple(sorted(indices)))
            for (pool_name, node), indices in groups.items()
            if pool_name == pool.name
        )
        if not pool_groups:
            raise PowerControlConfigError(f"pool {pool.name!r} matches no worker")
        layouts.append(PoolLayout(spec=pool, groups=pool_groups))
    return tuple(layouts)


def check_visible_gpus(
    nvml_minor_numbers: Sequence[int], gpu_indices: Sequence[int]
) -> None:
    """Verify that EPD will sense exactly this group's GPUs.

    EPD's power monitor reads NVML devices ``0 .. n-1`` where ``n`` is the
    group size, so the first ``n`` visible devices (by NVML minor number, the
    host's GPU index) must be the group's indices in order. That holds when
    the container sees only the group (``NVIDIA_VISIBLE_DEVICES`` = the
    indices) or when the group is a prefix of the node's GPUs.
    """
    expected = tuple(sorted(gpu_indices))
    visible = tuple(nvml_minor_numbers)
    if len(visible) < len(expected):
        raise PowerControlConfigError(
            f"NVML sees {len(visible)} GPUs but the GPU group has {len(expected)}: "
            f"{list(expected)}"
        )
    if visible[: len(expected)] != expected:
        raise PowerControlConfigError(
            f"the first {len(expected)} visible GPUs are {list(visible[: len(expected)])}"
            f" (NVML minor numbers) but the GPU group is {list(expected)}; EPD would "
            "sense the wrong GPUs. Launch the node_agent with "
            f"NVIDIA_VISIBLE_DEVICES={','.join(str(i) for i in expected)}"
        )


# ------------------------------------------------------------- metrics URLs


@dataclass(frozen=True)
class MetricsUrl:
    """One ``/metrics`` endpoint and the role the launcher tagged it with.

    ``--metrics-url frontend=http://...`` tags the frontend's HTTP port,
    ``worker=http://...`` a pool worker's system-metrics port; an untagged URL
    has ``role`` ``None`` and is scraped only by ``scope: all`` FoMs.
    """

    url: str
    role: str | None = None

    def to_cli(self) -> str:
        return f"{self.role}={self.url}" if self.role else self.url


def parse_metrics_urls(values: Sequence[str]) -> tuple[MetricsUrl, ...]:
    """Flatten repeated/comma-separated ``--metrics-url`` values, keeping tags."""
    entries = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            role, separator, rest = item.partition("=")
            if separator and role in METRICS_URL_ROLES:
                entries.append(MetricsUrl(url=rest.strip(), role=role))
            else:
                entries.append(MetricsUrl(url=item))
    for entry in entries:
        if not (entry.url.startswith("http://") or entry.url.startswith("https://")):
            raise PowerControlConfigError(
                f"metrics url {entry.url!r} must start with http:// or https:// "
                f"(optionally prefixed with one of {'/'.join(METRICS_URL_ROLES)}=)"
            )
    return tuple(entries)


def urls_for_scope(urls: Sequence[MetricsUrl], scope: str) -> tuple[str, ...]:
    """The URLs a FoM of ``scope`` scrapes (see :data:`FOM_SCOPES`)."""
    if scope not in _SCOPE_ROLE:
        raise PowerControlConfigError(
            f"fom scope {scope!r} must be one of {', '.join(FOM_SCOPES)}"
        )
    role = _SCOPE_ROLE[scope]
    if role is None:
        return tuple(entry.url for entry in urls)
    return tuple(entry.url for entry in urls if entry.role == role)


# ---------------------------------------------------------------- node_agent


@dataclass(frozen=True)
class NodeAgentSettings:
    """The node_agent's resolved command line.

    There is no ``min_tgp_w`` here on purpose: the node_agent refuses
    ``--min-tgp-w`` because EPD's cap law does not enforce it; the floor lives
    in the applier (``--min-w-per-gpu``).
    """

    pool_name: str
    rank: int
    nodelist: tuple[str, ...]
    log_dir: str
    gpu_indices: tuple[int, ...]
    mode: str
    budget_w_per_gpu: int
    fom: FomSpec
    interval_s: int = DEFAULT_INTERVAL_S
    perf_threshold: float = DEFAULT_PERF_THRESHOLD
    ubpc_buffer: float = DEFAULT_NODE_UBPC_BUFFER
    max_tgp_w: int | None = None
    warmup_policy: str = DEFAULT_WARMUP_POLICY
    metrics_urls: tuple[MetricsUrl, ...] = ()
    gpus_per_node: int | None = None
    # ``--pool-gpu-count``: the pool's exact GPU total when nodes hold unequal shares.
    pool_gpu_count_override: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise PowerControlConfigError(f"mode {self.mode!r} not in {MODES}")
        if self.warmup_policy not in WARMUP_POLICIES:
            raise PowerControlConfigError(
                f"warmup policy {self.warmup_policy!r} not in {WARMUP_POLICIES}"
            )
        if not self.nodelist:
            raise PowerControlConfigError("nodelist is empty")
        if not 0 <= self.rank < len(self.nodelist):
            raise PowerControlConfigError(
                f"rank {self.rank} outside the pool roster of {len(self.nodelist)} nodes"
            )
        _validated_gpu_indices(self.gpu_indices, "node_agent --gpu-indices")
        if self.budget_w_per_gpu <= 0:
            raise PowerControlConfigError("budget_w_per_gpu must be > 0 W")
        if self.interval_s < 1:
            raise PowerControlConfigError("interval_s must be >= 1 s")
        if not 0.0 < self.perf_threshold < 1.0:
            raise PowerControlConfigError("perf_threshold must be a fraction in (0, 1)")
        if self.ubpc_buffer <= 0.0:
            raise PowerControlConfigError("ubpc_buffer must be > 0")
        if self.gpus_per_node is not None and self.gpus_per_node <= 0:
            raise PowerControlConfigError("gpus_per_node must be > 0")
        if (
            self.pool_gpu_count_override is not None
            and self.pool_gpu_count_override <= 0
        ):
            raise PowerControlConfigError("pool_gpu_count must be > 0")
        if any(not isinstance(entry, MetricsUrl) for entry in self.metrics_urls):
            raise PowerControlConfigError(
                "metrics_urls must be MetricsUrl entries (use parse_metrics_urls)"
            )
        validate_fom(self.fom)
        if self.rank == 0:
            if not self.metrics_urls:
                raise PowerControlConfigError(
                    "rank 0 runs the FoM provider and needs at least one --metrics-url"
                )
            scope = self.fom.scope if self.fom.source == "prom" else "all"
            if not urls_for_scope(self.metrics_urls, scope):
                tag = _SCOPE_ROLE[scope]
                raise PowerControlConfigError(
                    f"fom scope {scope!r} selects none of the metrics URLs "
                    f"{[entry.to_cli() for entry in self.metrics_urls]}; tag the "
                    f"matching --metrics-url values as {tag}=<url>"
                )

    @property
    def use_ubpc(self) -> bool:
        return self.mode != "nupd"

    @property
    def is_controller(self) -> bool:
        return self.rank == 0

    @property
    def all_metrics_urls(self) -> tuple[str, ...]:
        return urls_for_scope(self.metrics_urls, "all")

    @property
    def frontend_metrics_urls(self) -> tuple[str, ...]:
        return urls_for_scope(self.metrics_urls, "frontend")

    @property
    def worker_metrics_urls(self) -> tuple[str, ...]:
        return urls_for_scope(self.metrics_urls, "pool_workers")

    @property
    def pool_gpu_count(self) -> int:
        """GPUs the pool controls, for per-GPU normalisation of the FoM.

        ``--pool-gpu-count`` when the launcher passes the exact total; else
        ``len(nodelist) x gpus_per_node`` when ``--gpus-per-node`` is given,
        else ``len(nodelist) x len(gpu_indices)``. Without the exact total, a
        node with fewer GPUs in the pool (TEP6's last node) makes this a
        constant over-count; EPD acts on relative change, so a constant divisor
        does not alter its decisions.
        """
        if self.pool_gpu_count_override is not None:
            return self.pool_gpu_count_override
        per_node = self.gpus_per_node or len(self.gpu_indices)
        return len(self.nodelist) * per_node


def epd_environment(
    settings: NodeAgentSettings, *, this_node: str, max_tgp_w: float
) -> dict[str, str]:
    """Map node_agent settings onto the environment variable names EPD reads.

    The names are EPD's own (``src/epd/config.py`` and ``_cli.py``):
    ``LOG_DIR``/``EPD_LOG_DIR`` for the output root, ``EPD_NODELIST`` /
    ``EPD_NUM_NODES`` to scope the controller to its pool inside a shared
    SLURM allocation, ``GPUS_PER_NODE`` for the per-node power monitor,
    ``MAX_TGP`` / ``AVERAGE_GPU_POWER_LIMIT`` for the cap law (``MIN_TGP`` is
    pinned to ``0``: EPD reads it only in the history-based ensemble, which
    ``NUPD_SKIP_HISTORY=1`` keeps off), ``NUPD_SKIP_*`` toggles,
    ``EPD_PERF_THRESHOLD``, ``UBPC_BUFFER``, ``EPD_WARMUP_POLICY``,
    ``EPD_ACTUATOR=file`` and ``SLURMD_NODENAME`` (the key of this node's
    snapshot file and decision entry). The ``DYNAMO_POWER_*`` variables are
    for ``python:`` providers.
    """
    env = {
        "LOG_DIR": settings.log_dir,
        "EPD_LOG_DIR": settings.log_dir,
        "EPD_NODELIST": ",".join(settings.nodelist),
        "EPD_NUM_NODES": str(len(settings.nodelist)),
        "SLURMD_NODENAME": this_node,
        "GPUS_PER_NODE": str(len(settings.gpu_indices)),
        "MAX_TGP": f"{max_tgp_w:g}",
        "AVERAGE_GPU_POWER_LIMIT": str(settings.budget_w_per_gpu),
        "MIN_TGP": "0",
        "NUPD_SKIP_PERFORMANCE_FEEDBACK": "0",
        "NUPD_SKIP_HISTORY": "1",
        "NUPD_SKIP_UBPC": "0" if settings.use_ubpc else "1",
        "EPD_PERF_THRESHOLD": f"{settings.perf_threshold:g}",
        "UBPC_BUFFER": f"{settings.ubpc_buffer:g}",
        "EPD_WARMUP_POLICY": settings.warmup_policy,
        "EPD_ACTUATOR": "file",
        "EPD_RUN_TRACE": "1",
        "EPD_SUMMARY_STDOUT": "1",
        "DYNAMO_POWER_POOL": settings.pool_name,
        "DYNAMO_POWER_METRICS_URLS": ",".join(settings.all_metrics_urls),
        "DYNAMO_POWER_FRONTEND_METRICS_URLS": ",".join(settings.frontend_metrics_urls),
        "DYNAMO_POWER_WORKER_METRICS_URLS": ",".join(settings.worker_metrics_urls),
        "DYNAMO_POWER_POOL_GPU_COUNT": str(settings.pool_gpu_count),
    }
    return env


# ------------------------------------------------------------------- parsing


def parse_gpu_indices(text: str) -> tuple[int, ...]:
    """Parse ``"0,1,2,3"`` into a sorted tuple of distinct non-negative ints."""
    items = [item.strip() for item in text.split(",") if item.strip()]
    if not items:
        raise PowerControlConfigError("gpu indices are empty")
    try:
        indices = tuple(int(item) for item in items)
    except ValueError as exc:
        raise PowerControlConfigError(f"gpu indices {text!r}: {exc}") from exc
    return _validated_gpu_indices(indices, "gpu indices")


def parse_nodelist(text: str) -> tuple[str, ...]:
    """Parse ``"node1,node2"`` (already expanded names) into a tuple."""
    nodes = tuple(item.strip() for item in text.split(",") if item.strip())
    if not nodes:
        raise PowerControlConfigError("nodelist is empty")
    if len(set(nodes)) != len(nodes):
        raise PowerControlConfigError(f"nodelist {text!r} repeats a node")
    return nodes


def _validated_gpu_indices(indices: Sequence[int], what: str) -> tuple[int, ...]:
    values = tuple(int(index) for index in indices)
    if not values:
        raise PowerControlConfigError(f"{what}: no GPU indices")
    if any(index < 0 for index in values):
        raise PowerControlConfigError(f"{what}: negative GPU index in {list(values)}")
    if len(set(values)) != len(values):
        raise PowerControlConfigError(f"{what}: repeated GPU index in {list(values)}")
    return tuple(sorted(values))


def _reject_extra_keys(keys: set, allowed: set | frozenset, where: str) -> None:
    extra = sorted(str(key) for key in keys - set(allowed))
    if extra:
        raise PowerControlConfigError(
            f"{where}: unknown key(s) {extra}; allowed: {sorted(allowed)}"
        )


def _one_of(value: object, choices: Sequence[str], where: str) -> str:
    text = str(value)
    if text not in choices:
        raise PowerControlConfigError(
            f"{where} must be one of {', '.join(choices)}; got {value!r}"
        )
    return text


def _bool(value: object, where: str) -> bool:
    if isinstance(value, bool):
        return value
    raise PowerControlConfigError(f"{where} must be true or false; got {value!r}")


def _positive_int(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PowerControlConfigError(
            f"{where} must be a positive integer; got {value!r}"
        )
    return value


def _optional_positive_int(value: object, where: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, where)


def _positive_float(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise PowerControlConfigError(f"{where} must be a number > 0; got {value!r}")
    return float(value)


def _fraction(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PowerControlConfigError(
            f"{where} must be a number in (0, 1); got {value!r}"
        )
    if not 0.0 < float(value) < 1.0:
        raise PowerControlConfigError(
            f"{where} must be a fraction in (0, 1); got {value!r}"
        )
    return float(value)


def _optional_str(value: object, where: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise PowerControlConfigError(f"{where} must be a non-empty string or null")
    return value


def _string_list(value: object, where: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and all(isinstance(item, str) for item in value):
        return list(value)
    raise PowerControlConfigError(f"{where} must be a string or a list of strings")


def _unique(items) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for item in items:
        seen.setdefault(item, None)
    return tuple(seen)


__all__ = [
    "APPLIERS",
    "DEFAULT_INTERVAL_S",
    "DEFAULT_NODE_UBPC_BUFFER",
    "DEFAULT_PERF_THRESHOLD",
    "DEFAULT_POLL_S",
    "DEFAULT_STALE_AFTER_S",
    "DEFAULT_WARMUP_POLICY",
    "FOM_PRESETS",
    "FOM_SCOPES",
    "FomSpec",
    "GpuGroup",
    "METRICS_URL_ROLES",
    "MIN_STALE_CYCLES",
    "MODES",
    "MetricsUrl",
    "NodeAgentSettings",
    "PoolLayout",
    "PoolSpec",
    "PowerControlConfig",
    "PowerControlConfigError",
    "STALE_CYCLES",
    "WARMUP_POLICIES",
    "WorkerPlacement",
    "build_pool_layouts",
    "check_stale_window",
    "check_visible_gpus",
    "epd_environment",
    "nupd_budget_binds",
    "parse_fom",
    "parse_gpu_indices",
    "parse_metrics_urls",
    "parse_nodelist",
    "resolve_pools",
    "stale_after_s_for",
    "urls_for_scope",
    "validate_fom",
    "validate_gpu_identity",
]
