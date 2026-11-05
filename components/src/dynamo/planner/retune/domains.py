# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Search-domain expansion for planner retune.

``retune.router`` and ``retune.planner`` use the AISimulate recommendation
syntax of the Dynamo router and planner adapters: public knob names, each a
scalar, ``{choices: [...]}`` or ``{range: {min, max, step?}}``, and the
planner preset groups. ``RouterRetuneConfig`` and ``PlannerRetuneConfig``
parse a section; the helpers here expand it into a finite Cartesian grid
keyed by live names (PATCH ``/router_policy`` parameters and ``PlannerConfig``
fields) plus the load-predictor candidates of the forecast-loss pre-sweep, so
config validation and the retune controller size and enumerate the same grid.

- An omitted AISimulate knob searches its AISimulate default domain; a scalar
  pins it. ``default`` omits every knob.
- Knobs fixed at startup keep their running value; worker minimums and GPU
  budgets are user constraints and are never searched.
- Retune extension knobs (no AISimulate default domain) and the
  ``scaling_policy`` group keep their current value when omitted.
- The load predictor is never a grid dimension: its candidates go to the
  pre-sweep, and every non-baseline candidate uses the winner of its
  throughput interval.
- The planner grid is a list of points, not a full Cartesian product: a
  scaling policy is one correlated choice of its four fields, points with an
  invalid scaling policy are dropped, and fields a point's scaling modes make
  inactive are folded to their running value, so equivalent points appear
  once (``canonical_planner_point``).

The adapter schemas and providers are imported on use: they import
``PlannerConfig``, whose validation imports this module, and the providers
pull in the AISimulate sweeper.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, overload

from pydantic import BaseModel, ValidationError

from dynamo.planner.core.types import RetuneParams

if TYPE_CHECKING:
    from dynamo.planner.config.planner_config import PlannerConfig, RetuneConfig
    from dynamo.planner.simulation.config import PlannerRetuneConfig
    from dynamo.planner.simulation.provider import (
        LoadPredictorSearch,
        LoadSensitivitySearch,
        ScalingPolicySearch,
    )
    from dynamo.router.simulation.config import RouterRetuneConfig

Section = Literal["router", "planner"]
# (label, mapping of every public predictor knob), as the pre-sweep ranks them.
# The sweep request carries the same pairs.
PredictorCandidate = tuple[str, dict[str, Any]]

OVERLAP_SCORE_CREDIT = "overlap_score_credit"
LOAD_SCALING_DOWN_SENSITIVITY = "load_scaling_down_sensitivity"
ENABLE_THROUGHPUT = "enable_throughput_scaling"
ENABLE_LOAD = "enable_load_scaling"
THROUGHPUT_INTERVAL = "throughput_adjustment_interval_seconds"
LOAD_INTERVAL = "load_adjustment_interval_seconds"
MAX_THROUGHPUT_REPLICAS = "max_throughput_scaling_replicas"
# The scaling-policy fields, in the order a grid lists them.
SCALING_POLICY_KEYS: tuple[str, ...] = (
    ENABLE_THROUGHPUT,
    ENABLE_LOAD,
    THROUGHPUT_INTERVAL,
    LOAD_INTERVAL,
)
# The load predictor's fields (``reconfiguration.PREDICTOR_FIELDS``).
PREDICTOR_KEYS: tuple[str, ...] = (
    "load_predictor",
    "load_predictor_log1p",
    "prophet_window_size",
    "kalman_q_level",
    "kalman_q_trend",
    "kalman_r",
    "kalman_min_points",
)
# Fields that act only with load or only with throughput scaling. Without
# their mode a point folds them to the running value. The load interval is
# not among them: it is also the cadence at which live FPM observations
# refit the performance model that throughput scaling uses.
LOAD_ONLY_KEYS: tuple[str, ...] = (LOAD_SCALING_DOWN_SENSITIVITY,)
THROUGHPUT_ONLY_KEYS: tuple[str, ...] = (
    THROUGHPUT_INTERVAL,
    MAX_THROUGHPUT_REPLICAS,
    *PREDICTOR_KEYS,
)
# Combinations a ``preset: false`` group may expand to before deduplication,
# which bounds the expansion's memory. The AISimulate default load-predictor
# domain is 256 combinations.
MAX_PRESET_COMBINATIONS = 65_536


@dataclass(frozen=True)
class PlannerDomain:
    """Expanded planner search domain.

    ``grid`` holds the goodput-grid dimensions (``PlannerConfig`` field ->
    every candidate value), the four scaling-policy fields included when the
    scaling policy is searched. ``points`` are the distinct grid points, each
    with every ``grid`` key: the scaling policies times the other dimensions,
    canonical (``canonical_planner_point``) and valid
    (``scaling_point_error``). Without dimensions there is one empty point,
    the current configuration. ``predictor_candidates`` is set when the load
    predictor is searched (``optimization_target='sla'`` with throughput
    scaling in some point): the forecast-loss pre-sweep picks one of these
    ``(label, mapping)`` candidates per throughput interval and sweep, and it
    is not a grid dimension.
    """

    grid: dict[str, list[Any]] = field(default_factory=dict)
    points: tuple[dict[str, Any], ...] = ({},)
    predictor_candidates: list[PredictorCandidate] | None = None

    @property
    def empty(self) -> bool:
        """True when the domain neither searches fields nor picks a predictor."""

        return not self.grid and not self.predictor_candidates


_ABSENT = object()


def _running_value(running: Mapping[str, Any] | PlannerConfig, key: str) -> Any:
    """``running``'s ``key``; ``_ABSENT`` when a config dump lacks it."""

    if isinstance(running, Mapping):
        return running.get(key, _ABSENT)
    return getattr(running, key)


def _point_value(
    point: Mapping[str, Any], running: Mapping[str, Any] | PlannerConfig, key: str
) -> Any:
    return point[key] if key in point else _running_value(running, key)


def canonical_planner_point(
    point: Mapping[str, Any], running: Mapping[str, Any] | PlannerConfig
) -> dict[str, Any]:
    """``point`` with the fields its scaling modes make inactive folded.

    A field that acts only with load scaling (``LOAD_ONLY_KEYS``) or only
    with throughput scaling (``THROUGHPUT_ONLY_KEYS``) takes its ``running``
    value when the point's own mode (its scaling-policy field, else the
    running one) is off. Only keys already in ``point`` are folded, so the
    key set is kept. ``running`` is the planner config or a config dump; a
    dump without a mode flag or a field leaves it unfolded.
    """

    folded = dict(point)
    for flag, keys in (
        (ENABLE_LOAD, LOAD_ONLY_KEYS),
        (ENABLE_THROUGHPUT, THROUGHPUT_ONLY_KEYS),
    ):
        active = _point_value(point, running, flag)
        if active is _ABSENT or active:
            continue
        for key in keys:
            value = _running_value(running, key)
            if key in folded and value is not _ABSENT:
                folded[key] = value
    return folded


def scaling_point_error(point: Mapping[str, Any], config: PlannerConfig) -> str | None:
    """Why the scaling policy of ``point`` (over ``config``) cannot run.

    Checks only what ``PlannerConfig`` validation would reject and the point
    can be dropped for without validating a full candidate: both modes off,
    a load interval not shorter than the throughput interval with both on,
    and an interval that ``scheduling.scale_interval_seconds`` (fixed at
    startup) does not divide. ``None`` when it can run.
    """

    throughput = bool(_point_value(point, config, ENABLE_THROUGHPUT))
    load = bool(_point_value(point, config, ENABLE_LOAD))
    load_interval = _point_value(point, config, LOAD_INTERVAL)
    throughput_interval = _point_value(point, config, THROUGHPUT_INTERVAL)
    if not throughput and not load:
        return "turns both scaling modes off"
    if throughput and load and load_interval >= throughput_interval:
        return (
            f"has load_adjustment_interval_seconds {load_interval} not shorter "
            f"than throughput_adjustment_interval_seconds {throughput_interval}"
        )
    # Imported on use: planner_config imports this module on use too.
    from dynamo.planner.config.planner_config import _is_multiple_of

    scale = config.scheduling.scale_interval_seconds
    intervals = [(LOAD_INTERVAL, load_interval)]
    if throughput:
        intervals.append((THROUGHPUT_INTERVAL, throughput_interval))
    for name, interval in intervals:
        if not _is_multiple_of(float(interval), scale):
            return (
                f"has {name} {interval}, not a multiple of the planner's "
                f"scheduling.scale_interval_seconds {scale:g} (fixed at startup)"
            )
    return None


@overload
def parse_retune_section(
    section: Literal["router"], spec: str | Mapping[str, Any]
) -> RouterRetuneConfig:
    ...


@overload
def parse_retune_section(
    section: Literal["planner"], spec: str | Mapping[str, Any]
) -> PlannerRetuneConfig:
    ...


def parse_retune_section(
    section: Section, spec: str | Mapping[str, Any]
) -> RouterRetuneConfig | PlannerRetuneConfig:
    """Validate one retune section against its adapter schema.

    ``"default"`` parses as ``{}`` (every knob omitted). Raises ``ValueError``
    with pydantic's errors, one ``<location>: <message>`` each, and a hint for
    a preset group written as a bare list.
    """

    schema: type[BaseModel]
    if section == "router":
        from dynamo.router.simulation.config import RouterRetuneConfig

        schema = RouterRetuneConfig
    else:
        from dynamo.planner.simulation.config import (
            PLANNER_RETUNE_GROUPS,
            PlannerRetuneConfig,
        )

        schema = PlannerRetuneConfig
    name = f"retune.{section}"
    raw: Any = {} if spec == "default" else spec
    if not isinstance(raw, Mapping):
        raise ValueError(f"{name} must be 'default' or a mapping, got {raw!r}")
    if section == "planner":
        for group in PLANNER_RETUNE_GROUPS:
            if isinstance(raw.get(group), list):
                raise ValueError(
                    f"{name}.{group} is a bare list; {_preset_group_hint(group)}"
                )
    try:
        return schema.model_validate(raw)
    except ValidationError as error:
        raise ValueError(_schema_error(name, error)) from None


def _preset_group_hint(group: str) -> str:
    if group == "load_predictor":
        from dynamo.planner.simulation.load_predictor import LOAD_PREDICTOR_PRESETS

        presets = list(LOAD_PREDICTOR_PRESETS)
        off = "{preset: false, type: {choices: [...]}} to search predictor types"
    else:
        from dynamo.planner.simulation.presets import LOAD_SENSITIVITY

        presets = list(LOAD_SENSITIVITY)
        off = (
            "{preset: false} with load_scaling_down_sensitivity: {choices: [...]} "
            "next to it"
        )
    return (
        f"{group} is a preset group: write {{preset: [...]}} listing preset names "
        f"({', '.join(presets)}) or complete mappings, or {off}"
    )


def _schema_error(name: str, error: ValidationError) -> str:
    """Pydantic's errors as ``<section>.<location>: <message>``.

    A knob's domain union is tagged by the form the value is written in, so
    pydantic reports only that form's errors. A validator's message that
    already names the section is used as is.
    """

    messages: list[str] = []
    for item in error.errors(include_url=False):
        context = item.get("ctx") or {}
        message = (
            str(context["error"])
            if item["type"] == "value_error" and "error" in context
            else item["msg"]
        )
        if item["loc"]:
            message = f"{'.'.join([name, *map(str, item['loc'])])}: {message}"
        elif not message.startswith(name):
            message = f"{name}: {message}"
        messages.append(message)
    return "; ".join(messages)


def expand_router_domain(
    retune: RetuneConfig,
    *,
    startup_load_model: str | None = None,
    zero_startup_credit: bool = False,
) -> dict[str, list[float]]:
    """Return the router search domain keyed by PATCH ``/router_policy`` names.

    An omitted AISimulate knob (``overlap_score_credit``,
    ``prefill_load_scale``, ``temperature``) takes its AISimulate default
    domain; ``router: default`` is all three (3 x 9 x 4 = 108 points). An
    extension knob is a dimension only when written. Each dimension may hold
    at most ``retune.max_candidates`` values.

    ``startup_load_model`` and ``zero_startup_credit`` need the startup
    ``KvRouterConfig``, so the retune controller passes them:
    a written ``prefill_load_model.type`` must equal the startup model (the
    serde form ``"aic"`` reads as ``"ais"``), and a zero startup credit
    collapses an omitted ``overlap_score_credit`` to ``[0.0]``, since a live
    router rejects a positive credit then. A written positive credit is left
    for the caller to reject per candidate.
    """

    spec = retune.router
    if spec is None:
        return {}
    from dynamo.aisimulate.domains import finite_values
    from dynamo.router.simulation.config import (
        ROUTER_KNOB_DEFAULTS,
        ROUTER_PUBLIC_TO_LIVE,
        ROUTER_RETUNE_EXTENSIONS,
    )

    public = parse_retune_section("router", spec)
    if startup_load_model is not None:
        _check_startup_load_model(public, startup_load_model)
    limit = retune.max_candidates
    domain: dict[str, list[float]] = {}
    for name, defaults in ROUTER_KNOB_DEFAULTS.items():
        value = getattr(public, name)
        live = ROUTER_PUBLIC_TO_LIVE[name]
        if name == OVERLAP_SCORE_CREDIT and value is None and zero_startup_credit:
            domain[live] = [0.0]
            continue
        values = finite_values(f"retune.router.{name}", value, defaults, limit=limit)
        domain[live] = [float(item) for item in values]
    for name in ROUTER_RETUNE_EXTENSIONS:
        if name in public.model_fields_set:
            values = finite_values(
                f"retune.router.{name}", getattr(public, name), None, limit=limit
            )
            domain[name] = [float(item) for item in values]
    return domain


def _check_startup_load_model(public: RouterRetuneConfig, startup: str) -> None:
    from aisimulate.config.common import Choices

    model = public.prefill_load_model
    if (
        "prefill_load_model" not in public.model_fields_set
        or "type" not in model.model_fields_set
    ):
        return
    running = "ais" if startup == "aic" else startup
    written = (
        list(model.type.choices) if isinstance(model.type, Choices) else [model.type]
    )
    if written != [running]:
        raise ValueError(
            "retune.router.prefill_load_model.type is fixed at startup, and "
            f"retune keeps the running {running!r}; got {written}"
        )


def expand_planner_domain(config: PlannerConfig) -> PlannerDomain:
    """Return the planner search domain of ``config.retune``.

    Only what a running planner can change and what applies to ``config`` is
    searched:

    - ``scaling_policy``, when written: its preset entries (``default`` is
      every AISimulate preset that scales; ``preset: false`` every
      combination of its four knobs). With ``optimization_target`` other
      than ``sla`` the planner always scales on load only, so only the load
      interval can vary. Points that turn both modes off, order the
      intervals wrongly or use an interval the scale interval does not
      divide are dropped; an entry written in a preset list must be valid.
    - ``load_sensitivity``, with load scaling in some point: the
      ``load_scaling_down_sensitivity`` of its entries
      (``load_min_observations`` is fixed at startup). Omitted, the three
      AISimulate presets give {70, 80, 90}.
    - ``load_predictor``, with ``optimization_target='sla'`` and throughput
      scaling in some point: the distinct predictor candidates of the
      pre-sweep. Omitted, the 11 AISimulate presets.
    - Any written ``PLANNER_RETUNE_KNOBS`` knob, under its ``PlannerConfig``
      name.

    An omitted group that does not apply is skipped; a written one raises, as
    does any written knob without load or throughput scaling. Grid values
    must affect some point and pass strict field validation
    (``PlannerConfig._validate_retune_planner_grid``). Points are made
    canonical and deduplicated; the full ``PlannerConfig`` validation of each
    point, which can drop more, is the retune controller's
    (``RetuneController._feasible_grid``).
    """

    retune = config.retune
    if retune is None:
        raise ValueError("planner config has no retune section")
    spec = retune.planner
    if spec is None:
        return PlannerDomain()
    from dynamo.aisimulate.domains import finite_values, require_enumerable
    from dynamo.planner.simulation.config import PLANNER_RETUNE_KNOBS

    public = parse_retune_section("planner", spec)
    explicit = public.model_fields_set
    if explicit and not config.scaling_enabled():
        raise ValueError(
            "retune.planner requires load or throughput scaling; "
            "retune-only mode searches router parameters only"
        )
    # Before any preset expansion: it would read a log integer range as every
    # integer in it and a log float range as its two endpoints.
    require_enumerable(public, prefix="retune.planner.")
    limit = retune.max_candidates

    policies = _scaling_policies(config, public, limit)
    modes = (
        [(config.enable_throughput_scaling, config.enable_load_scaling)]
        if policies is None
        else [(p[ENABLE_THROUGHPUT], p[ENABLE_LOAD]) for p in policies]
    )
    any_throughput = any(throughput for throughput, _ in modes)
    any_load = any(load for _, load in modes)

    grid: dict[str, list[Any]] = {}
    public_names: dict[str, str] = {}
    if policies is not None:
        for key in SCALING_POLICY_KEYS:
            grid[key] = list(dict.fromkeys(policy[key] for policy in policies))
    sensitivities = _load_sensitivities(config, public, limit, any_load)
    if sensitivities is not None:
        grid[LOAD_SCALING_DOWN_SENSITIVITY] = sensitivities
    for name, live in PLANNER_RETUNE_KNOBS.items():
        if name in explicit:
            grid[live] = finite_values(
                f"retune.planner.{name}", getattr(public, name), None, limit=limit
            )
            public_names[live] = name
    grid = config._validate_retune_planner_grid(
        grid,
        public_names,
        load_scaling=any_load,
        throughput_scaling=any_throughput,
    )
    if policies is not None:
        # The validated values (strict field types) back the policies too.
        policies = [
            {key: _validated(grid[key], policy[key]) for key in SCALING_POLICY_KEYS}
            for policy in policies
        ]
    return PlannerDomain(
        grid=grid,
        points=_planner_points(config, grid, policies),
        predictor_candidates=_predictor_candidates(
            config, public, limit, any_throughput
        ),
    )


def _validated(values: Sequence[Any], value: Any) -> Any:
    """The validated grid value that equals ``value`` (``5`` -> ``5``)."""

    for candidate in values:
        if (isinstance(candidate, bool), candidate) == (isinstance(value, bool), value):
            return candidate
    raise AssertionError(f"{value!r} is not a grid value")  # pragma: no cover


def _planner_points(
    config: PlannerConfig,
    grid: Mapping[str, Sequence[Any]],
    policies: Sequence[Mapping[str, Any]] | None,
) -> tuple[dict[str, Any], ...]:
    """Every distinct valid canonical point, in canonical key order."""

    independent = {
        key: values for key, values in grid.items() if key not in SCALING_POLICY_KEYS
    }
    combinations = math.prod(len(values) for values in independent.values()) * (
        len(policies) if policies is not None else 1
    )
    if combinations > MAX_PRESET_COMBINATIONS:
        raise ValueError(
            f"retune.planner expands to {combinations} points before "
            f"deduplication, more than {MAX_PRESET_COMBINATIONS}; narrow its "
            "knob domains"
        )
    points: dict[str, dict[str, Any]] = {}
    dropped: str | None = None
    for policy in policies if policies is not None else [{}]:
        for values in itertools.product(*independent.values()):
            point = canonical_planner_point(
                {**dict(zip(independent, values)), **policy}, config
            )
            error = scaling_point_error(point, config)
            if error is not None:
                dropped = dropped or f"{point} {error}"
                continue
            points.setdefault(RetuneParams(planner=point).key(), point)
    if not points:
        raise ValueError(
            "retune.planner has no valid point: every scaling policy is "
            f"invalid here, for example {dropped}"
        )
    return tuple(points[key] for key in sorted(points))


def _scaling_policies(
    config: PlannerConfig, public: PlannerRetuneConfig, limit: int
) -> list[dict[str, Any]] | None:
    """The searched scaling policies, each a mapping of its four fields.

    ``None`` when ``scaling_policy`` is omitted: the running policy is kept.
    A preset's null interval is the running value, except for an enabled
    mode, which raises. Policies a planner cannot run are dropped from a
    default or ``preset: false`` expansion (``scaling_point_error`` reports
    them again for a written entry). Without ``optimization_target='sla'``
    every policy must scale on load only.
    """

    control = public.scaling_policy
    if control is None:
        return None
    from dynamo.planner.simulation.presets import SCALING_POLICIES, scaling_fields
    from dynamo.planner.simulation.provider import ScalingPolicySearch

    sla = config.optimization_target == "sla"
    written: list[str | dict[str, Any]] | None = None
    entries: list[str | dict[str, Any]]
    if control.preset == "default":
        # Every AISimulate preset that scales: retune never turns scaling off.
        entries = [
            name
            for name, policy in SCALING_POLICIES.items()
            if policy.enable_throughput or policy.enable_load
        ]
    elif control.preset in (False, {}):
        if not sla:
            knobs = sorted(
                {ENABLE_THROUGHPUT, ENABLE_LOAD, THROUGHPUT_INTERVAL}
                & public.model_fields_set
            )
            if knobs:
                raise ValueError(
                    f"retune.planner {knobs} have no effect here; they require "
                    "optimization_target='sla' (other targets always scale on "
                    "load only)"
                )
        entries = list(_preset_off_entries("scaling_policy", public, limit))
    else:
        written = _preset_entries(ScalingPolicySearch, "scaling_policy", control)
        entries = list(written)

    policies: list[dict[str, Any]] = []
    for entry in entries:
        fields = scaling_fields(entry)
        for key, enabled in (
            (THROUGHPUT_INTERVAL, fields[ENABLE_THROUGHPUT]),
            (LOAD_INTERVAL, fields[ENABLE_LOAD]),
        ):
            if fields[key] is None:
                if enabled and written is not None:
                    raise ValueError(
                        f"retune.planner.scaling_policy entry {entry!r}: {key} "
                        "cannot be null while its scaling mode is enabled"
                    )
                fields[key] = getattr(config, key)
        policy = {key: fields[key] for key in SCALING_POLICY_KEYS}
        load_only = policy[ENABLE_LOAD] and not policy[ENABLE_THROUGHPUT]
        if not sla and not load_only:
            if written is not None:
                raise ValueError(
                    f"retune.planner.scaling_policy entry {entry!r} has no effect "
                    "here; with optimization_target "
                    f"{config.optimization_target!r} the planner always scales "
                    "on load only, so only load-only entries apply"
                )
            continue
        if written is not None:
            error = scaling_point_error(policy, config)
            if error is not None:
                hint = (
                    ""
                    if policy[ENABLE_THROUGHPUT] or policy[ENABLE_LOAD]
                    else "; retune never turns scaling off"
                )
                raise ValueError(
                    f"retune.planner.scaling_policy entry {entry!r} {error}{hint}"
                )
        # An interval of a mode the policy turns off is inert: fold it so
        # policies that differ only there are one choice.
        policies.append(canonical_planner_point(policy, config))
    if not policies:
        raise ValueError(
            "retune.planner.scaling_policy has no policy that applies here "
            f"(optimization_target {config.optimization_target!r})"
        )
    unique = {RetuneParams(planner=policy).key(): policy for policy in policies}
    if len(unique) > limit:
        raise ValueError(
            f"retune.planner.scaling_policy has {len(unique)} distinct policies, "
            f"exceeding retune.max_candidates={limit}"
        )
    return list(unique.values())


def _load_sensitivities(
    config: PlannerConfig, public: PlannerRetuneConfig, limit: int, load: bool
) -> list[Any] | None:
    explicit = "load_sensitivity" in public.model_fields_set
    if not load:
        if explicit:
            raise ValueError(
                "retune.planner.load_sensitivity has no effect here; it requires "
                "enable_load_scaling"
            )
        return None
    from dynamo.planner.simulation.presets import (
        LOAD_SENSITIVITY,
        load_sensitivity_fields,
    )
    from dynamo.planner.simulation.provider import LoadSensitivitySearch

    control = public.load_sensitivity
    running = config.load_min_observations
    entries: list[str | dict[str, Any]]
    if control is None or control.preset == "default":
        entries = list(LOAD_SENSITIVITY)
    elif control.preset in (False, {}):
        entries = list(_preset_off_entries("load_sensitivity", public, limit))
    else:
        entries = _preset_entries(LoadSensitivitySearch, "load_sensitivity", control)
        for entry in entries:
            if isinstance(entry, dict) and entry["load_min_observations"] != running:
                raise ValueError(
                    f"retune.planner.load_sensitivity.preset entry {entry} sets "
                    "load_min_observations, which is fixed at startup; it must "
                    f"equal the running value {running}"
                )
    # Only the runtime part of each entry is searched: the inner planner keeps
    # the running load_min_observations, which is fixed at startup.
    return list(
        dict.fromkeys(
            load_sensitivity_fields(entry)[LOAD_SCALING_DOWN_SENSITIVITY]
            for entry in entries
        )
    )


def _predictor_candidates(
    config: PlannerConfig, public: PlannerRetuneConfig, limit: int, throughput: bool
) -> list[PredictorCandidate] | None:
    explicit = "load_predictor" in public.model_fields_set
    if not (config.optimization_target == "sla" and throughput):
        if explicit:
            raise ValueError(
                "retune.planner.load_predictor has no effect here; it requires "
                "optimization_target='sla' with throughput scaling"
            )
        return None
    from dynamo.planner.simulation.load_predictor import (
        LOAD_PREDICTOR_PRESETS,
        distinct_predictor_candidates,
    )
    from dynamo.planner.simulation.provider import LoadPredictorSearch

    control = public.load_predictor
    entries: list[str | dict[str, Any]]
    if control is None or control.preset == "default":
        entries = list(LOAD_PREDICTOR_PRESETS)
    elif control.preset in (False, {}):
        entries = list(_preset_off_entries("load_predictor", public, limit))
    else:
        entries = _preset_entries(LoadPredictorSearch, "load_predictor", control)
    candidates = distinct_predictor_candidates(entries)
    if len(candidates) > limit:
        raise ValueError(
            f"retune.planner.load_predictor has {len(candidates)} distinct "
            f"predictor candidates, exceeding retune.max_candidates={limit}"
        )
    return candidates


def _preset_entries(
    search: type[ScalingPolicySearch]
    | type[LoadSensitivitySearch]
    | type[LoadPredictorSearch],
    group: str,
    control: Any,
) -> list[str | dict[str, Any]]:
    """Validate a preset list (names, or complete mappings) of ``group``.

    A mapping's values come back as their field types (``kalman_r: 1`` is
    ``1.0``), as in the named presets and ``PlannerConfig``: a candidate equal
    to the running predictor then has the running config's
    ``RetuneParams.key``.
    """

    from dynamo.planner.simulation.config import (
        LoadPredictorMapping,
        LoadSensitivityMapping,
        ScalingPolicyMapping,
    )

    mapping_models: dict[str, type[BaseModel]] = {
        "scaling_policy": ScalingPolicyMapping,
        "load_sensitivity": LoadSensitivityMapping,
        "load_predictor": LoadPredictorMapping,
    }
    mapping_model = mapping_models[group]
    try:
        entries = search.model_validate({"preset": control.preset}).preset
        return [
            mapping_model.model_validate(entry).model_dump()
            if isinstance(entry, dict)
            else entry
            for entry in entries
        ]
    except ValidationError as error:
        message = "; ".join(
            str(item["msg"]).removeprefix("Value error, ")
            for item in error.errors(include_url=False)
        )
        raise ValueError(f"retune.planner.{group}.preset: {message}") from None


def _preset_off_entries(
    group: str,
    public: PlannerRetuneConfig,
    limit: int,
) -> list[dict[str, Any]]:
    """Every combination ``preset: false`` of ``group`` stands for.

    Like AISimulate recommend: a written group knob uses its domain, an
    omitted one its AISimulate default domain. The written domains and the
    number of combinations are bounded before anything is built.
    """

    from dynamo.aisimulate.domains import check_domain_size
    from dynamo.planner.simulation.config import PLANNER_PRESET_GROUPS
    from dynamo.planner.simulation.provider import independent_preset_dimensions

    for knob in sorted(PLANNER_PRESET_GROUPS[group] & public.model_fields_set):
        value = getattr(public, knob)
        if value is not None:
            check_domain_size(f"retune.planner.{knob}", value, None, limit=limit)
    control = getattr(public, group).model_dump(mode="python", exclude_none=True)
    names, dimensions = independent_preset_dimensions(
        group, control, public.model_dump(mode="python", exclude_none=True)
    )
    combinations = math.prod(len(values) for values in dimensions)
    if combinations > MAX_PRESET_COMBINATIONS:
        raise ValueError(
            f"retune.planner.{group} with preset: false expands to {combinations} "
            f"combinations, more than {MAX_PRESET_COMBINATIONS}; narrow its knob "
            "domains"
        )
    return [
        dict(zip(names, combination, strict=True))
        for combination in itertools.product(*dimensions)
    ]


def describe_grid(
    router: Mapping[str, Sequence[Any]],
    planner: Mapping[str, Sequence[Any]],
    planner_points: int | None = None,
) -> str:
    """Each grid dimension's size under its public name, for the error of a
    grid larger than ``retune.max_candidates``. ``planner_points`` (the
    number of distinct planner points) is reported when it differs from the
    product of the planner dimensions."""

    from dynamo.planner.simulation.config import PLANNER_RETUNE_KNOBS
    from dynamo.router.simulation.config import ROUTER_PUBLIC_TO_LIVE

    parts = []
    for section, domain, names in (
        ("router", router, ROUTER_PUBLIC_TO_LIVE),
        ("planner", planner, PLANNER_RETUNE_KNOBS),
    ):
        public = {live: name for name, live in names.items()}
        if domain:
            sizes = " x ".join(
                f"{public.get(key, key)} {len(values)}"
                for key, values in domain.items()
            )
            if (
                section == "planner"
                and planner_points is not None
                and planner_points != grid_size(planner)
            ):
                sizes += f" = {planner_points} distinct points"
            parts.append(f"{section} {sizes}")
    return "; ".join(parts)


def grid_size(*domains: Mapping[str, Sequence[Any]]) -> int:
    """Number of Cartesian grid points over every dimension of ``domains``.

    A grid without dimensions has one point: the current configuration.
    """

    return math.prod(len(values) for domain in domains for values in domain.values())


def grid_points(
    router: Mapping[str, Sequence[float]],
    planner_points: Sequence[Mapping[str, Any]] = ({},),
) -> list[RetuneParams]:
    """The router Cartesian grid times ``planner_points``, in canonical
    ``RetuneParams.key`` order."""

    router_keys = sorted(router)
    points = [
        RetuneParams(
            router=dict(zip(router_keys, router_values)), planner=dict(planner)
        )
        for router_values in itertools.product(*(router[k] for k in router_keys))
        for planner in planner_points
    ]
    return sorted(points, key=RetuneParams.key)


def throughput_intervals(
    config: PlannerConfig, points: Sequence[Mapping[str, Any]]
) -> list[int]:
    """The throughput intervals of the points that scale on throughput, and
    of ``config`` when it does."""

    intervals = set()
    if config.enable_throughput_scaling:
        intervals.add(int(config.throughput_adjustment_interval_seconds))
    for point in points:
        if _point_value(point, config, ENABLE_THROUGHPUT):
            intervals.add(int(_point_value(point, config, THROUGHPUT_INTERVAL)))
    return sorted(intervals)


def required_history_seconds(config: PlannerConfig, domain: PlannerDomain) -> float:
    """Trace history needed to warm every predictor the sweep may run.

    The largest throughput interval (``throughput_intervals``) x the largest
    ``prophet_window_size`` or ``kalman_min_points`` over the current config
    and the pre-sweep candidates' predictor fields; 0 when no point scales on
    throughput.
    """

    intervals = throughput_intervals(config, domain.points)
    if not intervals:
        return 0.0
    window_sizes = [config.prophet_window_size]
    min_points = [config.kalman_min_points]
    if domain.predictor_candidates:
        # Imported on use: the module imports PlannerConfig at module scope.
        from dynamo.planner.simulation.load_predictor import predictor_fields

        for _, entry in domain.predictor_candidates:
            fields = predictor_fields(entry)
            window_sizes.append(
                fields.get("prophet_window_size", config.prophet_window_size)
            )
            min_points.append(fields.get("kalman_min_points", config.kalman_min_points))
    return float(max(intervals) * max(*window_sizes, *min_points))


def retune_history_seconds(config: PlannerConfig) -> float:
    """History each sweep reads before its window: ``retune.history_seconds``,
    else ``required_history_seconds`` (0 without throughput scaling, or with
    fixed replicas, whose sweeps run no inner planner)."""

    retune = config.retune
    if retune is None or not config.scaling_enabled():
        return 0.0
    if retune.history_seconds is not None:
        return float(retune.history_seconds)
    return required_history_seconds(config, expand_planner_domain(config))
