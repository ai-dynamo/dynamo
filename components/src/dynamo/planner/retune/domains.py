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
- Knobs fixed at startup keep their running value.
- Retune extension knobs (no AISimulate default domain) keep their current
  value when omitted.
- The load predictor is never a grid dimension: its candidates go to the
  pre-sweep, and every non-baseline candidate uses the winner.

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
    )
    from dynamo.router.simulation.config import RouterRetuneConfig

Section = Literal["router", "planner"]
# (label, mapping of every public predictor knob), as the pre-sweep ranks them.
# The sweep request carries the same pairs.
PredictorCandidate = tuple[str, dict[str, Any]]

OVERLAP_SCORE_CREDIT = "overlap_score_credit"
LOAD_SCALING_DOWN_SENSITIVITY = "load_scaling_down_sensitivity"
# Combinations a ``preset: false`` group may expand to before deduplication,
# which bounds the expansion's memory. The AISimulate default load-predictor
# domain is 256 combinations.
MAX_PRESET_COMBINATIONS = 65_536


@dataclass(frozen=True)
class PlannerDomain:
    """Expanded planner search domain.

    ``grid`` holds the goodput-grid dimensions (``PlannerConfig`` field ->
    candidate values). ``predictor_candidates`` is set when the load
    predictor is searched (``optimization_target='sla'`` with throughput
    scaling): the forecast-loss pre-sweep picks one of these ``(label,
    mapping)`` candidates per sweep, and it is not a grid dimension.
    """

    grid: dict[str, list[Any]] = field(default_factory=dict)
    predictor_candidates: list[PredictorCandidate] | None = None

    @property
    def empty(self) -> bool:
        """True when the domain neither searches fields nor picks a predictor."""

        return not self.grid and not self.predictor_candidates


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

    - ``load_sensitivity``, with load scaling: the ``load_scaling_down_sensitivity``
      of its entries (``load_min_observations`` is fixed at startup). Omitted,
      the three AISimulate presets give {70, 80, 90}.
    - ``load_predictor``, with ``optimization_target='sla'`` and throughput
      scaling: the distinct predictor candidates of the pre-sweep. Omitted,
      the 11 AISimulate presets.
    - Any written ``PLANNER_RETUNE_KNOBS`` knob, under its ``PlannerConfig``
      name.

    An omitted group that does not apply is skipped; a written one raises, as
    does any written knob without load or throughput scaling. Grid values
    must affect this configuration and pass strict field validation
    (``PlannerConfig._validate_retune_planner_grid``).
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

    grid: dict[str, list[Any]] = {}
    public_names: dict[str, str] = {}
    sensitivities = _load_sensitivities(config, public, limit)
    if sensitivities is not None:
        grid[LOAD_SCALING_DOWN_SENSITIVITY] = sensitivities
    for name, live in PLANNER_RETUNE_KNOBS.items():
        if name in explicit:
            grid[live] = finite_values(
                f"retune.planner.{name}", getattr(public, name), None, limit=limit
            )
            public_names[live] = name
    grid = config._validate_retune_planner_grid(grid, public_names)
    return PlannerDomain(
        grid=grid, predictor_candidates=_predictor_candidates(config, public, limit)
    )


def _load_sensitivities(
    config: PlannerConfig, public: PlannerRetuneConfig, limit: int
) -> list[Any] | None:
    explicit = "load_sensitivity" in public.model_fields_set
    if not config.enable_load_scaling:
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
    config: PlannerConfig, public: PlannerRetuneConfig, limit: int
) -> list[PredictorCandidate] | None:
    explicit = "load_predictor" in public.model_fields_set
    if not (config.optimization_target == "sla" and config.enable_throughput_scaling):
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
    search: type[LoadSensitivitySearch] | type[LoadPredictorSearch],
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
    )

    mapping_model: type[BaseModel] = (
        LoadPredictorMapping if group == "load_predictor" else LoadSensitivityMapping
    )
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
    router: Mapping[str, Sequence[Any]], planner: Mapping[str, Sequence[Any]]
) -> str:
    """Each grid dimension's size under its public name, for the error of a
    grid larger than ``retune.max_candidates``."""

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
            parts.append(f"{section} {sizes}")
    return "; ".join(parts)


def grid_size(*domains: Mapping[str, Sequence[Any]]) -> int:
    """Number of Cartesian grid points over every dimension of ``domains``.

    A grid without dimensions has one point: the current configuration.
    """

    return math.prod(len(values) for domain in domains for values in domain.values())


def grid_points(
    router: Mapping[str, Sequence[float]], planner: Mapping[str, Sequence[Any]]
) -> list[RetuneParams]:
    """Enumerate the Cartesian grid in canonical ``RetuneParams.key`` order."""

    router_keys = sorted(router)
    planner_keys = sorted(planner)
    points = [
        RetuneParams(
            router=dict(zip(router_keys, router_values)),
            planner=dict(zip(planner_keys, planner_values)),
        )
        for router_values in itertools.product(*(router[k] for k in router_keys))
        for planner_values in itertools.product(*(planner[k] for k in planner_keys))
    ]
    return sorted(points, key=RetuneParams.key)


def required_history_seconds(config: PlannerConfig, domain: PlannerDomain) -> float:
    """Trace history needed to warm every predictor the sweep may run.

    ``throughput_adjustment_interval_seconds`` x the largest
    ``prophet_window_size`` or ``kalman_min_points`` over the current config
    and the pre-sweep candidates' predictor fields.
    """

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
    return float(
        config.throughput_adjustment_interval_seconds * max(*window_sizes, *min_points)
    )
