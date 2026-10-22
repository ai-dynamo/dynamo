# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Search-domain expansion for planner retune.

``retune.router`` and ``retune.planner`` are either explicit Cartesian domains
or the literal ``"default"``. These helpers turn both forms into concrete
domains keyed by live wire names, so config validation and the retune
controller size and enumerate the same grid.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from dynamo.planner.core.types import RetuneParams

if TYPE_CHECKING:
    from dynamo.planner.config.planner_config import PlannerConfig, RetuneConfig

OVERLAP_SCORE_CREDIT = "overlap_score_credit"
PREFILL_LOAD_SCALE = "prefill_load_scale"
ROUTER_TEMPERATURE = "router_temperature"
LOAD_SCALING_DOWN_SENSITIVITY = "load_scaling_down_sensitivity"


@dataclass(frozen=True)
class PlannerDomain:
    """Expanded planner search domain.

    ``grid`` holds the goodput-grid dimensions (``PlannerConfig`` field ->
    candidate values). ``predictor_presets`` is set only by
    ``planner: default`` on an SLA planner with throughput scaling: the
    predictor is then chosen per sweep by the forecast-loss pre-sweep over
    these ``LOAD_PREDICTOR_PRESETS`` names and is not a grid dimension.
    """

    grid: dict[str, list[Any]] = field(default_factory=dict)
    predictor_presets: list[str] | None = None

    @property
    def empty(self) -> bool:
        """True when the domain neither searches fields nor picks a predictor."""

        return not self.grid and not self.predictor_presets


def expand_router_domain(
    retune: RetuneConfig, *, zero_startup_credit: bool = False
) -> dict[str, list[float]]:
    """Return the router search domain keyed by PATCH ``/router_policy`` names.

    ``router: default`` is the Dynamo router simulation default domain:
    ``overlap_score_credit`` x ``prefill_load_scale`` x ``router_temperature``.

    ``zero_startup_credit`` is the hook for the E3 credit rule. Resolving the
    startup credit needs the ``KvRouterConfig`` binding, so the caller (the
    retune controller) decides it. When true, the default domain collapses
    ``overlap_score_credit`` to ``[0.0]``. An explicit domain is returned
    unchanged; the caller rejects it instead if it lists a positive credit.
    """

    if retune.router != "default":
        return {key: list(values) for key, values in retune.router.items()}
    # Imported on use: the router simulation package loads its sweep provider,
    # which config validation (the caller) must not import eagerly.
    from dynamo.router.simulation.config import (
        OVERLAP_SCORE_CREDIT_DEFAULTS,
        PREFILL_LOAD_SCALE_DEFAULTS,
        TEMPERATURE_DEFAULTS,
    )

    return {
        OVERLAP_SCORE_CREDIT: (
            [0.0] if zero_startup_credit else list(OVERLAP_SCORE_CREDIT_DEFAULTS)
        ),
        PREFILL_LOAD_SCALE: list(PREFILL_LOAD_SCALE_DEFAULTS),
        ROUTER_TEMPERATURE: list(TEMPERATURE_DEFAULTS),
    }


def expand_planner_domain(config: PlannerConfig) -> PlannerDomain:
    """Return the planner search domain of ``config.retune``.

    ``planner: default`` keeps only the runtime-changeable part that applies to
    ``config``: ``load_scaling_down_sensitivity`` from the ``LOAD_SENSITIVITY``
    presets when load scaling is enabled, and the predictor pre-sweep presets
    when ``optimization_target == 'sla'`` with throughput scaling. It expands to
    an empty domain otherwise. An explicit domain becomes the grid as is,
    including any explicitly listed predictor fields.
    """

    retune = config.retune
    if retune is None:
        raise ValueError("planner config has no retune section")
    if retune.planner != "default":
        return PlannerDomain(
            grid={key: list(values) for key, values in retune.planner.items()}
        )

    grid: dict[str, list[Any]] = {}
    predictor_presets: list[str] | None = None
    # Imported on use: both simulation modules import PlannerConfig at module
    # scope, and config validation calls this function.
    if config.enable_load_scaling:
        from dynamo.planner.simulation.presets import LOAD_SENSITIVITY

        grid[LOAD_SCALING_DOWN_SENSITIVITY] = list(
            dict.fromkeys(
                preset[LOAD_SCALING_DOWN_SENSITIVITY]
                for preset in LOAD_SENSITIVITY.values()
            )
        )
    if config.optimization_target == "sla" and config.enable_throughput_scaling:
        from dynamo.planner.simulation.load_predictor import LOAD_PREDICTOR_PRESETS

        predictor_presets = list(LOAD_PREDICTOR_PRESETS)
    return PlannerDomain(grid=grid, predictor_presets=predictor_presets)


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
    ``prophet_window_size`` or ``kalman_min_points`` over the current config,
    the grid values and the pre-sweep presets' fields.
    """

    window_sizes = [
        config.prophet_window_size,
        *domain.grid.get("prophet_window_size", ()),
    ]
    min_points = [config.kalman_min_points, *domain.grid.get("kalman_min_points", ())]
    if domain.predictor_presets:
        # Imported on use: the module imports PlannerConfig at module scope.
        from dynamo.planner.simulation.load_predictor import predictor_fields

        for preset in domain.predictor_presets:
            fields = predictor_fields(preset)
            window_sizes.append(
                fields.get("prophet_window_size", config.prophet_window_size)
            )
            min_points.append(fields.get("kalman_min_points", config.kalman_min_points))
    return float(
        config.throughput_adjustment_interval_seconds * max(*window_sizes, *min_points)
    )
