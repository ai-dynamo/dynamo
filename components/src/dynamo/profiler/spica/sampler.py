# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vizier-backed sampler over a :class:`dynamo.profiler.spica.search_space.BranchSpace`.

One Vizier study per branch (per the design). The study's parameters are:

- structured parallel features projected onto the branch's KV-feasible config
  pool.
- a continuous ``kv_load_ratio`` when a Pareto workload supplies a range.
- one parameter per multi-choice searchable knob (categorical for string choices
  like ``planner_scaling_policy``/``router_mode``; discrete for the numeric
  batching / router-weight choices). Single-choice knobs are injected as
  constants (not Vizier params).

``suggest`` decodes each trial into a ``selection`` dict (the shape
:func:`dynamo.profiler.spica.sample.unroll_sample` consumes) plus the chosen parallel-config
object; ``observe`` reports the (higher-is-better) score back to Vizier.

The sampler is swappable behind the :class:`BranchSampler` Protocol so a lighter
backend can replace Vizier without touching the orchestration. The branch-space
builder removes dependent router/planner knobs when those components are pinned
off; mixed-mode studies retain them because Vizier has no conditional params here.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from .parallel_enum import DisaggParallelConfig, ReplicaParallelConfig
from .parallel_projection import ParallelConfigProjector, ParallelProjection
from .search_space import BranchSpace

_CONSTANT_PARAM = "_spica_constant"
_METRIC = "objective"


@dataclass
class Suggestion:
    """One sampled candidate: the unroll selection + chosen parallel config,
    plus an opaque handle the sampler uses to report the score."""

    selection: dict[str, Any]
    parallel_config: ReplicaParallelConfig | DisaggParallelConfig
    handle: Any = field(repr=False)
    projection: ParallelProjection | None = field(default=None, repr=False)


class BranchSampler(Protocol):
    """Stateful optimizer over one branch (swappable: Vizier, random, ...)."""

    branch: BranchSpace

    def suggest(self, count: int) -> list[Suggestion]:
        ...

    def observe(self, suggestion: Suggestion, metrics: dict[str, float]) -> None:
        ...

    def observe_infeasible(self, suggestion: Suggestion, reason: str) -> None:
        ...


def _decoder_for(choices: list[Any]) -> Callable[[Any], Any]:
    """How to turn a Vizier trial value back into the knob's native type."""
    if all(isinstance(c, str) for c in choices):
        return str  # categorical -> already a str
    if all(isinstance(c, int) and not isinstance(c, bool) for c in choices):
        return lambda v: round(float(v))  # discrete int (Vizier stores float)
    return float  # discrete float


def _index_decoder(choices: list[Any]) -> Callable[[Any], Any]:
    """Decode a categorical *index* back to the chosen entry. Used when a knob's
    choices include dicts (a composite knob with pinned-dict entries) — dicts can't
    be Vizier categorical values, so we categorize over the index instead."""
    return lambda v: choices[round(float(v))]


class VizierBranchSampler:
    """A Vizier study over one :class:`BranchSpace`.

    ``objectives`` is the list of ``(metric_name, maximize)`` the study optimizes; the
    default is a single ``("objective", maximize=True)`` (the caller pre-signs the score).
    Pass >=2 objectives for a **multi-objective / Pareto** study: the ``DEFAULT`` algorithm
    (GP-UCB-PE) optimizes the Pareto tradeoff via hypervolume scalarization, and ``observe``
    reports every objective's raw value in one measurement.
    """

    def __init__(
        self,
        branch: BranchSpace,
        *,
        study_id: str,
        objectives: list[tuple[str, bool]] | None = None,
    ):
        from vizier.service import clients
        from vizier.service import pyvizier as vz

        self.branch = branch
        self._objectives = objectives or [(_METRIC, True)]
        self._decoders: dict[str, Callable[[Any], Any]] = {}
        self._constants: dict[str, Any] = {}
        self._parallel_projector = ParallelConfigProjector(branch)
        self._parallel_pinned = len(branch.parallel_configs) == 1

        problem = vz.ProblemStatement()
        root = problem.search_space.root
        if not self._parallel_pinned:
            for parameter in self._parallel_projector.parameters:
                if parameter.is_constant:
                    continue
                if parameter.kind == "float":
                    root.add_float_param(
                        parameter.name,
                        min_value=parameter.minimum,
                        max_value=parameter.maximum,
                        default_value=parameter.default,
                    )
                elif parameter.kind == "discrete":
                    root.add_discrete_param(
                        parameter.name,
                        feasible_values=parameter.values,
                        default_value=parameter.default,
                        scale_type=vz.ScaleType.LOG
                        if parameter.log_scale
                        else vz.ScaleType.LINEAR,
                    )
                else:
                    root.add_categorical_param(
                        parameter.name,
                        feasible_values=parameter.values,
                        default_value=parameter.default,
                    )
        for knob, (minimum, maximum) in branch.float_ranges.items():
            root.add_float_param(
                knob,
                min_value=minimum,
                max_value=maximum,
                default_value=(minimum + maximum) / 2.0,
            )
            self._decoders[knob] = float
        for knob, choices in branch.knob_choices.items():
            if not any(isinstance(c, dict) for c in choices):
                # defensively dedupe hashable choices (order-preserving); duplicates
                # would otherwise crash Vizier study construction with an opaque error.
                # Composite (dict-bearing) knobs are left alone — dicts are unhashable
                # and their categorical decode is index-based, not value-based.
                choices = list(dict.fromkeys(choices))
            if len(choices) <= 1:
                if choices:
                    self._constants[knob] = choices[0]  # fixed -> inject, not a param
                continue
            if any(isinstance(c, dict) for c in choices):
                # composite knob with pinned-dict entries -> categorical over index
                root.add_categorical_param(knob, [str(i) for i in range(len(choices))])
                self._decoders[knob] = _index_decoder(choices)
            elif all(isinstance(c, str) for c in choices):
                kwargs = {"default_value": choices[0]} if knob == "backend" else {}
                root.add_categorical_param(knob, list(choices), **kwargs)
                self._decoders[knob] = _decoder_for(choices)
            else:
                root.add_discrete_param(knob, sorted(float(c) for c in choices))
                self._decoders[knob] = _decoder_for(choices)

        # GP designers reject a zero-dimensional study. A fully pinned request still
        # needs one internal constant so it can use the same ask/tell lifecycle.
        if problem.search_space.num_parameters() == 0:
            root.add_categorical_param(_CONSTANT_PARAM, ["0"], default_value="0")

        for name, maximize in self._objectives:
            goal = (
                vz.ObjectiveMetricGoal.MAXIMIZE
                if maximize
                else vz.ObjectiveMetricGoal.MINIMIZE
            )
            problem.metric_information.append(
                vz.MetricInformation(name=name, goal=goal)
            )
        study_config = vz.StudyConfig.from_problem(problem)
        # EXPERIMENT (env-gated; default DEFAULT = GP-bandit). The multi-objective GP suggest
        # can spin/hang at low observation counts; SPICA_VIZIER_ALGO=RANDOM_SEARCH bypasses the
        # GP (instant suggest, uniform exploration) to cover the curve ends without that stall.
        study_config.algorithm = os.environ.get("SPICA_VIZIER_ALGO", "DEFAULT")
        self._study = clients.Study.from_study_config(
            study_config, owner="dynamo.profiler.spica", study_id=study_id
        )

    def suggest(self, count: int) -> list[Suggestion]:
        suggestions: list[Suggestion] = []
        for trial in self._study.suggest(count=count):
            params = dict(trial.parameters)
            # backend is a searched knob now (in knob_choices) -> comes via _constants
            # (single backend) or _decoders (multiple), not a per-branch constant.
            selection: dict[str, Any] = {
                "deployment_mode": self.branch.deployment_mode,
                **self._constants,
            }
            for knob, decode in self._decoders.items():
                selection[knob] = decode(params[knob])
            if self._parallel_pinned:
                parallel_config = self.branch.parallel_configs[0]
                projection = None
            else:
                projection = self._parallel_projector.project(
                    params, selection["backend"]
                )
                parallel_config = projection.config
            suggestions.append(
                Suggestion(
                    selection=selection,
                    parallel_config=parallel_config,
                    handle=trial,
                    projection=projection,
                )
            )
        return suggestions

    @staticmethod
    def _update_projection_metadata(suggestion: Suggestion) -> None:
        if suggestion.projection is None:
            return
        from vizier.service import pyvizier as vz

        metadata = vz.Metadata()
        metadata["spica_projection"] = json.dumps(
            suggestion.projection.metadata(), sort_keys=True
        )
        suggestion.handle.update_metadata(metadata)

    def observe(self, suggestion: Suggestion, metrics: dict[str, float]) -> None:
        from vizier.service import pyvizier as vz

        self._update_projection_metadata(suggestion)
        suggestion.handle.complete(
            vz.Measurement(metrics={k: float(v) for k, v in metrics.items()})
        )

    def observe_infeasible(self, suggestion: Suggestion, reason: str) -> None:
        """Mark a candidate that could not be evaluated (e.g. replay error) so the
        study still closes the trial and the optimizer moves on."""
        from vizier.service import pyvizier as vz

        self._update_projection_metadata(suggestion)
        suggestion.handle.complete(vz.Measurement(), infeasible_reason=reason)


def make_branch_sampler(
    branch: BranchSpace,
    *,
    study_id: str,
    objectives: list[tuple[str, bool]] | None = None,
) -> BranchSampler:
    """Construct the default (Vizier) sampler for a branch. ``objectives`` (name, maximize)
    pairs default to a single maximized ``"objective"``; pass >=2 for a Pareto study."""
    return VizierBranchSampler(branch, study_id=study_id, objectives=objectives)
