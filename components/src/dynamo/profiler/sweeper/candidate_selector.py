# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded candidate selection (tracking issue #13545, item 5: "select
bounded scalar/Pareto results"). Pure function: given the current
selection and one newly-scored Candidate, returns the updated selection.
Maintaining state across a stream of candidates is the caller's job.

Confirmed against the real Sweeper Results/Optimization Goals docs, not
assumed: Candidate.objectives is a dict keyed by OptimizationTarget value;
each objective's maximize/minimize direction is a search-level property
("[(t.value, t.maximize) for t in resolved_pareto_objectives]"), not a
per-candidate field -- callers supply it via ObjectiveSpec. Scalar goals
break score ties by fewer GPUs ("rank (best score, ties -> fewer GPUs)
for every scalar goal").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence


class CandidateLike(Protocol):
    """Matches the real, confirmed Sweeper Candidate's relevant fields."""

    score: float
    objectives: Mapping[str, float]
    used_gpus: int


@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    maximize: bool


@dataclass(frozen=True)
class ScalarGoal:
    """Selection is bounded to exactly one candidate."""


@dataclass(frozen=True)
class ParetoGoal:
    objectives: Sequence[ObjectiveSpec]
    max_candidates: int = 5  # matches the real, confirmed RecommendationSpec.MaxCandidates default


Goal = ScalarGoal | ParetoGoal


def _dominates(a: CandidateLike, b: CandidateLike, objectives: Sequence[ObjectiveSpec]) -> bool:
    """True if `a` dominates `b`: at least as good on every objective (per
    each objective's own maximize/minimize direction) and strictly better
    on at least one."""
    at_least_as_good_everywhere = True
    strictly_better_somewhere = False
    for obj in objectives:
        a_val = a.objectives[obj.name]
        b_val = b.objectives[obj.name]
        if obj.maximize:
            if a_val < b_val:
                at_least_as_good_everywhere = False
            if a_val > b_val:
                strictly_better_somewhere = True
        else:
            if a_val > b_val:
                at_least_as_good_everywhere = False
            if a_val < b_val:
                strictly_better_somewhere = True
    return at_least_as_good_everywhere and strictly_better_somewhere


def _crowding_distances(
    front: Sequence[CandidateLike], objectives: Sequence[ObjectiveSpec]
) -> list[float]:
    """NSGA-II crowding distance: how isolated each candidate is from its
    neighbors on the front, per objective, summed. Boundary candidates
    (best/worst on an objective) get infinite distance on that objective,
    so they're never evicted for being "crowded" -- eviction should thin
    dense clusters in the interior, not trim the front's own extremes.
    """
    n = len(front)
    if n <= 2:
        return [float("inf")] * n

    distances = [0.0] * n
    for obj in objectives:
        order = sorted(range(n), key=lambda i: front[i].objectives[obj.name])
        distances[order[0]] = float("inf")
        distances[order[-1]] = float("inf")
        obj_range = front[order[-1]].objectives[obj.name] - front[order[0]].objectives[obj.name]
        if obj_range == 0:
            continue
        for pos in range(1, n - 1):
            i = order[pos]
            prev_val = front[order[pos - 1]].objectives[obj.name]
            next_val = front[order[pos + 1]].objectives[obj.name]
            distances[i] += (next_val - prev_val) / obj_range
    return distances


def _evict_most_crowded(
    front: list[CandidateLike], objectives: Sequence[ObjectiveSpec], max_candidates: int
) -> list[CandidateLike]:
    while len(front) > max_candidates:
        distances = _crowding_distances(front, objectives)
        most_crowded_index = min(range(len(front)), key=lambda i: distances[i])
        front.pop(most_crowded_index)
    return front


def update_selection(
    current: Sequence[CandidateLike], new_candidate: CandidateLike, *, goal: Goal
) -> list[CandidateLike]:
    """Return the updated selection after observing one new candidate.

    ScalarGoal: keeps whichever of current/new has the better score;
    ties break to fewer used_gpus (both confirmed against the real
    Sweeper docs). Selection is always 0 or 1 candidates.

    ParetoGoal: standard incremental non-dominated-set update. A new
    candidate dominated by anything already selected is discarded
    outright. Otherwise it's added, and anything it now dominates is
    removed (a previously non-dominated candidate can become dominated
    by a new, stronger arrival). If the result exceeds max_candidates,
    the most crowded candidate (smallest NSGA-II crowding distance) is
    evicted repeatedly until it fits -- thinning dense interior clusters
    rather than trimming the front's own extremes.
    """
    if isinstance(goal, ScalarGoal):
        if not current:
            return [new_candidate]
        (best,) = current
        if new_candidate.score > best.score:
            return [new_candidate]
        if new_candidate.score == best.score and new_candidate.used_gpus < best.used_gpus:
            return [new_candidate]
        return list(current)

    if any(_dominates(existing, new_candidate, goal.objectives) for existing in current):
        return list(current)

    survivors = [
        existing
        for existing in current
        if not _dominates(new_candidate, existing, goal.objectives)
    ]
    survivors.append(new_candidate)

    return _evict_most_crowded(survivors, goal.objectives, goal.max_candidates)
