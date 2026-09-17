# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for bounded candidate selection (tracking issue #13545,
item 5). Uses a plain dataclass matching CandidateLike -- no real Sweeper
Candidate needed."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import pytest

from dynamo.profiler.sweeper.candidate_selector import (
    ObjectiveSpec,
    ParetoGoal,
    ScalarGoal,
    update_selection,
)


@dataclass(frozen=True)
class _Candidate:
    score: float = 0.0
    objectives: Mapping[str, float] = field(default_factory=dict)
    used_gpus: int = 8
    name: str = ""


def test_scalar_first_candidate_is_selected() -> None:
    selection = update_selection([], _Candidate(score=1.0), goal=ScalarGoal())
    assert len(selection) == 1
    assert selection[0].score == 1.0


def test_scalar_higher_score_new_candidate_is_ranked_first() -> None:
    """Both survive (within the default bound) -- the new, better
    candidate takes list position 0, the old one moves to position 1.
    Confirmed real behavior is a ranked list, not a single replaced
    winner: "candidates = sweeper.run(config); best = candidates[0]" --
    Sweeper itself returns the full best-first list."""
    current = [_Candidate(score=1.0, name="old")]
    selection = update_selection(current, _Candidate(score=2.0, name="new"), goal=ScalarGoal())
    assert [c.name for c in selection] == ["new", "old"]


def test_scalar_lower_score_new_candidate_is_ranked_last() -> None:
    current = [_Candidate(score=2.0, name="old")]
    selection = update_selection(current, _Candidate(score=1.0, name="new"), goal=ScalarGoal())
    assert [c.name for c in selection] == ["old", "new"]


def test_scalar_tied_score_breaks_to_fewer_gpus() -> None:
    """Confirmed against the real Sweeper docs: "rank (best score, ties ->
    fewer GPUs) for every scalar goal.\""""
    current = [_Candidate(score=1.0, used_gpus=8, name="old")]
    new = _Candidate(score=1.0, used_gpus=4, name="new")
    selection = update_selection(current, new, goal=ScalarGoal())
    assert [c.name for c in selection] == ["new", "old"]


def test_scalar_tied_score_more_gpus_ranks_after_fewer_gpus() -> None:
    current = [_Candidate(score=1.0, used_gpus=4, name="old")]
    new = _Candidate(score=1.0, used_gpus=8, name="new")
    selection = update_selection(current, new, goal=ScalarGoal())
    assert [c.name for c in selection] == ["old", "new"]


def test_scalar_bounded_top_n_evicts_the_worst_ranked_candidate() -> None:
    goal = ScalarGoal(max_candidates=2)
    current = [
        _Candidate(score=3.0, name="best"),
        _Candidate(score=2.0, name="middle"),
    ]
    new = _Candidate(score=1.0, name="worst")

    selection = update_selection(current, new, goal=goal)

    assert [c.name for c in selection] == ["best", "middle"]


def test_scalar_new_candidate_displacing_the_worst_is_kept() -> None:
    goal = ScalarGoal(max_candidates=2)
    current = [
        _Candidate(score=3.0, name="best"),
        _Candidate(score=2.0, name="middle"),
    ]
    new = _Candidate(score=2.5, name="better_than_middle")

    selection = update_selection(current, new, goal=goal)

    assert [c.name for c in selection] == ["best", "better_than_middle"]



_MAXIMIZE_THROUGHPUT = (ObjectiveSpec(name="throughput", maximize=True),)
_MAXIMIZE_THROUGHPUT_MINIMIZE_TTFT = (
    ObjectiveSpec(name="throughput", maximize=True),
    ObjectiveSpec(name="ttft", maximize=False),
)


def test_pareto_first_candidate_is_selected() -> None:
    goal = ParetoGoal(objectives=_MAXIMIZE_THROUGHPUT)
    candidate = _Candidate(objectives={"throughput": 100.0})
    assert update_selection([], candidate, goal=goal) == [candidate]


def test_pareto_dominated_new_candidate_is_discarded() -> None:
    goal = ParetoGoal(objectives=_MAXIMIZE_THROUGHPUT)
    current = [_Candidate(objectives={"throughput": 100.0}, name="better")]
    dominated = _Candidate(objectives={"throughput": 50.0}, name="worse")
    assert update_selection(current, dominated, goal=goal) == current


def test_pareto_dominating_new_candidate_evicts_the_dominated_incumbent() -> None:
    goal = ParetoGoal(objectives=_MAXIMIZE_THROUGHPUT)
    current = [_Candidate(objectives={"throughput": 50.0}, name="worse")]
    better = _Candidate(objectives={"throughput": 100.0}, name="better")
    assert update_selection(current, better, goal=goal) == [better]


def test_pareto_non_dominated_candidates_both_survive() -> None:
    """Trade-off case: neither candidate dominates the other (one better
    on throughput, the other better on ttft) -- both must be kept."""
    goal = ParetoGoal(objectives=_MAXIMIZE_THROUGHPUT_MINIMIZE_TTFT)
    current = [_Candidate(objectives={"throughput": 100.0, "ttft": 50.0}, name="fast_throughput")]
    tradeoff = _Candidate(objectives={"throughput": 80.0, "ttft": 20.0}, name="low_latency")

    selection = update_selection(current, tradeoff, goal=goal)

    assert len(selection) == 2
    assert set(c.name for c in selection) == {"fast_throughput", "low_latency"}


def test_pareto_respects_minimize_direction_not_just_maximize() -> None:
    """A lower ttft must count as strictly better when maximize=False --
    confirms direction is actually read per-objective, not hardcoded to
    "higher is always better.\""""
    goal = ParetoGoal(objectives=(ObjectiveSpec(name="ttft", maximize=False),))
    current = [_Candidate(objectives={"ttft": 100.0}, name="slow")]
    faster = _Candidate(objectives={"ttft": 50.0}, name="fast")
    assert update_selection(current, faster, goal=goal) == [faster]


def test_pareto_bounded_front_evicts_when_exceeding_max_candidates() -> None:
    """Single-objective Pareto degenerates to "only the max survives", so
    two objectives are used here to let three genuinely non-dominated
    points coexist momentarily, exercising real crowding-based eviction.
    Points are NOT collinear -- a straight-line trade-off makes crowding
    distances tie exactly, making eviction ambiguous to assert on."""
    goal = ParetoGoal(objectives=_MAXIMIZE_THROUGHPUT_MINIMIZE_TTFT, max_candidates=2)
    current = [
        _Candidate(objectives={"throughput": 10.0, "ttft": 10.0}, name="a"),
        _Candidate(objectives={"throughput": 45.0, "ttft": 55.0}, name="b"),
    ]
    new = _Candidate(objectives={"throughput": 90.0, "ttft": 90.0}, name="c")

    selection = update_selection(current, new, goal=goal)

    assert len(selection) == 2
    # "a" and "c" are boundary candidates on both objectives and get
    # infinite crowding distance; "b" is the only interior, finite-distance
    # candidate and must be the one evicted.
    assert set(c.name for c in selection) == {"a", "c"}


def test_pareto_boundary_candidates_are_never_evicted_for_crowding() -> None:
    """The best-on-each-objective candidates get infinite crowding
    distance by construction and must survive eviction regardless of how
    many interior points are added. Values follow the same trade-off
    pattern verified above (throughput and ttft move together, since a
    higher-throughput point pays for it with a higher/worse ttft) --
    otherwise "best_throughput" would simply dominate "best_ttft"
    outright rather than trade off against it."""
    goal = ParetoGoal(objectives=_MAXIMIZE_THROUGHPUT_MINIMIZE_TTFT, max_candidates=3)
    current = [
        _Candidate(objectives={"throughput": 10.0, "ttft": 10.0}, name="best_ttft"),
        _Candidate(objectives={"throughput": 90.0, "ttft": 90.0}, name="best_throughput"),
        _Candidate(objectives={"throughput": 45.0, "ttft": 55.0}, name="middle"),
    ]
    new_interior = _Candidate(objectives={"throughput": 50.0, "ttft": 60.0}, name="new_middle")

    selection = update_selection(current, new_interior, goal=goal)

    names = {c.name for c in selection}
    assert "best_ttft" in names
    assert "best_throughput" in names
