# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Squeeze-Evolve operators (binding-free)."""

from __future__ import annotations

import random

import pytest

from dynamo.squeeze_evolve.operators import (
    assign_routes,
    compute_thresholds,
    extract_answer,
    extract_boxed_math_answer,
    group_diversity,
    make_aggregate_prompt,
    select_uniform,
    strip_think_blocks,
    update_replace,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]

# -- answer extraction ------------------------------------------------------


def test_extract_boxed_basic_and_normalized():
    assert extract_boxed_math_answer("the answer is \\boxed{42}") == "42"
    assert extract_boxed_math_answer("\\boxed{1,000}") == "1000"  # comma stripped
    assert extract_boxed_math_answer("\\boxed{007}") == "7"  # int-canonicalized


def test_extract_boxed_nested_braces():
    assert extract_boxed_math_answer("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}".lower()


def test_extract_falls_back_to_tag_then_final_line():
    assert extract_boxed_math_answer("<answer> 13 </answer>") == "13"
    assert extract_boxed_math_answer("blah\nFinal answer: 99") == "99"


def test_extract_strips_think_blocks():
    assert extract_boxed_math_answer("<think>lots</think>\\boxed{5}") == "5"
    assert strip_think_blocks("<think>x</think>done") == "done"


def test_extract_empty():
    assert extract_boxed_math_answer("") == ""
    assert extract_boxed_math_answer(None) == ""  # type: ignore[arg-type]


def test_extract_answer_task_aware():
    assert extract_answer("\\boxed{7}", "math") == "7"
    # generic: the boxed/normalized extraction is used when it yields something
    assert extract_answer("answer: 7", "generic") == "7"
    # only falls back to strip_think when extraction yields nothing (empty input)
    assert extract_answer("", "generic") == ""


# -- fitness / selection / thresholds / routing -----------------------------


def test_group_diversity_counts_unique():
    assert group_diversity(["1", "1", "2"]) == 2.0
    assert group_diversity(["3", "3", "3"]) == 1.0


def test_select_uniform_seeded_shapes():
    random.seed(0)
    groups = select_uniform(list("abcdefgh"), k=3, m=5)
    assert len(groups) == 5
    assert all(len(g) == 3 for g in groups)
    assert all(all(0 <= i < 8 for i in g) for g in groups)
    assert all(len(set(g)) == 3 for g in groups)  # sampled without replacement
    # determinism under the same seed
    random.seed(0)
    assert select_uniform(list("abcdefgh"), 3, 5) == groups


def test_select_uniform_request_local_rng():
    # A request-local Random is reproducible and independent of global RNG state.
    out = select_uniform(list("abcdefgh"), 3, 5, random.Random(123))
    assert out == select_uniform(list("abcdefgh"), 3, 5, random.Random(123))


def test_compute_thresholds_edges_and_sorted():
    gf = [1.0, 2.0, 3.0, 4.0]
    assert compute_thresholds(gf, [100]) == [max(gf) + 1.0]
    assert compute_thresholds(gf, [0]) == [min(gf) - 1.0]
    thr = compute_thresholds(gf, [66, 33])
    assert thr == sorted(thr)  # ascending regardless of input order


def test_assign_routes_low_fitness_to_expensive():
    # 2 tiers, threshold [2.0]: f<=2 -> tier 1 (expensive), f>2 -> tier 0 (cheap)
    routes = assign_routes([1.0, 2.0, 3.0, 4.0], [2.0], n_tiers=2)
    assert routes == [1, 1, 0, 0]


def test_assign_routes_three_tiers():
    # thresholds [2,3]: f<=2 -> tier 2; 2<f<=3 -> tier 1; f>3 -> tier 0
    routes = assign_routes([1.0, 2.5, 3.0, 4.0], [2.0, 3.0], n_tiers=3)
    assert routes == [2, 1, 1, 0]


def test_diversity_routing_sends_consensus_to_cheap():
    # Routing fitness is negative diversity (as the orchestrator computes it):
    # a consensus group (low diversity) routes to the cheap tier, a diverse group
    # (high diversity) to the expensive tier. Easy goes cheap, hard goes expensive.
    gf = [-group_diversity(["7", "7"]), -group_diversity(["1", "2"])]  # div 1, 2
    routes = assign_routes(gf, compute_thresholds(gf, [50.0]), n_tiers=2)
    assert routes == [0, 1]  # consensus -> cheap (0), diverse -> expensive (1)


# -- recombination / update -------------------------------------------------


def test_make_aggregate_prompt_branches():
    assert make_aggregate_prompt("Q?", [], "\\boxed{}") == "Q?"  # no candidates
    one = make_aggregate_prompt("Q?", ["sol"], "\\boxed{}")
    assert "candidate solution" in one.lower() and "\\boxed{}" in one
    many = make_aggregate_prompt("Q?", ["a", "b"], "\\boxed{}")
    assert "Solution 1" in many and "Solution 2" in many and "\\boxed{}" in many


def test_update_replace():
    assert update_replace(["old"], ["new1", "new2"]) == ["new1", "new2"]
