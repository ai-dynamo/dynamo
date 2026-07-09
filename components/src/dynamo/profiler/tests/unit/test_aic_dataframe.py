# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for ``make_parallel_label`` collisions.

The legacy 3-bucket label collapsed distinct 5-tuples to the same string,
which corrupted ``thorough.py`` sweep work_dir naming and the
``aiconfigurator.sdk.picking`` ``groupby("parallel")`` dedup. These tests
pin the post-fix unique encoding.
"""
from __future__ import annotations

import pandas as pd
import pytest

from dynamo.profiler.utils.aic_dataframe import make_parallel_label

pytestmark = [
    pytest.mark.aiconfigurator,
    pytest.mark.gpu_0,
    # TODO: revert to pytest.mark.post_merge after pre_merge validation on this PR
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]

_NUM_GPUS = [1, 2, 4, 8, 16]


def _enumerate() -> list[tuple[int, int, int, int, int]]:
    """Enumerate realistic ``(tp, pp, dp, moe_tp, moe_ep)`` tuples used as
    test inputs across four archetypes (pure TP-of-experts /
    pure TEP-of-attention / pure DEP / dense) at
    ``num_gpus ∈ {1, 2, 4, 8, 16}``."""
    out: list[tuple[int, int, int, int, int]] = []
    for n in _NUM_GPUS:
        if n == 1:
            out.append((1, 1, 1, 1, 1))
            continue
        out.append((n, 1, 1, n, 1))  # pure TP-of-experts
        out.append((n, 1, 1, 1, n))  # pure TEP-of-attention
        out.append((1, 1, n, 1, n))  # pure DEP
        out.append((n, 1, 1, 1, 1))  # dense
    return out


def test_label_distinct_across_all_enumerated_tuples() -> None:
    """Every distinct 5-tuple must yield a distinct label string."""
    seen: dict[str, tuple[int, int, int, int, int]] = {}
    for tup in _enumerate():
        label = make_parallel_label(*tup)
        assert (
            label not in seen
        ), f"label collision: {label!r} from {tup} also from {seen[label]}"
        seen[label] = tup


def test_groupby_does_not_merge_distinct_topologies() -> None:
    """``aiconfigurator.sdk.picking`` does ``df.groupby("parallel")``. Two
    distinct topologies must produce two groups, not one."""
    df = pd.DataFrame(
        [
            {"parallel": make_parallel_label(2, 1, 1, 1, 2), "seq/s/gpu": 100.0},
            {"parallel": make_parallel_label(1, 1, 2, 1, 2), "seq/s/gpu": 90.0},
        ]
    )
    groups = list(df.groupby("parallel"))
    assert (
        len(groups) == 2
    ), "groupby merged distinct topologies — label is not injective"


# (tp, pp, dp, moe_tp, moe_ep, expected_label)
_FORMAT_PINS: list[tuple[int, int, int, int, int, str]] = [
    # All-default baseline.
    (1, 1, 1, 1, 1, "tp1"),
    # Default-1 omission: only tp varies.
    (4, 1, 1, 1, 1, "tp4"),
    # Docstring examples (mirrored from PickedParallelConfig.label).
    (2, 1, 1, 2, 1, "tp2-moetp2"),
    (2, 1, 1, 1, 2, "tp2-moeep2"),
    (1, 1, 2, 1, 2, "tp1-dp2-moeep2"),
    (4, 1, 1, 4, 1, "tp4-moetp4"),
    # PP dimension (otherwise unexercised by _enumerate()).
    (2, 2, 1, 1, 1, "tp2-pp2"),
    # Dense-with-attention-DP (no MoE).
    (2, 1, 2, 1, 1, "tp2-dp2"),
]


@pytest.mark.parametrize("tp,pp,dp,moe_tp,moe_ep,expected", _FORMAT_PINS)
def test_make_parallel_label_format_pin(
    tp: int, pp: int, dp: int, moe_tp: int, moe_ep: int, expected: str
) -> None:
    """Pin the exact label string — ``aiconfigurator.sdk.picking`` and
    ``thorough.py`` consume this string verbatim, so format drift must be
    a deliberate, test-breaking edit."""
    assert make_parallel_label(tp, pp, dp, moe_tp, moe_ep) == expected
