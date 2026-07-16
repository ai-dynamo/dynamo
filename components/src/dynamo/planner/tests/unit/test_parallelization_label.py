# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for ``PickedParallelConfig.label`` collisions.

The legacy 3-bucket label (``dep{moe_ep}`` / ``tep{moe_tp}`` / ``tp{tp}``)
collapsed distinct 5-tuples to the same string — notably
``(tp=2, moe_ep=2)`` and ``(tp=1, dp=2, moe_ep=2)`` both mapped to
``"dep2"``. That broke ``thorough.py`` work_dir naming and
``aiconfigurator.sdk.picking`` ``groupby("parallel")`` dedup. These tests
pin the post-fix unique encoding.
"""

from __future__ import annotations

import pytest

pytestmark = [
    pytest.mark.gpu_0,
    # TODO: revert to pytest.mark.post_merge after pre_merge validation on this PR
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]

try:
    from dynamo.planner.config.parallelization import PickedParallelConfig
except ImportError as e:
    pytest.skip(f"Skip (missing dependency): {e}", allow_module_level=True)


# Realistic enumeration: pure TEP / pure DEP / pure TP / dense at
# num_gpus in {1, 2, 4, 8, 16}.
_NUM_GPUS = [1, 2, 4, 8, 16]


def _enumerate() -> list[tuple[int, int, int, int, int]]:
    """Enumerate realistic ``(tp, pp, dp, moe_tp, moe_ep)`` tuples used as
    test inputs — mirrors ``filter_real_silicon_configs`` across four
    archetypes (pure TP-of-experts / pure TEP-of-attention / pure DEP /
    dense) at ``num_gpus ∈ {1, 2, 4, 8, 16}``."""
    out: list[tuple[int, int, int, int, int]] = []
    for n in _NUM_GPUS:
        if n == 1:
            out.append((1, 1, 1, 1, 1))
            continue
        out.append((n, 1, 1, n, 1))  # pure TP-of-experts (moe_tp = tp = n)
        out.append((n, 1, 1, 1, n))  # pure TEP-of-attention (tp = n, moe_ep = n)
        out.append((1, 1, n, 1, n))  # pure DEP (dp = n, moe_ep = n)
        out.append((n, 1, 1, 1, 1))  # dense (no MoE sharding)
    return out


@pytest.mark.parametrize("tup", _enumerate())
def test_label_unique_per_tuple(tup: tuple[int, int, int, int, int]) -> None:
    """Every enumerated tuple must produce a non-empty label."""
    tp, pp, dp, moe_tp, moe_ep = tup
    cfg = PickedParallelConfig(tp=tp, pp=pp, dp=dp, moe_tp=moe_tp, moe_ep=moe_ep)
    assert cfg.label()


def test_label_distinct_across_all_enumerated_tuples() -> None:
    """Every distinct 5-tuple must yield a distinct label string."""
    seen: dict[str, tuple[int, int, int, int, int]] = {}
    for tup in _enumerate():
        cfg = PickedParallelConfig(
            tp=tup[0], pp=tup[1], dp=tup[2], moe_tp=tup[3], moe_ep=tup[4]
        )
        label = cfg.label()
        assert (
            label not in seen
        ), f"label collision: {label!r} from {tup} also from {seen[label]}"
        seen[label] = tup


def test_regression_dep2_collision() -> None:
    """Explicit regression for the historical ``dep2`` collision."""
    a = PickedParallelConfig(tp=2, dp=1, moe_ep=2)
    b = PickedParallelConfig(tp=1, dp=2, moe_ep=2)
    assert a != b
    assert a.label() != b.label()


# (tp, pp, dp, moe_tp, moe_ep, expected_label)
_FORMAT_PINS: list[tuple[int, int, int, int, int, str]] = [
    # All-default baseline.
    (1, 1, 1, 1, 1, "tp1"),
    # Default-1 omission: only tp varies.
    (4, 1, 1, 1, 1, "tp4"),
    # Docstring examples.
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
def test_label_format_pin(
    tp: int, pp: int, dp: int, moe_tp: int, moe_ep: int, expected: str
) -> None:
    """Pin the exact label string — downstream consumers (``thorough.py``
    work_dir, ``aiconfigurator.sdk.picking``) depend on the format, so a
    separator / prefix change must be a deliberate, test-breaking edit."""
    cfg = PickedParallelConfig(tp=tp, pp=pp, dp=dp, moe_tp=moe_tp, moe_ep=moe_ep)
    assert cfg.label() == expected
