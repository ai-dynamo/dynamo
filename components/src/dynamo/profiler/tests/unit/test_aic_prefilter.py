# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for AIC-based THOROUGH pre-filter."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

pytestmark = [
    pytest.mark.post_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.planner,
]

try:
    from dynamo.profiler.utils.aic_dataframe import make_parallel_label
    from dynamo.profiler.utils.aic_prefilter import (
        prefilter_decode_candidates,
        prefilter_prefill_candidates,
    )
except ImportError as e:
    pytest.skip(f"Skip (missing dependency): {e}", allow_module_level=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeCandidate:
    tp: int
    pp: int
    dp: int
    moe_tp: int
    moe_ep: int
    num_gpus: int


def _make_candidates(
    specs: list[tuple[int, int, int, int, int]],
) -> list[_FakeCandidate]:
    """Build candidates from (tp, pp, dp, moe_tp, moe_ep) tuples."""
    return [
        _FakeCandidate(
            tp=tp, pp=pp, dp=dp, moe_tp=mtp, moe_ep=mep, num_gpus=tp * pp * dp,
        )
        for tp, pp, dp, mtp, mep in specs
    ]


def _label(c: _FakeCandidate) -> str:
    return make_parallel_label(c.tp, c.pp, c.dp, c.moe_tp, c.moe_ep)


# ---------------------------------------------------------------------------
# Prefill pre-filter
# ---------------------------------------------------------------------------


class TestPrefilterPrefill:
    def test_noop_when_top_n_none(self):
        candidates = _make_candidates([(1, 1, 1, 1, 1)] * 5)
        result = prefilter_prefill_candidates(candidates, None, pd.DataFrame())
        assert result == candidates

    def test_noop_when_candidates_within_top_n(self):
        candidates = _make_candidates([(1, 1, 1, 1, 1)] * 3)
        result = prefilter_prefill_candidates(candidates, 5, pd.DataFrame())
        assert result == candidates

    def test_selects_top_n_by_ttft(self):
        candidates = _make_candidates([
            (1, 1, 1, 1, 1),   # label: tp1
            (2, 1, 1, 1, 1),   # label: tp2
            (4, 1, 1, 1, 1),   # label: tp4
            (1, 1, 2, 1, 2),   # label: dep2
            (1, 1, 4, 1, 4),   # label: dep4
        ])
        pareto_df = pd.DataFrame([
            {"parallel": "tp1", "ttft": 50.0},
            {"parallel": "tp2", "ttft": 30.0},
            {"parallel": "tp4", "ttft": 10.0},
            {"parallel": "dep2", "ttft": 40.0},
            {"parallel": "dep4", "ttft": 20.0},
        ])

        result = prefilter_prefill_candidates(candidates, 3, pareto_df)

        assert len(result) == 3
        labels = [_label(c) for c in result]
        assert labels == ["tp4", "dep4", "tp2"]  # ttft: 10, 20, 30

    def test_unmatched_candidates_sorted_last(self):
        candidates = _make_candidates([
            (1, 1, 1, 1, 1),   # tp1 — has prediction
            (2, 1, 1, 1, 1),   # tp2 — no prediction
        ])
        pareto_df = pd.DataFrame([{"parallel": "tp1", "ttft": 50.0}])

        result = prefilter_prefill_candidates(candidates, 1, pareto_df)

        assert len(result) == 1
        assert result[0].tp == 1  # tp1 has ttft=50, tp2 has inf → tp1 wins

    def test_fallback_on_empty_pareto_df(self):
        candidates = _make_candidates([(1, 1, 1, 1, 1)] * 5)
        result = prefilter_prefill_candidates(candidates, 3, pd.DataFrame())
        assert len(result) == 5

    def test_fallback_on_missing_ttft_column(self):
        candidates = _make_candidates([(1, 1, 1, 1, 1)] * 5)
        pareto_df = pd.DataFrame([{"parallel": "tp1", "tpot": 10.0}])
        result = prefilter_prefill_candidates(candidates, 3, pareto_df)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# Decode pre-filter
# ---------------------------------------------------------------------------


class TestPrefilterDecode:
    def test_selects_top_n_by_throughput_descending(self):
        candidates = _make_candidates([
            (1, 1, 1, 1, 1),   # tp1
            (2, 1, 1, 1, 1),   # tp2
            (4, 1, 1, 1, 1),   # tp4
            (1, 1, 2, 1, 2),   # dep2
            (1, 1, 4, 1, 4),   # dep4
        ])
        pareto_df = pd.DataFrame([
            {"parallel": "tp1", "seq/s/gpu": 100.0},
            {"parallel": "tp2", "seq/s/gpu": 500.0},
            {"parallel": "tp4", "seq/s/gpu": 300.0},
            {"parallel": "dep2", "seq/s/gpu": 200.0},
            {"parallel": "dep4", "seq/s/gpu": 400.0},
        ])

        result = prefilter_decode_candidates(candidates, 3, pareto_df)

        assert len(result) == 3
        labels = [_label(c) for c in result]
        assert labels == ["tp2", "dep4", "tp4"]  # thpt: 500, 400, 300

    def test_fallback_on_empty_pareto_df(self):
        candidates = _make_candidates([(1, 1, 1, 1, 1)] * 5)
        result = prefilter_decode_candidates(candidates, 3, pd.DataFrame())
        assert len(result) == 5

    def test_fallback_on_missing_throughput_column(self):
        candidates = _make_candidates([(1, 1, 1, 1, 1)] * 5)
        pareto_df = pd.DataFrame([{"parallel": "tp1", "ttft": 10.0}])
        result = prefilter_decode_candidates(candidates, 3, pareto_df)
        assert len(result) == 5

    def test_noop_when_top_n_zero(self):
        candidates = _make_candidates([(1, 1, 1, 1, 1)] * 5)
        result = prefilter_decode_candidates(candidates, 0, pd.DataFrame())
        assert result == candidates

    def test_prefers_tokens_per_s_gpu_over_seq_s_gpu(self):
        """When both throughput columns exist, prefer tokens/s/gpu."""
        candidates = _make_candidates([
            (1, 1, 1, 1, 1),   # tp1
            (2, 1, 1, 1, 1),   # tp2
        ])
        pareto_df = pd.DataFrame([
            {"parallel": "tp1", "tokens/s/gpu": 200.0, "seq/s/gpu": 999.0},
            {"parallel": "tp2", "tokens/s/gpu": 100.0, "seq/s/gpu": 1.0},
        ])

        result = prefilter_decode_candidates(candidates, 1, pareto_df)
        assert len(result) == 1
        assert result[0].tp == 1  # tokens/s/gpu: tp1=200 > tp2=100
