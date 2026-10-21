# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import math
import random

import pytest

from benchmarks.gateway_ab import report, stats

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.parallel,
]


def _write_run(root, side, rnd, ttft, **extra):
    """A run file of agg at 14 req/s whose TTFT statistics are all ``ttft`` ms."""
    path = root / "runs" / "agg" / side / "rps14" / f"r{rnd}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    run = {
        "records": 100,
        "errors": 0,
        "expected_requests": 100,
        "request_throughput": 14.0,
        "output_token_throughput": 1700.0,
        "latency": {
            metric: dict.fromkeys(stats.LATENCY_STATS.values(), ttft)
            for metric in stats.LATENCY_METRICS
        },
    }
    run.update(extra)
    path.write_text(json.dumps(run))


@pytest.mark.parametrize(
    "df,expected", [(1, 12.7062047), (2, 4.30265273), (10, 2.22813885)]
)
def test_t_quantile_matches_tables(df, expected):
    assert stats.t_quantile(0.975, df) == pytest.approx(expected, abs=1e-6)


def test_interval_holds_its_95_percent_on_a_a_runs():
    """A/A rounds: about 5% of intervals may exclude zero, not 20%.

    Round-to-round noise both shared by the two sides (the seed's arrival
    pattern) and not shared (node noise) is in play. Three rounds is the
    default, where a bootstrap over rounds is far too narrow.
    """
    rng = random.Random(7)
    trials, misses = 2000, 0
    for _ in range(trials):
        a, b = [], []
        for _ in range(3):
            shared = rng.gauss(0, 0.05)
            a.append(50 * math.exp(shared + rng.gauss(0, 0.02)))
            b.append(50 * math.exp(shared + rng.gauss(0, 0.02)))
        _, (lo, hi) = stats.paired_change(a, b)
        misses += lo > 0 or hi < 0
    assert 0.03 < misses / trials < 0.07


def test_rounds_pair_by_index_and_the_columns_match_the_delta():
    # Round 1 is missing on B and round 3 on A, so only rounds 0 and 2 pair.
    row = stats.compare_scalar(
        {0: 50.0, 1: 52.0, 2: 51.0},
        {0: 55.0, 2: 56.1, 3: 99.0, 4: None},
        higher_is_better=False,
    )
    assert row["rounds"] == 2
    assert row["delta_pct"] == pytest.approx(10, abs=0.01)
    # The columns are geometric means, so Δ is B against A as they show it.
    assert row["delta_pct"] == pytest.approx((row["b"] / row["a"] - 1) * 100)
    assert row["verdict"] == "B worse"


@pytest.mark.parametrize(
    "ci,higher,expected",
    [
        ((5.0, 9.0), False, "B worse"),
        ((5.0, 9.0), True, "B better"),
        ((0.5, 1.5), False, "B slightly worse"),
        ((-1.0, 1.0), False, "same"),
        ((-1.0, 5.0), False, "inconclusive"),
        ((math.nan, math.nan), False, "n/a"),
    ],
)
def test_verdict(ci, higher, expected):
    assert stats.verdict(ci, 2.0, higher) == expected
    gateway = expected.replace("B ", "gateway ")
    assert stats.verdict(ci, 2.0, higher, "gateway") == gateway


def test_summary_gates_on_the_low_end_of_the_interval(tmp_path):
    rng = random.Random(4)
    for rnd in range(3):
        shared = rng.gauss(0, 2)
        _write_run(tmp_path, "frontend", rnd, 50 + shared)
        _write_run(tmp_path, "epp", rnd, 65 + shared + rng.gauss(0, 0.5))
    # An invalid run is listed and left out, not averaged in.
    _write_run(tmp_path, "epp", 3, 1.0, errors=5)

    summary = report.build_summary(tmp_path)
    (point,) = summary["points"]
    assert point["rounds"] == {"a": 3, "b": 3, "paired": 3}
    mean = point["latency"]["time_to_first_token"]["mean"]
    assert mean["verdict"] == "B worse"
    assert mean["delta_pct"] == pytest.approx(30, abs=3)
    assert report.regressions(summary, 10.0)
    assert not report.regressions(summary, 50.0)
    assert len(report.incomplete(summary)) == 1
    md = report.render_markdown(summary)
    assert "## agg" in md and "TTFT p90" in md and "Excluded runs" in md


def test_point_with_only_failure_markers_is_reported(tmp_path):
    for side in ("frontend", "epp"):
        marker = tmp_path / "runs" / "disagg" / side / "c8" / "r0.error"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("failed\n")
    summary = report.build_summary(tmp_path)
    (point,) = summary["points"]
    assert point["topology"] == "disagg" and not point["latency"]
    assert len(summary["failed_runs"]) == 2
    assert any("failed, no result" in g for g in report.incomplete(summary))
    assert "failed, no result" in report.render_markdown(summary)


def test_gateway_cost_compares_a_with_its_direct_copy(tmp_path):
    rng = random.Random(6)
    for rnd in range(3):
        shift = rng.gauss(0, 1)
        for side, ttft in (("direct", 50.0), ("frontend", 55.0), ("epp", 56.1)):
            _write_run(tmp_path, side, rnd, ttft + shift + rng.gauss(0, 0.05))
    summary = report.build_summary(tmp_path)
    (point,) = summary["points"]
    (cost,) = summary["gateway_points"]
    assert point["sides"] == {"a": "frontend", "b": "epp"}
    assert cost["sides"] == {"a": "direct", "b": "frontend"}
    a_vs_b = point["latency"]["time_to_first_token"]["mean"]
    gateway = cost["latency"]["time_to_first_token"]["mean"]
    assert a_vs_b["delta_pct"] == pytest.approx(2, abs=1)
    assert gateway["delta_pct"] == pytest.approx(10, abs=1)
    assert gateway["verdict"] == "gateway worse"
    # The gate judges B against A only; the gateway's own cost is not B's.
    assert report.regressions(summary, 5.0) == []
    md = report.render_markdown(summary)
    assert "## Gateway cost" in md and "| Direct | Via gateway |" in md


def test_trend_orders_runs_and_marks_other_setups(tmp_path):
    for name, started, b_ttft, mocker_args in (
        ("late", "2026-02-01", 55.0, []),
        ("early", "2026-01-01", 50.0, ["--num-gpu-blocks-override", "4096"]),
    ):
        run = tmp_path / name
        for rnd in range(2):
            _write_run(run, "frontend", rnd, 50.0 + rnd)
            _write_run(run, "epp", rnd, b_ttft + rnd)
        meta = {"started": started, "config": {"mocker_extra_args": mocker_args}}
        (run / "meta.json").write_text(json.dumps(meta))
        (run / "summary.json").write_text(json.dumps(report.build_summary(run)))
    table = report.render_trend(tmp_path)
    rows = [
        line for line in table.splitlines() if line.startswith(("| early", "| late"))
    ]
    # In start order; a smaller KV pool is not comparable with the newest run.
    assert [r.split("|")[1].strip() for r in rows] == ["early †", "late"]
    assert "+9.9%" in rows[1]
