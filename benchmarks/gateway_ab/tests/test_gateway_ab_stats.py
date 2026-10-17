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


def _rounds(rng, n_rounds, n, mean, round_jitter=0.0):
    out = []
    for _ in range(n_rounds):
        shift = rng.gauss(0, round_jitter)
        out.append([max(0.1, rng.gauss(mean + shift, mean * 0.1)) for _ in range(n)])
    return out


def test_percentile_matches_linear_interpolation():
    data = [1.0, 2.0, 3.0, 4.0]
    assert stats.percentile(data, 0) == 1.0
    assert stats.percentile(data, 100) == 4.0
    assert stats.percentile(data, 50) == 2.5
    assert stats.percentile(data, 90) == pytest.approx(3.7)
    assert stats.percentile([5.0], 99) == 5.0
    with pytest.raises(ValueError):
        stats.percentile([], 50)


def test_pct_delta():
    assert stats.pct_delta(100, 110) == pytest.approx(10)
    assert stats.pct_delta(100, 90) == pytest.approx(-10)
    assert stats.pct_delta(0, 0) == 0
    assert math.isinf(stats.pct_delta(0, 1))


def test_bootstrap_detects_real_shift():
    rng = random.Random(1)
    a = _rounds(rng, 3, 300, 50.0)
    b = _rounds(rng, 3, 300, 60.0)
    lo, hi = stats.hierarchical_bootstrap_delta(a, b, 50, iterations=300)
    assert lo < 20 < hi
    assert lo > 2


def test_bootstrap_identical_distributions_are_same():
    rng = random.Random(2)
    a = _rounds(rng, 3, 300, 50.0)
    b = _rounds(rng, 3, 300, 50.0)
    res = stats.compare_latency(a, b, iterations=300)
    assert res["mean"]["verdict"] == "same"
    # A noisier statistic may not resolve to "same", but it must not claim
    # a difference that is not there.
    assert res["p50"]["verdict"] in ("same", "inconclusive")
    lo, hi = res["p50"]["ci_pct"]
    assert lo < 0 < hi


def test_round_level_noise_widens_interval():
    """Between-round jitter must show up in the interval, not be averaged away."""
    quiet = random.Random(3)
    noisy = random.Random(3)
    a_q, b_q = _rounds(quiet, 3, 300, 50.0), _rounds(quiet, 3, 300, 50.0)
    a_n, b_n = _rounds(noisy, 3, 300, 50.0, 8.0), _rounds(noisy, 3, 300, 50.0, 8.0)
    lo_q, hi_q = stats.hierarchical_bootstrap_delta(a_q, b_q, 50, iterations=300)
    lo_n, hi_n = stats.hierarchical_bootstrap_delta(a_n, b_n, 50, iterations=300)
    assert (hi_n - lo_n) > 2 * (hi_q - lo_q)


@pytest.mark.parametrize(
    "ci,higher,expected",
    [
        ((5.0, 9.0), False, "B worse"),
        ((-9.0, -5.0), False, "B better"),
        ((5.0, 9.0), True, "B better"),
        ((-1.0, 1.0), False, "same"),
        ((-1.0, 5.0), False, "inconclusive"),
        ((-20.0, 1.0), True, "inconclusive"),
        ((0.5, 1.5), False, "B slightly worse"),
        ((-1.5, -0.5), True, "B slightly worse"),
        ((math.nan, math.nan), False, "n/a"),
    ],
)
def test_verdict(ci, higher, expected):
    assert stats.verdict(ci, 2.0, higher) == expected


def test_compare_scalar():
    res = stats.compare_scalar([100, 102, 98], [90, 91, 89], higher_is_better=True)
    assert res["a"] == 100 and res["b"] == 90
    assert res["verdict"] == "B worse"
    assert res["a_cv_pct"] == pytest.approx(2.0)


def _write_run(root, topology, side, conc, rnd, samples, **extra):
    path = root / "runs" / topology / side / f"c{conc}" / f"r{rnd}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    run = {
        "records": len(samples),
        "errors": 0,
        "expected_requests": len(samples),
        "request_throughput": 10.0,
        "output_token_throughput": 1000.0,
        "samples": {
            "time_to_first_token": samples,
            "inter_token_latency": [5.0] * len(samples),
            "request_latency": [s + 500 for s in samples],
        },
    }
    run.update(extra)
    path.write_text(json.dumps(run))


def test_summary_and_regression(tmp_path):
    rng = random.Random(4)
    for rnd in range(3):
        _write_run(
            tmp_path, "agg", "frontend", 8, rnd, [rng.gauss(50, 2) for _ in range(200)]
        )
        _write_run(
            tmp_path, "agg", "gateway", 8, rnd, [rng.gauss(65, 2) for _ in range(200)]
        )
    # An invalid run is reported and excluded, not averaged in.
    _write_run(tmp_path, "agg", "gateway", 8, 3, [1.0] * 200, errors=5)
    (tmp_path / "meta.json").write_text(json.dumps({"model": "m"}))

    summary = report.build_summary(tmp_path, bootstrap_iterations=200)
    (point,) = summary["points"]
    assert point["rounds"] == {"a": 3, "b": 3}
    ttft = point["latency"]["time_to_first_token"]["p50"]
    assert ttft["verdict"] == "B worse"
    assert ttft["delta_pct"] == pytest.approx(30, abs=3)
    assert len(summary["invalid_runs"]) == 1

    assert report.regressions(summary, 10.0)
    assert not report.regressions(summary, 50.0)

    md = report.render_markdown(summary)
    assert "## agg" in md and "TTFT p50" in md and "Excluded runs" in md


def test_summary_with_one_side_missing(tmp_path):
    _write_run(tmp_path, "disagg", "frontend", 4, 0, [10.0] * 50)
    summary = report.build_summary(tmp_path)
    (point,) = summary["points"]
    assert point["latency"] == {}
    assert "missing data" in report.render_markdown(summary)


def test_trend_orders_runs_and_fills_gaps(tmp_path):
    rng = random.Random(5)
    for name, started, b_mean in (
        ("late", "2026-02-01", 55),
        ("early", "2026-01-01", 50),
    ):
        run = tmp_path / name
        for rnd in range(2):
            _write_run(
                run, "agg", "frontend", 8, rnd, [rng.gauss(50, 1) for _ in range(100)]
            )
            _write_run(
                run,
                "agg",
                "gateway",
                8,
                rnd,
                [rng.gauss(b_mean, 1) for _ in range(100)],
            )
        (run / "meta.json").write_text(
            json.dumps({"started": started, "git_sha": name})
        )
        summary = report.build_summary(run, bootstrap_iterations=50)
        (run / "summary.json").write_text(json.dumps(summary))
    # A run directory without a summary is ignored.
    (tmp_path / "incomplete").mkdir()

    table = report.render_trend(tmp_path)
    rows = [
        line
        for line in table.splitlines()
        if line.startswith("| early") or line.startswith("| late")
    ]
    assert [r.split("|")[1].strip() for r in rows] == ["early", "late"]
    assert "+10." in rows[1] or "+9." in rows[1]


def test_mean_reported_next_to_percentiles(tmp_path):
    # Bimodal latencies: the median sits on one mode, the mean between them.
    fast_slow = [40.0] * 120 + [80.0] * 80
    for rnd in range(2):
        _write_run(tmp_path, "agg", "frontend", 8, rnd, fast_slow)
        _write_run(tmp_path, "agg", "gateway", 8, rnd, fast_slow)
    (point,) = report.build_summary(tmp_path, bootstrap_iterations=50)["points"]
    ttft = point["latency"]["time_to_first_token"]
    assert list(ttft) == ["mean", "p50", "p90", "p99"]
    assert ttft["mean"]["a"] == pytest.approx(56.0)
    assert ttft["p50"]["a"] == 40.0
    assert "TTFT mean (ms)" in report.render_markdown(
        report.build_summary(tmp_path, bootstrap_iterations=50)
    )


def test_single_round_gets_no_verdict_and_no_gate(tmp_path):
    _write_run(tmp_path, "agg", "frontend", 8, 0, [50.0] * 100)
    _write_run(tmp_path, "agg", "gateway", 8, 0, [80.0] * 100)
    summary = report.build_summary(tmp_path, bootstrap_iterations=50)
    (point,) = summary["points"]
    assert point["latency"]["time_to_first_token"]["p50"]["verdict"] == "n/a (1 round)"
    assert report.regressions(summary, 10.0) == []


def test_incomplete_flags_missing_and_excluded(tmp_path):
    _write_run(tmp_path, "agg", "frontend", 8, 0, [50.0] * 100)
    _write_run(tmp_path, "agg", "gateway", 8, 0, [50.0] * 100, errors=100)
    gaps = report.incomplete(report.build_summary(tmp_path, bootstrap_iterations=50))
    assert len(gaps) == 2
    assert "no comparison" in gaps[0] and "excluded" in gaps[1]


def test_trend_skips_summaries_without_load_labels(tmp_path):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "summary.json").write_text(
        json.dumps({"meta": {}, "points": [{"topology": "agg", "concurrency": 8}]})
    )
    assert "legacy" not in report.render_trend(tmp_path)
