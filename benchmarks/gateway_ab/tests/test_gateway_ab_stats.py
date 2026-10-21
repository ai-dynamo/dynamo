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
    """Per-round request samples, keyed by round index."""
    out = {}
    for r in range(n_rounds):
        shift = rng.gauss(0, round_jitter)
        out[r] = [max(0.1, rng.gauss(mean + shift, mean * 0.1)) for _ in range(n)]
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


@pytest.mark.parametrize(
    "df,expected",
    [(1, 12.7062047), (2, 4.30265273), (3, 3.18244631), (10, 2.22813885)],
)
def test_t_quantile_matches_tables(df, expected):
    assert stats.t_quantile(0.975, df) == pytest.approx(expected, abs=1e-6)


def test_detects_real_shift():
    rng = random.Random(1)
    a = _rounds(rng, 3, 300, 50.0)
    b = _rounds(rng, 3, 300, 60.0)
    p50 = stats.compare_latency(a, b)["p50"]
    lo, hi = p50["ci_pct"]
    assert lo < 20 < hi
    assert p50["verdict"] == "B worse"


def test_identical_distributions_are_not_called_different():
    rng = random.Random(2)
    a = _rounds(rng, 5, 300, 50.0)
    b = _rounds(rng, 5, 300, 50.0)
    res = stats.compare_latency(a, b)
    for q in ("mean", "p50"):
        lo, hi = res[q]["ci_pct"]
        assert lo < 0 < hi
        assert res[q]["verdict"] in ("same", "inconclusive")


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


def test_noise_both_sides_share_cancels_and_unshared_noise_widens():
    """Pairing removes round effects common to A and B, not the rest."""
    rng = random.Random(3)
    shared = [rng.gauss(0, 5.0) for _ in range(3)]
    a = {r: [50 + shared[r]] * 50 for r in range(3)}
    b = {r: [55 + shared[r] * 1.1] * 50 for r in range(3)}
    lo, hi = stats.compare_latency(a, b)["mean"]["ci_pct"]
    assert lo > 5 and hi - lo < 4
    unshared = {r: [55 + rng.gauss(0, 5.0)] * 50 for r in range(3)}
    lo_u, hi_u = stats.compare_latency(a, unshared)["mean"]["ci_pct"]
    assert hi_u - lo_u > 2 * (hi - lo)


def test_rounds_pair_by_index_and_unmatched_rounds_are_left_out():
    a = {0: [50.0], 1: [52.0], 2: [51.0]}
    b = {0: [55.0], 2: [56.1], 3: [99.0]}
    row = stats.compare_latency(a, b, quantiles=("mean",))["mean"]
    assert row["rounds"] == 2
    assert row["delta_pct"] == pytest.approx(10, abs=0.01)


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
    res = stats.compare_scalar(
        {0: 100, 1: 102, 2: 98}, {0: 90, 1: 91, 2: 89}, higher_is_better=True
    )
    assert res["a"] == pytest.approx(100, abs=0.1)
    assert res["b"] == pytest.approx(90, abs=0.1)
    assert res["verdict"] == "B worse"
    assert res["a_cv_pct"] == pytest.approx(2.0)
    # Columns and delta agree: delta is B against A as the columns show.
    assert res["delta_pct"] == pytest.approx(stats.pct_delta(res["a"], res["b"]))


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
            tmp_path, "agg", "epp", 8, rnd, [rng.gauss(65, 2) for _ in range(200)]
        )
    # An invalid run is reported and excluded, not averaged in.
    _write_run(tmp_path, "agg", "epp", 8, 3, [1.0] * 200, errors=5)
    (tmp_path / "meta.json").write_text(json.dumps({"model": "m"}))

    summary = report.build_summary(tmp_path)
    (point,) = summary["points"]
    assert point["rounds"] == {"a": 3, "b": 3, "paired": 3}
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
                "epp",
                8,
                rnd,
                [rng.gauss(b_mean, 1) for _ in range(100)],
            )
        (run / "meta.json").write_text(
            json.dumps({"started": started, "git_sha": name})
        )
        summary = report.build_summary(run)
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
        _write_run(tmp_path, "agg", "epp", 8, rnd, fast_slow)
    (point,) = report.build_summary(tmp_path)["points"]
    ttft = point["latency"]["time_to_first_token"]
    assert list(ttft) == ["mean", "p50", "p90", "p99"]
    assert ttft["mean"]["a"] == pytest.approx(56.0)
    assert ttft["p50"]["a"] == 40.0
    assert "TTFT mean (ms)" in report.render_markdown(report.build_summary(tmp_path))


def test_single_round_gets_no_verdict_and_no_gate(tmp_path):
    _write_run(tmp_path, "agg", "frontend", 8, 0, [50.0] * 100)
    _write_run(tmp_path, "agg", "epp", 8, 0, [80.0] * 100)
    summary = report.build_summary(tmp_path)
    (point,) = summary["points"]
    p50 = point["latency"]["time_to_first_token"]["p50"]
    assert p50["verdict"] == "n/a (too few rounds)"
    assert report.regressions(summary, 10.0) == []
    # The gate cannot pass what it could not evaluate.
    assert len(report.ungated(summary)) == len(report.GATED_STATS)


def test_incomplete_flags_missing_and_excluded(tmp_path):
    _write_run(tmp_path, "agg", "frontend", 8, 0, [50.0] * 100)
    _write_run(tmp_path, "agg", "epp", 8, 0, [50.0] * 100, errors=100)
    gaps = report.incomplete(report.build_summary(tmp_path))
    assert len(gaps) == 2
    assert "no comparison" in gaps[0] and "excluded" in gaps[1]


def test_trend_skips_summaries_without_load_labels(tmp_path):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "summary.json").write_text(
        json.dumps({"meta": {}, "points": [{"topology": "agg", "concurrency": 8}]})
    )
    assert "legacy" not in report.render_trend(tmp_path)


def _write_rps_run(root, side, rnd, samples):
    path = root / "runs" / "agg" / side / "rps6" / f"r{rnd}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "records": len(samples),
                "errors": 0,
                "expected_requests": len(samples),
                "request_throughput": 6.0,
                "output_token_throughput": 760.0,
                "samples": {
                    "time_to_first_token": samples,
                    "inter_token_latency": [5.0] * len(samples),
                    "request_latency": [s + 500 for s in samples],
                },
            }
        )
    )


def test_open_loop_throughput_gets_no_verdict(tmp_path):
    for rnd in range(2):
        _write_rps_run(tmp_path, "frontend", rnd, [50.0] * 50)
        _write_rps_run(tmp_path, "epp", rnd, [52.0] * 50)
    (point,) = report.build_summary(tmp_path)["points"]
    assert (
        point["throughput"]["output_token_throughput"]["verdict"] == "n/a (open loop)"
    )
    assert (
        point["latency"]["time_to_first_token"]["mean"]["verdict"] != "n/a (open loop)"
    )


def test_point_with_only_failure_markers_is_reported(tmp_path):
    for side in ("frontend", "epp"):
        marker = tmp_path / "runs" / "disagg" / side / "c8" / "r0.error"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("failed twice\n")
    summary = report.build_summary(tmp_path)
    (point,) = summary["points"]
    assert point["topology"] == "disagg" and not point["latency"]
    assert len(summary["failed_runs"]) == 2
    gaps = report.incomplete(summary)
    assert any("failed, no result" in g for g in gaps)
    assert "failed, no result" in report.render_markdown(summary)


def test_failure_marker_superseded_by_valid_run_is_ignored(tmp_path):
    _write_run(tmp_path, "agg", "frontend", 8, 0, [50.0] * 10)
    (tmp_path / "runs/agg/frontend/c8/r0.error").write_text("old\n")
    assert report.load_failures(tmp_path) == []


def test_noise_line_uses_per_round_ttft_means(tmp_path):
    for rnd, mean in enumerate((50.0, 60.0)):
        _write_run(tmp_path, "agg", "frontend", 8, rnd, [mean] * 20)
        _write_run(tmp_path, "agg", "epp", 8, rnd, [55.0] * 20)
    (point,) = report.build_summary(tmp_path)["points"]
    assert point["ttft_round_cv_pct"]["a"] == pytest.approx(12.856, abs=0.01)
    assert point["ttft_round_cv_pct"]["b"] == 0.0


def test_report_header_names_the_backend(tmp_path):
    _write_run(tmp_path, "agg", "frontend", 8, 0, [50.0] * 10)
    _write_run(tmp_path, "agg", "epp", 8, 0, [50.0] * 10)
    (tmp_path / "meta.json").write_text(json.dumps({"backend": "local"}))
    md = report.render_markdown(report.build_summary(tmp_path))
    assert "without Kubernetes" in md and "sidecars" not in md
    (tmp_path / "meta.json").write_text(json.dumps({"backend": "kubernetes"}))
    md = report.render_markdown(report.build_summary(tmp_path))
    assert "sidecars" in md


def test_run_failing_validation_twice_is_listed_once(tmp_path):
    _write_run(tmp_path, "agg", "frontend", 8, 0, [50.0] * 10, errors=3)
    (tmp_path / "runs/agg/frontend/c8/r0.error").write_text("failed twice\n")
    summary = report.build_summary(tmp_path)
    assert len(summary["invalid_runs"]) == 1
    assert summary["failed_runs"] == []


def test_trend_marks_runs_with_another_setup(tmp_path):
    for name, started, isl in (
        ("old", "2026-01-01", 4096),
        ("new", "2026-02-01", 1024),
    ):
        run = tmp_path / name
        for rnd in range(2):
            _write_run(run, "agg", "frontend", 8, rnd, [50.0 + rnd] * 20)
            _write_run(run, "agg", "epp", 8, rnd, [55.0 + rnd] * 20)
        (run / "meta.json").write_text(
            json.dumps({"started": started, "workload": {"isl": isl}})
        )
        (run / "summary.json").write_text(json.dumps(report.build_summary(run)))
    table = report.render_trend(tmp_path)
    assert "| old † |" in table and "| new |" in table
    assert "not comparable" in table


def test_summary_json_is_strict(tmp_path):
    _write_run(tmp_path, "agg", "frontend", 8, 0, [50.0] * 10)
    _write_run(tmp_path, "agg", "epp", 8, 0, [55.0] * 10)
    summary = report.build_summary(tmp_path)
    text = json.dumps(report.json_safe(summary), allow_nan=False)
    assert json.loads(text)["points"][0]["ttft_round_cv_pct"]["a"] is None


def test_verdict_names_its_subject():
    assert stats.verdict((5.0, 9.0), 2.0, False, "gateway") == "gateway worse"
    assert stats.verdict((0.5, 1.5), 2.0, False, "gateway") == "gateway slightly worse"
    assert stats.verdict((-1.0, 1.0), 2.0, False, "gateway") == "same"


def test_gateway_cost_compares_a_with_its_direct_copy(tmp_path):
    rng = random.Random(6)
    for rnd in range(3):
        shift = rng.gauss(0, 1)
        for side, mean in (("direct", 50.0), ("frontend", 55.0), ("epp", 56.1)):
            samples = [rng.gauss(mean + shift, 0.5) for _ in range(200)]
            _write_run(tmp_path, "agg", side, 8, rnd, samples)
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
    assert report.incomplete(summary) == []


def test_gateway_cost_appears_only_with_the_direct_arm(tmp_path):
    for rnd in range(2):
        _write_run(tmp_path, "agg", "frontend", 8, rnd, [50.0 + rnd] * 20)
        _write_run(tmp_path, "agg", "epp", 8, rnd, [51.0 + rnd] * 20)
    summary = report.build_summary(tmp_path)
    assert summary["gateway_points"] == []
    assert "Gateway cost" not in report.render_markdown(summary)
    # A direct arm that failed outright still leaves a gap to report.
    marker = tmp_path / "runs" / "agg" / "direct" / "c8" / "r0.error"
    marker.parent.mkdir(parents=True)
    marker.write_text("failed twice\n")
    gaps = report.incomplete(report.build_summary(tmp_path))
    assert any("no gateway-cost comparison" in g for g in gaps)


def test_trend_marks_runs_from_another_harness_version(tmp_path):
    for name, started, harness in (("v2", "2026-01-01", 2), ("v3", "2026-02-01", 3)):
        run = tmp_path / name
        for rnd in range(2):
            _write_run(run, "agg", "frontend", 8, rnd, [50.0 + rnd] * 20)
            _write_run(run, "agg", "epp", 8, rnd, [55.0 + rnd] * 20)
        meta = {"started": started, "identity": {"harness": harness}}
        (run / "meta.json").write_text(json.dumps(meta))
        (run / "summary.json").write_text(json.dumps(report.build_summary(run)))
    table = report.render_trend(tmp_path)
    assert "| v2 † |" in table and "| v3 |" in table
