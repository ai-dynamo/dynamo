# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn a results directory into ``summary.json`` and ``report.md``.

Layout read here (written by ``cli.py``)::

    <results>/meta.json
    <results>/runs/<topology>/<side>/<load point>/r<round>.json

where a load point is ``c<concurrency>`` or ``rps<requests per second>``.
"""

from __future__ import annotations

import json
import math
import pathlib
from typing import Any

from . import stats
from .loadgen import LoadPoint

SIDE_A = "frontend"
SIDE_B = "gateway"
# Fewer rounds measured validly on both sides than this and no verdict is given.
MIN_ROUNDS = 2
GATED_STATS = ("mean", "p50", "p90")


def load_runs(
    results: pathlib.Path,
) -> dict[tuple[str, LoadPoint, str], list[dict[str, Any]]]:
    runs: dict[tuple[str, LoadPoint, str], list[dict[str, Any]]] = {}
    for path in sorted((results / "runs").glob("*/*/*/r*.json")):
        topology, side, label = path.parts[-4], path.parts[-3], path.parts[-2]
        run = json.loads(path.read_text())
        run.setdefault("round", int(path.stem[1:]))
        runs.setdefault((topology, LoadPoint.parse(label), side), []).append(run)
    return runs


def load_failures(results: pathlib.Path) -> list[tuple[str, LoadPoint, str, int]]:
    """Measurements that failed outright: an ``.error`` marker and no run file.

    A marker next to a run file is already covered: a valid run supersedes
    it, and an invalid one is listed with its errors under excluded runs.
    """
    out = []
    for marker in sorted((results / "runs").glob("*/*/*/r*.error")):
        if marker.with_suffix(".json").exists():
            continue
        topology, side, label = marker.parts[-4], marker.parts[-3], marker.parts[-2]
        out.append((topology, LoadPoint.parse(label), side, int(marker.stem[1:])))
    return out


def is_valid(run: dict[str, Any]) -> bool:
    """A run counts only if every request succeeded and none went missing."""
    expected = run.get("expected_requests")
    if run.get("errors", 0) > 0:
        return False
    if expected and run.get("records", 0) < expected:
        return False
    return bool(run.get("samples", {}).get("time_to_first_token"))


def build_summary(results: pathlib.Path, min_effect_pct: float = 2.0) -> dict[str, Any]:
    runs = load_runs(results)
    failures = load_failures(results)
    meta_path = results / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    points: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    # A point whose every run failed has only markers, and must still show.
    keys = sorted(
        {(t, lp) for (t, lp, _) in runs} | {(t, lp) for t, lp, _, _ in failures}
    )
    for topology, load in keys:
        sides = {}
        for side in (SIDE_A, SIDE_B):
            all_runs = runs.get((topology, load, side), [])
            for r in all_runs:
                if not is_valid(r):
                    invalid.append(
                        {
                            "topology": topology,
                            "load": load.describe(),
                            "side": side,
                            "round": r.get("round"),
                            "errors": r.get("errors"),
                            "records": r.get("records"),
                            "expected": r.get("expected_requests"),
                        }
                    )
            sides[side] = [r for r in all_runs if is_valid(r)]
        a_runs, b_runs = sides[SIDE_A], sides[SIDE_B]
        a_by_round = {r["round"]: r for r in a_runs}
        b_by_round = {r["round"]: r for r in b_runs}
        paired = sorted(set(a_by_round) & set(b_by_round))
        point: dict[str, Any] = {
            "topology": topology,
            "load": load.label,
            "load_desc": load.describe(),
            "rounds": {"a": len(a_runs), "b": len(b_runs), "paired": len(paired)},
            "ttft_round_cv_pct": {
                "a": _round_mean_cv(a_runs),
                "b": _round_mean_cv(b_runs),
            },
            "latency": {},
            "throughput": {},
        }
        # Rounds pair up by index: both sides replayed the same seed in them.
        for metric in stats.LATENCY_METRICS:
            rows = stats.compare_latency(
                {r: a_by_round[r]["samples"].get(metric, []) for r in paired},
                {r: b_by_round[r]["samples"].get(metric, []) for r in paired},
                min_effect_pct=min_effect_pct,
            )
            if rows:
                point["latency"][metric] = rows
        for metric in stats.THROUGHPUT_METRICS:
            row = stats.compare_scalar(
                {r: a_by_round[r].get(metric) for r in paired},
                {r: b_by_round[r].get(metric) for r in paired},
                higher_is_better=True,
                min_effect_pct=min_effect_pct,
            )
            if row is None:
                continue
            if load.kind == "rps":
                # Below saturation an open-loop run delivers what it offers,
                # so throughput cannot tell the sides apart.
                row["verdict"] = "n/a (open loop)"
            point["throughput"][metric] = row
        for row in _rows(point):
            if row["rounds"] < MIN_ROUNDS:
                # One pair has no spread to estimate an interval from.
                row["verdict"] = "n/a (too few rounds)"
        points.append(point)
    return {
        "meta": meta,
        "min_effect_pct": min_effect_pct,
        "points": points,
        "invalid_runs": invalid,
        "failed_runs": [
            {"topology": t, "load": lp.describe(), "side": side, "round": rnd}
            for t, lp, side, rnd in failures
        ],
    }


def _rows(point: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for rows in point["latency"].values() for row in rows.values()] + list(
        point["throughput"].values()
    )


def json_safe(value: Any) -> Any:
    """``value`` with NaN and infinities replaced by None, for strict JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def _round_mean_cv(runs: list[dict[str, Any]]) -> float:
    """Run-to-run noise: the CV of each round's mean TTFT, in percent."""
    means = [
        sum(t) / len(t)
        for t in (r["samples"].get("time_to_first_token") for r in runs)
        if t
    ]
    return stats.cv(means)


def incomplete(summary: dict[str, Any]) -> list[str]:
    """Points that lack a comparison, and runs excluded as invalid."""
    out = [
        f"{p['topology']} {p['load_desc']}: no comparison "
        f"(valid rounds A={p['rounds']['a']} B={p['rounds']['b']}, "
        f"{p['rounds'].get('paired', 0)} on both sides)"
        for p in summary["points"]
        if not p["latency"]
    ]
    out += [
        f"{r['topology']} {r['load']} {r['side']} round {r['round']}: excluded "
        f"(errors={r['errors']} records={r['records']} expected={r['expected']})"
        for r in summary["invalid_runs"]
    ]
    out += [
        f"{r['topology']} {r['load']} {r['side']} round {r['round']}: failed, no result"
        for r in summary.get("failed_runs", [])
    ]
    return out


def regressions(summary: dict[str, Any], max_pct: float) -> list[str]:
    """TTFT points where B is worse than A by more than ``max_pct`` with confidence.

    Uses the lower end of the 95% interval, so a delta within the run-to-run
    noise does not trip it, and gates only the mean, p50, and p90: p99 over a
    few rounds is noisy enough that testing it as well would trip the gate on
    A/A runs.
    """
    out = []
    for p in summary["points"]:
        ttft = p["latency"].get("time_to_first_token", {})
        for q, row in ttft.items():
            if q not in GATED_STATS or row["verdict"].startswith("n/a"):
                continue
            lo = row["ci_pct"][0]
            if not math.isnan(lo) and lo > max_pct:
                out.append(
                    f"{p['topology']} {p['load_desc']} TTFT {q}: "
                    f"B {row['delta_pct']:+.1f}% vs A (95% CI {lo:+.1f}%..{row['ci_pct'][1]:+.1f}%)"
                )
    return out


def ungated(summary: dict[str, Any]) -> list[str]:
    """Gated TTFT statistics the regression gate could not evaluate."""
    out = []
    for p in summary["points"]:
        for q, row in p["latency"].get("time_to_first_token", {}).items():
            if q in GATED_STATS and row["verdict"].startswith("n/a"):
                out.append(
                    f"{p['topology']} {p['load_desc']} TTFT {q}: {row['verdict']}"
                )
    return out


def render_markdown(summary: dict[str, Any]) -> str:
    meta = summary.get("meta", {})
    lines = ["# Frontend vs Gateway+EPP A/B", ""]
    if meta.get("backend") == "local":
        side_b = (
            "B = Envoy + EPP (KV router, Dynamo runtime mode, static endpoint) + "
            "one direct-mode Frontend, in Docker without Kubernetes. "
        )
    else:
        side_b = (
            "B = Gateway + EPP (KV router, Dynamo runtime mode) + "
            "direct-mode Frontend sidecars. "
        )
    lines += [
        "A = Dynamo Frontend (KV router), no EPP. "
        + side_b
        + "Workers are mocker in both, built from the same spec.",
        "",
        "Each round gives one value per side; A and B are their geometric means "
        "over the rounds measured on both sides. Δ is B relative to A; negative "
        "latency Δ means B is faster. Intervals are 95% paired t-intervals over "
        "the per-round B/A ratios (both sides replay the same seed each round). "
        f"`same` means the interval lies within ±{summary['min_effect_pct']:g}%; "
        "`inconclusive` means it includes zero but reaches past that.",
        "",
    ]
    if meta:
        lines += _meta_table(meta)
    by_topology: dict[str, list[dict[str, Any]]] = {}
    for p in summary["points"]:
        by_topology.setdefault(p["topology"], []).append(p)
    for topology, pts in by_topology.items():
        lines += [f"## {topology}", ""]
        lines += [
            "| Load | Metric | A | B | Δ | 95% CI | Verdict |",
            "|---:|---|---:|---:|---:|---|---|",
        ]
        for p in pts:
            c = p["load_desc"]
            if not p["latency"] and not p["throughput"]:
                lines.append(
                    f"| {c} | — | — | — | — | — | missing data "
                    f"(valid rounds A={p['rounds']['a']} B={p['rounds']['b']}) |"
                )
                continue
            for metric, label in stats.LATENCY_METRICS.items():
                for q, row in p["latency"].get(metric, {}).items():
                    lines.append(_row(c, f"{label} {q} (ms)", row, "{:.2f}"))
            for metric, label in stats.THROUGHPUT_METRICS.items():
                row = p["throughput"].get(metric)
                if row:
                    lines.append(_row(c, label, row, "{:.1f}"))
        lines.append("")
        noise = [(p["load_desc"], p["ttft_round_cv_pct"]) for p in pts if p["latency"]]
        if noise:
            lines += [
                "Run-to-run noise (coefficient of variation of the per-round TTFT mean):",
                "",
            ]
            for c, cv in noise:
                lines.append(f"- {c}: A {_fmt_pct(cv['a'])}, B {_fmt_pct(cv['b'])}")
            lines.append("")
    failed = summary.get("failed_runs", [])
    if summary["invalid_runs"] or failed:
        lines += ["## Excluded runs", ""]
        for r in summary["invalid_runs"]:
            lines.append(
                f"- {r['topology']} {r['load']} {r['side']} round {r['round']}: "
                f"errors={r['errors']} records={r['records']} expected={r['expected']}"
            )
        for r in failed:
            lines.append(
                f"- {r['topology']} {r['load']} {r['side']} round {r['round']}: "
                "failed, no result"
            )
        lines.append("")
    return "\n".join(lines)


def render_trend(
    history: pathlib.Path,
    metric: str = "time_to_first_token",
    quantile: str = "p50",
) -> str:
    """B's delta for one metric across every archived run under ``history``.

    Each subdirectory holding a ``summary.json`` is one run, ordered by its
    start time. Watching this column drift catches slow regressions that stay
    under a per-run threshold. A run measured with another backend, workload,
    or worker shape than the newest one is marked, since its deltas are not
    comparable with the rest of its column.
    """
    runs = []
    for path in history.glob("*/summary.json"):
        summary = json.loads(path.read_text())
        if not all("load" in p for p in summary.get("points", [])):
            continue  # written by an older harness without load-point labels
        started = summary.get("meta", {}).get("started", path.parent.name)
        runs.append((started, path.parent.name, summary))
    runs.sort(key=lambda r: r[0])
    newest = _setup(runs[-1][2]) if runs else None
    columns = sorted(
        {
            (p["topology"], LoadPoint.parse(p["load"]))
            for _, _, s in runs
            for p in s["points"]
        }
    )
    label = f"{stats.LATENCY_METRICS.get(metric, metric)} {quantile} Δ (B vs A)"
    lines = [
        f"# {label}",
        "",
        "| Run | Commit | "
        + " | ".join(f"{t} {lp.describe()}" for t, lp in columns)
        + " |",
        "|---|---|" + "---:|" * len(columns),
    ]
    for started, name, summary in runs:
        cells = {}
        for p in summary["points"]:
            row = p["latency"].get(metric, {}).get(quantile)
            if row:
                key = (p["topology"], LoadPoint.parse(p["load"]))
                cells[key] = f"{row['delta_pct']:+.1f}%"
        commit = summary.get("meta", {}).get("git_sha", "")
        mark = "" if _setup(summary) == newest else " †"
        lines.append(
            f"| {name}{mark} | {commit} | "
            + " | ".join(cells.get(col, "—") for col in columns)
            + " |"
        )
    if any(_setup(s) != newest for _, _, s in runs):
        lines += [
            "",
            "† Measured with a different backend, workload, or worker shape than "
            "the newest run; not comparable with the rest of its column.",
        ]
    return "\n".join(lines)


def _setup(summary: dict[str, Any]) -> tuple[str, str, str]:
    meta = summary.get("meta", {})
    return (
        meta.get("backend", "kubernetes"),
        json.dumps(meta.get("workload"), sort_keys=True),
        json.dumps(meta.get("workers"), sort_keys=True),
    )


def _row(load: str, label: str, row: dict[str, Any], fmt: str) -> str:
    lo, hi = row["ci_pct"]
    return (
        f"| {load} | {label} | {fmt.format(row['a'])} | {fmt.format(row['b'])} | "
        f"{row['delta_pct']:+.1f}% | {lo:+.1f}%..{hi:+.1f}% | {row['verdict']} |"
    )


def _fmt_pct(v: float) -> str:
    return "n/a" if v is None or math.isnan(v) else f"{v:.1f}%"


def _meta_table(meta: dict[str, Any]) -> list[str]:
    keys = [
        ("started", "Started"),
        ("context", "Cluster context"),
        ("git_sha", "Repository commit"),
        ("frontend_image", "Frontend / EPP image"),
        ("epp_binary", "EPP binary"),
        ("worker_image", "Mocker image"),
        ("aiperf_version", "aiperf"),
        ("model", "Model (tokenizer)"),
        ("workers", "Workers"),
        ("workload", "Workload"),
        ("rounds", "Rounds"),
        ("nodes", "Nodes"),
    ]
    out = ["| | |", "|---|---|"]
    for key, label in keys:
        if key in meta:
            out.append(f"| {label} | {_meta_value(meta[key])} |")
    out.append("")
    return out


def _meta_value(v: Any) -> str:
    if isinstance(v, dict):
        return ", ".join(f"{k}={val}" for k, val in v.items())
    if isinstance(v, list):
        return "; ".join(_meta_value(x) for x in v)
    return str(v)
