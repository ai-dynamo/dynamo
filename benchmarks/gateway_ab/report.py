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
# Fewer valid rounds per side than this and no verdict is given.
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
    """Measurements that failed outright: an ``.error`` marker, no valid run."""
    out = []
    for marker in sorted((results / "runs").glob("*/*/*/r*.error")):
        run_file = marker.with_suffix(".json")
        try:
            if is_valid(json.loads(run_file.read_text())):
                continue
        except (OSError, ValueError):
            pass
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


def build_summary(
    results: pathlib.Path,
    min_effect_pct: float = 2.0,
    bootstrap_iterations: int = 1000,
) -> dict[str, Any]:
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
        point: dict[str, Any] = {
            "topology": topology,
            "load": load.label,
            "load_desc": load.describe(),
            "rounds": {"a": len(a_runs), "b": len(b_runs)},
            "ttft_round_cv_pct": {
                "a": _round_mean_cv(a_runs),
                "b": _round_mean_cv(b_runs),
            },
            "latency": {},
            "throughput": {},
        }
        if a_runs and b_runs:
            for metric in stats.LATENCY_METRICS:
                rounds_a = [r["samples"].get(metric, []) for r in a_runs]
                rounds_b = [r["samples"].get(metric, []) for r in b_runs]
                if any(rounds_a) and any(rounds_b):
                    point["latency"][metric] = stats.compare_latency(
                        rounds_a,
                        rounds_b,
                        iterations=bootstrap_iterations,
                        min_effect_pct=min_effect_pct,
                    )
            for metric in stats.THROUGHPUT_METRICS:
                va = [r[metric] for r in a_runs if r.get(metric) is not None]
                vb = [r[metric] for r in b_runs if r.get(metric) is not None]
                if va and vb:
                    row = stats.compare_scalar(
                        va, vb, higher_is_better=True, min_effect_pct=min_effect_pct
                    )
                    if load.kind == "rps":
                        # Below saturation an open-loop run delivers what it
                        # offers, so throughput cannot tell the sides apart.
                        row["verdict"] = "n/a (open loop)"
                    point["throughput"][metric] = row
            if min(len(a_runs), len(b_runs)) < MIN_ROUNDS:
                # One round has no between-round variance to resample, so its
                # intervals are far too narrow to support a verdict.
                for rows in point["latency"].values():
                    for row in rows.values():
                        row["verdict"] = "n/a (1 round)"
                for row in point["throughput"].values():
                    row["verdict"] = "n/a (1 round)"
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
        f"(valid rounds A={p['rounds']['a']} B={p['rounds']['b']})"
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

    Uses the lower end of the interval, so noise alone cannot trip it, and
    gates only the mean, p50, and p90: p99 over a few rounds is noisy enough
    that testing it as well would trip the gate on A/A runs.
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


def render_markdown(summary: dict[str, Any]) -> str:
    meta = summary.get("meta", {})
    lines = ["# Frontend vs Gateway+EPP A/B", ""]
    lines += [
        "A = Dynamo Frontend (KV router), no EPP. "
        "B = Gateway + EPP (KV router, Dynamo runtime mode) + direct-mode Frontend sidecars. "
        "Workers are mocker in both, built from the same spec.",
        "",
        "Δ is B relative to A; negative latency Δ means B is faster. "
        "Intervals are 95% hierarchical-bootstrap (rounds, then requests). "
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
    under a per-run threshold.
    """
    runs = []
    for path in history.glob("*/summary.json"):
        summary = json.loads(path.read_text())
        if not all("load" in p for p in summary.get("points", [])):
            continue  # written by an older harness without load-point labels
        started = summary.get("meta", {}).get("started", path.parent.name)
        runs.append((started, path.parent.name, summary))
    runs.sort(key=lambda r: r[0])
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
        lines.append(
            f"| {name} | {commit} | "
            + " | ".join(cells.get(col, "—") for col in columns)
            + " |"
        )
    return "\n".join(lines)


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
