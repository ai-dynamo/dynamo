# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Final-metric rank tables from a published Match Config results JSON.

For one SLA target (default: the first one in the matrix) the report is
reduced to two tables, one per workload set, on the single evaluation metric:

* metric: ``goodput_per_gpu`` — good requests per second per average GPU,
  i.e. goodput per GPU-hour up to the constant 3600 (identical ordering).
  ``cloudai-rl`` is the RL entry (named in the table header). ``good_rate`` is
  deliberately not ranked.
* workload sets: ``all`` (every workload in the matrix) and ``golden`` (the
  last six workloads in matrix order, i.e. the Astra Golden Set traces).

A workload is ranked only when every method has at least one ok repetition
for it; otherwise it is dropped from the table and listed in the header.

Each table has one row per method and two columns:

* **mean rank** — per workload, methods are ranked on the metric averaged over
  repetitions (rank 1 = best, ties share the average rank); the column is the
  mean of those ranks over the workload set. Lower is better.
* **pairwise win rate** — over every (workload, opponent) pair, the fraction
  of comparisons the method wins; a tie counts as half a win. Higher is better.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

METRICS: tuple[str, ...] = ("goodput_per_gpu",)
# The RL entries that belong in each metric's table (named in its header); RL
# entries not listed for a metric are dropped from that metric's table.
RL_ENTRIES_FOR_METRIC: dict[str, tuple[str, ...]] = {
    "goodput_per_gpu": ("cloudai-rl",),
}
RL_ENTRY_FOR_METRIC: dict[str, str] = {
    metric: entries[0] for metric, entries in RL_ENTRIES_FOR_METRIC.items()
}
RL_ENTRIES: frozenset[str] = frozenset(
    entry for entries in RL_ENTRIES_FOR_METRIC.values() for entry in entries
)
WORKLOAD_SETS: tuple[str, ...] = ("all", "golden")
GOLDEN_SET_SIZE = 6


def workload_order(report: Mapping[str, Any]) -> list[str]:
    """Workloads in first-appearance order, from the matrix (falls back to results)."""
    seen: dict[str, None] = {}
    for row in report.get("matrix") or report.get("results") or []:
        seen.setdefault(str(row["workload"]), None)
    return list(seen)


def sla_order(report: Mapping[str, Any]) -> list[str]:
    seen: dict[str, None] = {}
    for row in report.get("matrix") or report.get("results") or []:
        seen.setdefault(str(row["sla"]), None)
    return list(seen)


def select_workloads(report: Mapping[str, Any], choice: str) -> list[str]:
    order = workload_order(report)
    if choice == "all":
        return order
    if choice == "golden":
        return order[-GOLDEN_SET_SIZE:]
    raise ValueError(
        f"unknown workload set {choice!r}; expected one of {WORKLOAD_SETS}"
    )


def methods_for_metric(autoscalers: Iterable[str], metric: str) -> list[str]:
    """Every method except the RL entries that do not belong to this metric."""
    keep = RL_ENTRIES_FOR_METRIC.get(metric, ())
    return [a for a in autoscalers if a not in RL_ENTRIES or a in keep]


def metric_table(
    results: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    sla: str,
    workloads: Sequence[str],
    methods: Sequence[str],
) -> dict[str, dict[str, float]]:
    """``{workload: {method: metric averaged over repetitions}}`` for ok rows."""
    samples: dict[str, dict[str, list[float]]] = {
        w: {m: [] for m in methods} for w in workloads
    }
    for row in results:
        if row.get("status") != "ok" or row.get("sla") != sla:
            continue
        w, m = row.get("workload"), row.get("autoscaler")
        if w in samples and m in samples[w]:
            value = (row.get("metrics") or {}).get(metric)
            if isinstance(value, (int, float)):
                samples[w][m].append(float(value))
    table: dict[str, dict[str, float]] = {}
    for w in workloads:
        # A workload is ranked only when every method has at least one ok
        # repetition there; otherwise it is dropped from this table (and
        # reported) rather than ranking the remaining methods on a smaller
        # field, which would bias the mean ranks.
        if any(not samples[w][m] for m in methods):
            continue
        table[w] = {m: statistics.fmean(samples[w][m]) for m in methods}
    if not table:
        raise ValueError(
            f"no workload has ok {metric!r} rows for every method, sla {sla!r}"
        )
    return table


def average_ranks(values: Mapping[str, float]) -> dict[str, float]:
    """Rank 1 = highest value; tied values share the average of their positions."""
    ordered = sorted(values.items(), key=lambda kv: -kv[1])
    ranks: dict[str, float] = {}
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1][1] == ordered[i][1]:
            j += 1
        shared = (i + 1 + j + 1) / 2.0  # positions are 1-based
        for k in range(i, j + 1):
            ranks[ordered[k][0]] = shared
        i = j + 1
    return ranks


def mean_rank(
    table: Mapping[str, Mapping[str, float]], methods: Sequence[str]
) -> dict[str, float]:
    per_method: dict[str, list[float]] = {m: [] for m in methods}
    for row in table.values():
        for m, r in average_ranks(row).items():
            per_method[m].append(r)
    return {m: statistics.fmean(rs) for m, rs in per_method.items()}


def pairwise_win_rate(
    table: Mapping[str, Mapping[str, float]], methods: Sequence[str]
) -> dict[str, float]:
    """Fraction of (workload, opponent) comparisons won; a tie counts 0.5."""
    wins: dict[str, list[float]] = {m: [] for m in methods}
    for row in table.values():
        for a in methods:
            for b in methods:
                if a == b:
                    continue
                wins[a].append(
                    1.0 if row[a] > row[b] else 0.5 if row[a] == row[b] else 0.0
                )
    return {m: statistics.fmean(w) for m, w in wins.items()}


def rank_table(
    report: Mapping[str, Any], *, metric: str, workload_set: str, sla: str
) -> dict[str, Any]:
    results = report["results"]
    autoscalers = list(
        dict.fromkeys(str(r["autoscaler"]) for r in report.get("matrix") or results)
    )
    methods = methods_for_metric(autoscalers, metric)
    workloads = select_workloads(report, workload_set)
    table = metric_table(
        results, metric=metric, sla=sla, workloads=workloads, methods=methods
    )
    ranks = mean_rank(table, methods)
    win = pairwise_win_rate(table, methods)
    rows = sorted(methods, key=lambda m: (ranks[m], -win[m], m))
    ranked = [w for w in workloads if w in table]
    rl_entries = [e for e in RL_ENTRIES_FOR_METRIC.get(metric, ()) if e in methods]
    return {
        "metric": metric,
        "workload_set": workload_set,
        "workloads": ranked,
        "dropped_workloads": [w for w in workloads if w not in table],
        "sla": sla,
        "rl_entries": rl_entries,
        "rl_entry": rl_entries[0] if rl_entries else None,
        "rows": [
            {"method": m, "mean_rank": ranks[m], "pairwise_win_rate": win[m]}
            for m in rows
        ],
    }


def all_rank_tables(
    report: Mapping[str, Any], *, sla: str | None = None
) -> list[dict[str, Any]]:
    chosen = sla or sla_order(report)[0]
    return [
        rank_table(report, metric=metric, workload_set=ws, sla=chosen)
        for metric in METRICS
        for ws in WORKLOAD_SETS
    ]


def format_markdown(tables: Sequence[Mapping[str, Any]]) -> str:
    out: list[str] = []
    for t in tables:
        header = (
            f"### {t['metric']} — {t['workload_set']} workloads "
            f"({len(t['workloads'])}), SLA `{t['sla']}`"
        )
        rl_entries = t.get("rl_entries") or (
            [t["rl_entry"]] if t.get("rl_entry") else []
        )
        if rl_entries:
            label = "RL entry" if len(rl_entries) == 1 else "RL entries"
            header += f", {label} " + ", ".join(f"`{e}`" for e in rl_entries)
        out.append(header)
        if t.get("dropped_workloads"):
            out.append("")
            out.append(
                "Dropped (no ok repetition for every method): "
                + ", ".join(f"`{w}`" for w in t["dropped_workloads"])
            )
        out.append("")
        out.append("| Method | Mean rank | Pairwise win rate |")
        out.append("|---|---:|---:|")
        for r in t["rows"]:
            out.append(
                f"| {r['method']} | {r['mean_rank']:.2f} | {r['pairwise_win_rate']:.3f} |"
            )
        out.append("")
    out.append(
        "Mean rank: 1 = best per workload, ties share the average rank, lower is better. "
        "Pairwise win rate: share of (workload, opponent) comparisons won, ties count 0.5, higher is better. "
        "Metric values are averaged over repetitions before ranking."
    )
    return "\n".join(out) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", help="published Match Config results JSON")
    parser.add_argument(
        "--sla", help="SLA target name (default: the first one in the matrix)"
    )
    parser.add_argument("--out", help="also write the markdown tables to this file")
    parser.add_argument(
        "--json", dest="json_out", help="also write the tables as JSON to this file"
    )
    args = parser.parse_args(argv)

    report = json.loads(Path(args.results).read_text())
    slas = sla_order(report)
    if args.sla is not None and args.sla not in slas:
        parser.error(f"unknown --sla {args.sla!r}; available: {', '.join(slas)}")
    tables = all_rank_tables(report, sla=args.sla)
    text = format_markdown(tables)
    sys.stdout.write(text)
    if args.out:
        Path(args.out).write_text(text)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(tables, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
