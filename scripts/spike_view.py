#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
spike_view — render per-worker time-series sparklines from aiperf server-metrics output.

Reads a test-run output directory containing aiperf's server-metrics exports and
emits a markdown report (`spike_view.md`) with one sparkline per worker per metric.
Used for offline post-mortem of cascade-causality test runs.

Inputs (any subset; will use whichever exists):
  <run>/load/server_metrics_export.jsonl   — raw 1Hz time-series (preferred for sparklines)
  <run>/load/server_metrics_export.json    — aggregate stats (fallback summary)
  <run>/load/profile_export_aiperf.json    — request-side TTFT / latency summaries

Output:
  <run>/spike_view.md                      — markdown with system + per-worker sections

Metrics surfaced (whatever subset is in the data):
  - vllm:num_requests_running, vllm:num_requests_waiting (per worker, per dp_rank)
  - vllm:kv_cache_usage_perc
  - vllm:nixl_post_time_seconds (histogram → p99 estimate per timestep)
  - vllm:nixl_num_failed_transfers, vllm:nixl_num_kv_expired_reqs (counters)
  - dynamo_frontend_time_to_first_token_seconds (histogram → p99)
  - dynamo_frontend_request_duration_seconds (histogram → p99)
  - dynamo_frontend_queued_requests, dynamo_frontend_inflight_requests

Stdlib only — no pyarrow, no pandas. Sparklines via 8-row Unicode block chars.

Usage:
  scripts/spike_view.py <run-dir>
  scripts/spike_view.py test_outputs/test_sanity_vllm_qwen3_30b
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# 8-level Unicode block characters for sparklines (▁ ▂ ▃ ▄ ▅ ▆ ▇ █)
SPARK_CHARS = "▁▂▃▄▅▆▇█"

# Metrics we care about for cascade analysis. Order = display order.
KEY_METRICS = [
    ("vllm:num_requests_running", "running"),
    ("vllm:num_requests_waiting", "waiting"),
    ("vllm:kv_cache_usage_perc", "kv_pct"),
    ("vllm:nixl_post_time_seconds", "nixl_p99"),  # histogram → p99
    ("vllm:nixl_num_failed_transfers", "nxl_fail"),
    ("vllm:nixl_num_kv_expired_reqs", "kv_exp"),
    ("dynamo_frontend_queued_requests", "fe_queued"),
    ("dynamo_frontend_inflight_requests", "fe_inflight"),
    ("dynamo_frontend_time_to_first_token_seconds", "ttft_p99"),  # histogram → p99
    ("dynamo_frontend_request_duration_seconds", "rtt_p99"),  # histogram → p99
]

# Column width for sparklines. 60 chars ≈ 1 minute @ 1Hz scrape.
SPARK_WIDTH = 60


# ────────────────────────────────────────────────────────────────────────────
# Sparkline rendering


def sparkline(values: list[float], width: int = SPARK_WIDTH) -> str:
    """Render a sequence of floats as a Unicode block-character sparkline.

    Resamples to `width` columns by binning. Empty/all-NaN input → blanks.
    """
    if not values:
        return " " * width
    clean = [
        v
        for v in values
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    ]
    if not clean:
        return " " * width
    lo, hi = min(clean), max(clean)
    if lo == hi:
        # all same value: pick mid-band character
        return SPARK_CHARS[3] * min(len(clean), width)
    span = hi - lo
    # Resample to width via simple binning (mean per bin)
    n = len(values)
    if n <= width:
        binned = values
    else:
        binned = []
        bin_size = n / width
        for i in range(width):
            start = int(i * bin_size)
            end = max(int((i + 1) * bin_size), start + 1)
            chunk = [v for v in values[start:end] if v is not None]
            binned.append(statistics.mean(chunk) if chunk else None)
    out = []
    for v in binned:
        if v is None:
            out.append(" ")
        else:
            idx = int((v - lo) / span * (len(SPARK_CHARS) - 1))
            out.append(SPARK_CHARS[max(0, min(idx, len(SPARK_CHARS) - 1))])
    return "".join(out)


# ────────────────────────────────────────────────────────────────────────────
# JSONL parsing (per-record aiperf SlimRecord format)
#
# Schema (from aiperf v0.7.0 src/aiperf/common/models/server_metrics_models.py):
#   {
#     "endpoint_url": "http://<ip>:9090/metrics",
#     "timestamp_ns": <int>,
#     "request_sent_ns": <int|null>,
#     "first_byte_ns": <int|null>,
#     "metrics": {
#       "<metric_name>": [
#         {
#           "labels": {"dp_rank": "0", "dynamo_component": "prefill", ...},
#           "value": <float>,                   # for counter/gauge
#           "buckets": {"0.001": 0, "0.005": 12, ...},  # for histogram
#           "sum": <float>,                     # for histogram
#           "count": <int>,                     # for histogram
#         }, ...
#       ]
#     }
#   }


def parse_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"warn: skipping malformed line in {path}: {e}", file=sys.stderr)
    return records


def histogram_quantile(buckets: dict[str, float], q: float) -> float | None:
    """Estimate quantile q (0..1) from a Prometheus histogram bucket dict.

    Buckets are cumulative counts keyed by `le` upper bound (string representation).
    Linear interpolation within the matching bucket; +Inf → returns the previous bound.
    """
    if not buckets:
        return None
    # Sort by le, with +Inf last
    ordered = []
    for k, v in buckets.items():
        try:
            le = float(k) if k not in ("+Inf", "Inf") else float("inf")
        except ValueError:
            continue
        ordered.append((le, float(v)))
    ordered.sort()
    if not ordered or ordered[-1][1] == 0:
        return None
    total = ordered[-1][1]
    target = q * total
    prev_le, prev_count = 0.0, 0.0
    for le, count in ordered:
        if count >= target:
            if le == float("inf"):
                return prev_le
            if count == prev_count:
                return le
            # Linear interp within the bucket
            frac = (target - prev_count) / (count - prev_count)
            return prev_le + frac * (le - prev_le)
        prev_le, prev_count = le, count
    return ordered[-1][0]


def extract_per_endpoint_series(
    records: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, list[tuple[int, float]]]]:
    """Return: {(endpoint_url, label_signature): {metric_name: [(ts_ns, value), ...]}}.

    label_signature is a stable string of the form 'dp=0,component=prefill,...'.
    For histograms, value = p99 estimate via histogram_quantile.
    """
    series: dict[tuple[str, str], dict[str, list[tuple[int, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for rec in records:
        ts = rec.get("timestamp_ns") or rec.get("first_byte_ns")
        if ts is None:
            continue
        endpoint = rec.get("endpoint_url", "?")
        for metric_name, samples in (rec.get("metrics") or {}).items():
            for sample in samples or []:
                labels = sample.get("labels") or {}
                # Build a compact signature for grouping by label combo
                sig_parts = []
                for k in sorted(labels):
                    if k in ("dynamo_namespace", "model", "model_name", "engine"):
                        continue  # mostly invariant; skip to keep label sig short
                    sig_parts.append(f"{k}={labels[k]}")
                sig = ",".join(sig_parts) or "_"
                key = (endpoint, sig)
                if "value" in sample and sample["value"] is not None:
                    series[key][metric_name].append((ts, float(sample["value"])))
                elif "buckets" in sample and sample["buckets"]:
                    p99 = histogram_quantile(sample["buckets"], 0.99)
                    if p99 is not None:
                        series[key][metric_name].append((ts, p99))
    return series


# ────────────────────────────────────────────────────────────────────────────
# Aggregate-JSON parsing (fallback when no JSONL — produces stats only, no sparklines)


def parse_aggregate_json(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Returns {metric_name: [series_dict, ...]} from server_metrics_export.json."""
    with path.open() as f:
        d = json.load(f)
    return d.get("metrics", {})


# ────────────────────────────────────────────────────────────────────────────
# Reporting


def short_endpoint(url: str) -> str:
    """http://100.67.132.189:9090/metrics  →  100.67.132.189"""
    if "://" in url:
        url = url.split("://", 1)[1]
    return url.split("/", 1)[0].split(":", 1)[0]


def short_label(sig: str) -> str:
    parts = sig.split(",")
    out = {}
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        if k == "dynamo_component":
            out["c"] = v
        elif k == "dynamo_endpoint":
            out["e"] = v
        elif k == "dp_rank":
            out["r"] = v
    return ",".join(f"{k}={v}" for k, v in out.items()) or sig[:40]


def render_jsonl_report(
    series: dict[tuple[str, str], dict[str, list[tuple[int, float]]]],
    out: Path,
) -> None:
    lines = ["# spike_view — per-worker time-series sparklines\n"]
    if not series:
        lines.append("(no records parsed; jsonl was empty or malformed)\n")
        out.write_text("".join(lines))
        return

    # Group by endpoint, then by label signature.
    by_endpoint: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for (ep, sig), metrics_dict in series.items():
        by_endpoint[ep].append((sig, metrics_dict))

    # System-wide totals (sum across endpoints) for the few system-level metrics
    lines.append("## System view\n")
    lines.append("Sums across endpoints; one row per metric.\n\n")
    lines.append("```\n")
    for metric_name, label in KEY_METRICS:
        # Sum at each unique timestamp
        ts_to_sum: dict[int, float] = defaultdict(float)
        seen_any = False
        for (_, _), metrics_dict in series.items():
            for ts, val in metrics_dict.get(metric_name, []):
                ts_to_sum[ts] += val
                seen_any = True
        if not seen_any:
            continue
        ordered = [v for _, v in sorted(ts_to_sum.items())]
        lo, hi = min(ordered), max(ordered)
        spark = sparkline(ordered)
        lines.append(
            f"{label:<14} | {spark} | min={lo:.3g} max={hi:.3g} n={len(ordered)}\n"
        )
    lines.append("```\n\n")

    # Per-endpoint detail
    for ep in sorted(by_endpoint):
        lines.append(f"## {short_endpoint(ep)}\n")
        lines.append(f"`{ep}`\n\n")
        for sig, metrics_dict in sorted(by_endpoint[ep]):
            lines.append(f"### labels: `{short_label(sig)}`\n\n")
            lines.append("```\n")
            for metric_name, label in KEY_METRICS:
                pts = metrics_dict.get(metric_name, [])
                if not pts:
                    continue
                values = [v for _, v in sorted(pts)]
                lo, hi = min(values), max(values)
                spark = sparkline(values)
                lines.append(
                    f"{label:<14} | {spark} | min={lo:.3g} max={hi:.3g} n={len(values)}\n"
                )
            lines.append("```\n\n")
    out.write_text("".join(lines))


def render_aggregate_summary(
    metrics: dict[str, list[dict[str, Any]]],
    out: Path,
) -> None:
    """Fallback when only aggregate JSON exists — render summary stats per endpoint."""
    lines = ["# spike_view — aggregate summary (no JSONL time-series available)\n\n"]
    lines.append(
        "JSONL output (`server_metrics_export.jsonl`) was not produced. "
        "Showing aggregate stats from `server_metrics_export.json` instead — "
        "no time-series, but per-endpoint avg/min/max/p99 still useful for sanity.\n\n"
    )
    # Group by endpoint for tabular display
    rows: list[tuple[str, str, str, dict[str, float]]] = []
    for metric_name, _ in KEY_METRICS:
        if metric_name not in metrics:
            continue
        for s in metrics[metric_name].get("series", []):
            ep = short_endpoint(s.get("endpoint_url", "?"))
            labels = s.get("labels", {}) or {}
            label_str = short_label(
                ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
            )
            stats = s.get("stats", {})
            rows.append((ep, label_str, metric_name, stats))
    if not rows:
        lines.append("(no metrics matched the key list — check metric names)\n")
        out.write_text("".join(lines))
        return
    # Group output by endpoint
    by_ep: dict[str, list] = defaultdict(list)
    for ep, lbl, metric, stats in rows:
        by_ep[ep].append((lbl, metric, stats))
    for ep in sorted(by_ep):
        lines.append(f"## {ep}\n\n")
        lines.append("| labels | metric | avg | max | p99 |\n")
        lines.append("|---|---|---:|---:|---:|\n")
        for lbl, metric, stats in by_ep[ep]:
            avg = stats.get("avg", float("nan"))
            mx = stats.get("max", float("nan"))
            p99 = stats.get("p99", float("nan"))
            lines.append(
                f"| `{lbl}` | `{metric}` | {avg:.3g} | {mx:.3g} | {p99:.3g} |\n"
            )
        lines.append("\n")
    out.write_text("".join(lines))


# ────────────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("run_dir", type=Path, help="Test run directory (parent of load/)")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output markdown path (default: <run_dir>/spike_view.md)",
    )
    ap.add_argument(
        "--echo",
        action="store_true",
        help="Also print the report to stdout",
    )
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    load_dir = run_dir / "load"
    if not load_dir.is_dir():
        print(f"error: {load_dir} not found", file=sys.stderr)
        return 2

    out = args.out or (run_dir / "spike_view.md")
    jsonl = load_dir / "server_metrics_export.jsonl"
    summary = load_dir / "server_metrics_export.json"

    if jsonl.exists():
        records = parse_jsonl(jsonl)
        print(f"parsed {len(records)} JSONL records from {jsonl}", file=sys.stderr)
        series = extract_per_endpoint_series(records)
        render_jsonl_report(series, out)
    elif summary.exists():
        metrics = parse_aggregate_json(summary)
        print(
            f"warn: no JSONL — falling back to aggregate JSON ({summary})",
            file=sys.stderr,
        )
        render_aggregate_summary(metrics, out)
    else:
        print(
            f"error: neither {jsonl} nor {summary} exists. "
            "Did the run pass --server-metrics-formats jsonl?",
            file=sys.stderr,
        )
        return 2

    print(f"wrote {out}", file=sys.stderr)
    if args.echo:
        sys.stdout.write(out.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
