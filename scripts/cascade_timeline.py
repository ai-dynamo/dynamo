#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
cascade_timeline — console "movie" of a cascade run.

Reads `<run>/load/server_metrics_export.jsonl` and renders, per-pod, a
horizontally-time-aligned view of all four cascade signals (and frontend
TTFT/RTT/queued). Each row is a worker or frontend pod; each column is one
sample (~1s). Sparklines run left→right across the run.

This is the static "everything at once" view. For an interactive scrub-
through, pass `--play` to step through every second printed as a Rich Live
table.

Usage:
  scripts/cascade_timeline.py <run-dir>
  scripts/cascade_timeline.py <run-dir> --play         # animate, 1 sample/sec
  scripts/cascade_timeline.py <run-dir> --play --speed 10   # 10x faster

Reads the same JSONL spike_view does. Where spike_view collapses to a system
summary, this one keeps every pod visible.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

# 8-level Unicode block characters
SPARK_CHARS = "▁▂▃▄▅▆▇█"

# Metrics surfaced for workers (port 9090 endpoints)
WORKER_METRICS = [
    ("vllm:num_requests_running", "running"),
    ("vllm:num_requests_waiting", "waiting"),
    ("vllm:kv_cache_usage_perc", "kv_pct"),
    ("vllm:nixl_post_time_seconds", "nixl_p99"),  # histogram → p99
    ("vllm:num_preemptions_total", "preempt"),
    # Error / failure indicators — cumulative counters, plotted as-is so
    # rises are visible. Use `--metrics …,errors,nxl_fail,kv_exp` to pull
    # only error tracks into the view.
    ("vllm:nixl_num_failed_transfers", "nxl_fail"),
    ("vllm:nixl_num_kv_expired_reqs", "kv_exp"),
    ("vllm:request_success_total", "req_ok"),
    # Catch-all "bad request" counter aiperf surfaces for the engine if
    # vllm exposes it; missing on older builds, harmless if absent.
    ("vllm:request_failures_total", "errors"),
]

# Metrics surfaced for frontend (port 8000 endpoint)
FRONTEND_METRICS = [
    ("dynamo_frontend_queued_requests", "queued"),
    ("dynamo_frontend_inflight_requests", "inflight"),
    ("dynamo_frontend_time_to_first_token_seconds", "ttft_p99"),  # histogram → p99
    ("dynamo_frontend_request_duration_seconds", "rtt_p99"),  # histogram → p99
    # Frontend-side error indicators
    ("dynamo_frontend_disconnected_clients", "disc"),
    ("dynamo_frontend_requests_total", "fe_req_total"),
    ("dynamo_frontend_request_errors_total", "fe_errors"),
]

# Default sparkline width. Override via --width.
DEFAULT_WIDTH = 80


def histogram_quantile(buckets, q: float) -> Optional[float]:
    if not buckets:
        return None
    ordered = []
    items = buckets.items() if isinstance(buckets, dict) else buckets
    for k, v in items:
        try:
            le = float(k) if str(k) not in ("+Inf", "Inf") else float("inf")
        except (ValueError, TypeError):
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
            frac = (target - prev_count) / (count - prev_count)
            return prev_le + frac * (le - prev_le)
        prev_le, prev_count = le, count
    return ordered[-1][0]


def parse_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def short_endpoint(url: str) -> str:
    if "://" in url:
        url = url.split("://", 1)[1]
    return url.split("/", 1)[0]


def build_per_pod_series(
    records: list[dict],
) -> dict[str, dict[str, dict[str, list[tuple[int, float]]]]]:
    """Returns: {pod_id: {role: {metric_name: [(ts_ns, value), ...]}}}.

    pod_id = endpoint_url IP:port (one pod per IP).
    role = "frontend" / "prefill" / "decode" / "unknown".
    Aggregation:
      - counter/gauge with `value` field: sum across samples (e.g. across
        dp_rank/engine labels for a single pod);
      - histogram (`buckets` field): per-pod prior-buckets are kept, and
        each scrape is converted to a *delta histogram* (current minus
        prior) before computing p99 — so the rendered p99 reflects the
        last scrape window's transfers, not all-time. First sample of
        each pod is skipped (no baseline yet) to avoid boostrap noise.
    """
    series: dict[str, dict[str, dict[str, list[tuple[int, float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    # Track prior cumulative buckets per (pod, metric) for delta-p99 path.
    # Key: (pod_id, metric_name) → {le_float: cumulative_count}
    prior_buckets: dict[tuple[str, str], dict[float, float]] = {}

    for rec in records:
        ts = rec.get("timestamp_ns") or rec.get("first_byte_ns")
        if ts is None:
            continue
        endpoint = rec.get("endpoint_url", "?")
        pod_id = short_endpoint(endpoint)
        role = "unknown"
        if pod_id.endswith(":8000"):
            role = "frontend"
        for metric_name, samples in (rec.get("metrics") or {}).items():
            samples = samples or []
            if not samples:
                continue
            if role == "unknown":
                for s in samples:
                    labels = s.get("labels") or {}
                    comp = labels.get("dynamo_component")
                    if comp:
                        role = (
                            "decode"
                            if comp == "backend"
                            else ("prefill" if comp == "prefill" else "frontend")
                        )
                        break
            simple_total = 0.0
            simple_seen = False
            agg_buckets: dict[float, float] = {}
            for s in samples:
                if "value" in s and s["value"] is not None:
                    simple_total += float(s["value"])
                    simple_seen = True
                elif "buckets" in s and s["buckets"]:
                    bs = s["buckets"]
                    bs_items = bs.items() if isinstance(bs, dict) else bs
                    for le, cnt in bs_items:
                        try:
                            le_f = (
                                float(le)
                                if str(le) not in ("+Inf", "Inf")
                                else float("inf")
                            )
                        except (ValueError, TypeError):
                            continue
                        agg_buckets[le_f] = agg_buckets.get(le_f, 0.0) + float(cnt)
            if simple_seen:
                series[pod_id][role][metric_name].append((ts, simple_total))
            elif agg_buckets:
                # Compute delta vs prior cumulative — per-window p99
                key = (pod_id, metric_name)
                prior = prior_buckets.get(key, {})
                delta: dict[float, float] = {}
                for le_f, cum in agg_buckets.items():
                    delta[le_f] = max(0.0, cum - prior.get(le_f, 0.0))
                prior_buckets[key] = agg_buckets  # update baseline
                # Skip first scrape (delta == cumulative meaningless)
                if not prior:
                    continue
                p99 = histogram_quantile(delta, 0.99)
                if p99 is not None:
                    series[pod_id][role][metric_name].append((ts, p99))
    return series


def make_spark(values, width: int) -> str:
    if not values:
        return " " * width
    clean = [v for v in values if v is not None]
    if not clean:
        return " " * width
    lo, hi = min(clean), max(clean)
    if lo == hi:
        return SPARK_CHARS[3] * min(len(clean), width)
    # Resample into width bins
    n = len(clean)
    if n <= width:
        binned = clean
    else:
        bin_size = n / width
        binned = []
        for i in range(width):
            s = int(i * bin_size)
            e = max(int((i + 1) * bin_size), s + 1)
            chunk = clean[s:e]
            binned.append(sum(chunk) / len(chunk) if chunk else None)
    span = hi - lo
    out = []
    for v in binned:
        if v is None:
            out.append(" ")
        else:
            idx = int((v - lo) / span * (len(SPARK_CHARS) - 1))
            out.append(SPARK_CHARS[max(0, min(idx, len(SPARK_CHARS) - 1))])
    return "".join(out)


def fmt_value(v: float, kind: str) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "?"
    if kind == "kv_pct":
        return f"{v:.1%}"
    if kind in ("nixl_p99", "ttft_p99", "rtt_p99"):
        if v >= 1.0:
            return f"{v:.2f}s"
        return f"{v * 1000:.0f}ms"
    if v >= 100:
        return f"{int(v)}"
    if v >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def role_sort_key(pod_id: str, role: str) -> tuple:
    role_order = {"frontend": 0, "prefill": 1, "decode": 2, "unknown": 9}
    return (role_order.get(role, 9), pod_id)


def render_static(series: dict, width: int, time_axis_seconds: bool = True) -> str:
    lines: list[str] = []
    lines.append("# cascade timeline — per-pod\n")
    if not series:
        lines.append("(no records)\n")
        return "".join(lines)

    # Compute global time range to put under the spark
    all_ts = []
    for pod, roles in series.items():
        for role, metrics_map in roles.items():
            for metric, points in metrics_map.items():
                for ts, _ in points:
                    all_ts.append(ts)
    if not all_ts:
        return "(no timestamps)\n"
    t0_ns, t1_ns = min(all_ts), max(all_ts)
    span_s = (t1_ns - t0_ns) / 1e9
    lines.append(f"\nrun span: {span_s:.0f}s  ({len(set(all_ts))} unique samples)\n\n")

    # Group rows by role first, then pod
    rows: list[tuple[str, str, str, list[float]]] = []
    for pod_id, roles_map in series.items():
        for role, metrics_map in roles_map.items():
            metrics_order = FRONTEND_METRICS if role == "frontend" else WORKER_METRICS
            for metric_name, label in metrics_order:
                pts = metrics_map.get(metric_name, [])
                if not pts:
                    continue
                values = [v for _, v in sorted(pts)]
                rows.append((role, pod_id, label, values))

    metric_display_order = [
        "queued",
        "inflight",
        "ttft_p99",
        "rtt_p99",
        "disc",
        "fe_req_total",
        "fe_errors",
        "running",
        "waiting",
        "kv_pct",
        "nixl_p99",
        "preempt",
        "nxl_fail",
        "kv_exp",
        "req_ok",
        "errors",
    ]
    rows.sort(
        key=lambda r: (
            {"frontend": 0, "prefill": 1, "decode": 2, "unknown": 9}.get(r[0], 9),
            r[1],
            metric_display_order.index(r[2]) if r[2] in metric_display_order else 99,
        )
    )

    # Render
    last_pod = None
    for role, pod_id, label, values in rows:
        pod_role = f"{pod_id}  [{role}]"
        if pod_role != last_pod:
            if last_pod is not None:
                lines.append("\n")
            lines.append(f"### {pod_role}\n")
            lines.append("```\n")
            last_pod = pod_role
        spark = make_spark(values, width)
        lo = min(values)
        hi = max(values)
        cur = values[-1] if values else None
        lo_f = fmt_value(lo, label)
        hi_f = fmt_value(hi, label)
        cur_f = fmt_value(cur, label) if cur is not None else "—"
        lines.append(
            f"{label:<10} | {spark} | "
            f"now={cur_f:>7s}  min={lo_f:>7s}  max={hi_f:>7s}  n={len(values)}\n"
        )
    if last_pod is not None:
        lines.append("```\n")

    if time_axis_seconds:
        # Add a time-axis bar under one example sparkline (just informational)
        marks = "".join(
            f"{i*int(span_s/8):>8d}" if i % 1 == 0 else " " for i in range(9)
        )
        lines.append(f"\nTime axis (s, left→right): {marks.strip()}\n")
    return "".join(lines)


def render_play(series: dict, speed: float = 1.0) -> None:
    """Animate: print one row of values per second, advancing through the run."""
    try:
        from rich.console import Console
        from rich.live import Live
        from rich.table import Table
    except ImportError:
        print("error: rich not installed; pip install rich", file=sys.stderr)
        sys.exit(2)

    # Build a unified time→pod→metric map
    all_ts = sorted(
        {
            ts
            for pod, roles in series.items()
            for role, metrics in roles.items()
            for points in metrics.values()
            for ts, _ in points
        }
    )
    if not all_ts:
        print("(no data to animate)", file=sys.stderr)
        return

    t0 = all_ts[0]

    # Bucket samples into 1s windows
    buckets: dict[int, dict] = defaultdict(dict)  # second-offset → {(pod,metric): val}
    for pod_id, roles_map in series.items():
        for role, metrics_map in roles_map.items():
            for metric_name, pts in metrics_map.items():
                for ts, val in pts:
                    sec = int((ts - t0) / 1e9)
                    buckets[sec][(pod_id, role, metric_name)] = val

    second_keys = sorted(buckets)
    console = Console()

    def make_table(sec: int) -> Table:
        tbl = Table(title=f"cascade @ t={sec}s   ({sec}/{second_keys[-1]}s)")
        tbl.add_column("pod", style="cyan")
        tbl.add_column("role")
        tbl.add_column("running", justify="right")
        tbl.add_column("waiting", justify="right")
        tbl.add_column("kv%", justify="right")
        tbl.add_column("nixl_p99", justify="right")
        tbl.add_column("ttft_p99", justify="right")
        tbl.add_column("rtt_p99", justify="right")
        # Aggregate at this second (use latest sample per pod-metric ≤ sec)
        snapshot: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
        for s in range(sec + 1):
            for (pod_id, role, metric), val in buckets.get(s, {}).items():
                snapshot[(pod_id, role)][metric] = val
        for pod_id, role in sorted(
            snapshot,
            key=lambda r: (
                {"frontend": 0, "prefill": 1, "decode": 2, "unknown": 9}.get(r[1], 9),
                r[0],
            ),
        ):
            m = snapshot[(pod_id, role)]
            row = [
                pod_id,
                role,
                fmt_value(m.get("vllm:num_requests_running"), "running"),
                fmt_value(m.get("vllm:num_requests_waiting"), "waiting"),
                fmt_value(m.get("vllm:kv_cache_usage_perc"), "kv_pct"),
                fmt_value(m.get("vllm:nixl_post_time_seconds"), "nixl_p99"),
                fmt_value(
                    m.get("dynamo_frontend_time_to_first_token_seconds"), "ttft_p99"
                ),
                fmt_value(m.get("dynamo_frontend_request_duration_seconds"), "rtt_p99"),
            ]
            tbl.add_row(*row)
        return tbl

    with Live(console=console, refresh_per_second=max(speed, 1.0)) as live:
        for sec in second_keys:
            live.update(make_table(sec))
            time.sleep(max(0.0, 1.0 / speed))


_ALL_LABELS = [
    "running",
    "waiting",
    "kv_pct",
    "nixl_p99",
    "preempt",
    "nxl_fail",
    "kv_exp",
    "req_ok",
    "errors",
    "queued",
    "inflight",
    "ttft_p99",
    "rtt_p99",
    "disc",
    "fe_req_total",
    "fe_errors",
]


def _filter_metrics_by_labels(selected: set[str]) -> tuple[list, list]:
    """Filter the global METRIC tables to only the requested label set."""
    w = [(name, label) for name, label in WORKER_METRICS if label in selected]
    f = [(name, label) for name, label in FRONTEND_METRICS if label in selected]
    return w, f


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Available metrics (--metrics):\n"
            "  worker:        running, waiting, kv_pct, nixl_p99, preempt\n"
            "  worker errors: nxl_fail, kv_exp, req_ok, errors\n"
            "  frontend:      queued, inflight, ttft_p99, rtt_p99\n"
            "  frontend err:  disc, fe_req_total, fe_errors\n"
            "  Pass a comma-separated subset to focus, or omit for all.\n"
        ),
    )
    ap.add_argument("run_dir", type=Path, help="Test run directory (parent of load/)")
    ap.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    ap.add_argument("--play", action="store_true", help="Animate (rich Live table)")
    ap.add_argument(
        "--speed", type=float, default=10.0, help="Playback speed (1.0 = realtime)"
    )
    ap.add_argument(
        "--metrics",
        default=None,
        help="Comma-separated metric labels to show (e.g. 'running,waiting,nixl_p99'). "
        "Default: all available.",
    )
    ap.add_argument(
        "--list-metrics",
        action="store_true",
        help="Print the metric labels available for this run and exit.",
    )
    args = ap.parse_args()

    # Apply --metrics filter to globals so all rendering paths agree.
    if args.metrics:
        selected = {m.strip() for m in args.metrics.split(",") if m.strip()}
        unknown = selected - set(_ALL_LABELS)
        if unknown:
            print(f"error: unknown metric label(s): {sorted(unknown)}", file=sys.stderr)
            print(f"valid: {_ALL_LABELS}", file=sys.stderr)
            return 2
        global WORKER_METRICS, FRONTEND_METRICS
        WORKER_METRICS, FRONTEND_METRICS = _filter_metrics_by_labels(selected)

    jsonl = args.run_dir / "load" / "server_metrics_export.jsonl"
    if not jsonl.exists():
        print(f"error: {jsonl} not found", file=sys.stderr)
        return 2

    if args.list_metrics:
        records = parse_jsonl(jsonl)
        seen: set[str] = set()
        for rec in records:
            for m in rec.get("metrics") or {}:
                seen.add(m)
        print("Prometheus families present in this run:")
        for m in sorted(seen):
            print(f"  {m}")
        print("\nLabel shortcuts (use with --metrics):")
        for lbl in _ALL_LABELS:
            print(f"  {lbl}")
        return 0

    records = parse_jsonl(jsonl)
    print(f"parsed {len(records)} records", file=sys.stderr)
    series = build_per_pod_series(records)

    if args.play:
        render_play(series, speed=args.speed)
    else:
        sys.stdout.write(render_static(series, args.width))
    return 0


if __name__ == "__main__":
    sys.exit(main())
