#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
data parsing utilities.

Pure data-extraction functions for Prometheus histograms, nsys SQLite databases,
perf stat output, bpftrace histograms, and timeseries files. Returns structured
Python objects — no formatting or report generation.

Used by create_report.py for report generation.
"""

import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Optional

# Reuse parsers from the existing analysis module (same directory)
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
from frontend_perf_analysis import (  # noqa: E402
    AiperfResult,
    PrometheusSnapshot,
    _extract_aiperf_metrics,
)

# ─── Prometheus parsing ────────────────────────────────────────────────────


def parse_prometheus_text(path: Path) -> Optional[PrometheusSnapshot]:
    """Parse Prometheus text format from a specific file path.

    Extracts stage durations, request plane metrics, transport breakdown,
    Tokio runtime metrics, and transport/compute gauges into a PrometheusSnapshot.
    """
    if not path.exists() or path.stat().st_size == 0:
        return None

    text = path.read_text()
    snap = PrometheusSnapshot()

    def get_gauge(name: str) -> Optional[float]:
        m = re.search(rf"^{re.escape(name)}\s+(\S+)", text, re.MULTILINE)
        return float(m.group(1)) if m else None

    def get_gauge_by_label(name: str, label_key: str) -> dict:
        pattern = rf'^{re.escape(name)}\{{[^}}]*{re.escape(label_key)}="([^"]+)"[^}}]*\}}\s+(\S+)'
        return {
            m.group(1): float(m.group(2))
            for m in re.finditer(pattern, text, re.MULTILINE)
        }

    def histogram_quantile(name: str, quantile: float, filter_label: str = "") -> float:
        bucket_pattern = rf"^{re.escape(name)}_bucket\{{[^}}]*{re.escape(filter_label)}[^}}]*le=\"([^\"]+)\"[^}}]*\}}\s+(\S+)"
        buckets = []
        for m in re.finditer(bucket_pattern, text, re.MULTILINE):
            le_str, count_str = m.group(1), m.group(2)
            le = float("inf") if le_str == "+Inf" else float(le_str)
            buckets.append((le, float(count_str)))
        if not buckets:
            return 0.0
        buckets.sort(key=lambda x: x[0])

        count_m = re.search(
            rf"^{re.escape(name)}_count\{{{re.escape(filter_label)}[^}}]*\}}\s+(\S+)",
            text,
            re.MULTILINE,
        )
        total = float(count_m.group(1)) if count_m else buckets[-1][1]
        if total == 0:
            return 0.0

        target = quantile * total
        prev_le, prev_count = 0.0, 0.0
        for le, count in buckets:
            if count >= target:
                if count == prev_count:
                    return prev_le
                frac = (target - prev_count) / (count - prev_count)
                return prev_le + frac * (le - prev_le)
            prev_le, prev_count = le, count
        return buckets[-1][0] if buckets else 0.0

    # Stage durations
    for stage in ["preprocess", "route", "transport_roundtrip", "postprocess"]:
        label_filter = f'stage="{stage}"'
        p50 = histogram_quantile(
            "dynamo_frontend_stage_duration_seconds", 0.50, label_filter
        )
        p95 = histogram_quantile(
            "dynamo_frontend_stage_duration_seconds", 0.95, label_filter
        )
        p99 = histogram_quantile(
            "dynamo_frontend_stage_duration_seconds", 0.99, label_filter
        )
        count_m = re.search(
            rf"^dynamo_frontend_stage_duration_seconds_count\{{[^}}]*stage=\"{re.escape(stage)}\"[^}}]*\}}\s+(\S+)",
            text,
            re.MULTILINE,
        )
        if count_m and float(count_m.group(1)) > 0:
            snap.stage_durations[stage] = {"p50": p50, "p95": p95, "p99": p99}

    snap.request_plane_queue_p50 = histogram_quantile(
        "dynamo_request_plane_queue_seconds", 0.50
    )
    snap.request_plane_send_p50 = histogram_quantile(
        "dynamo_request_plane_send_seconds", 0.50
    )
    snap.request_plane_roundtrip_ttft_p50 = histogram_quantile(
        "dynamo_request_plane_roundtrip_ttft_seconds", 0.50
    )
    snap.request_plane_inflight = get_gauge("dynamo_request_plane_inflight") or 0

    # Transport breakdown (backend-side metrics)
    for metric_name, attr_name in [
        ("dynamo_component_network_transit_seconds", "work_handler_network_transit"),
        (
            "dynamo_component_time_to_first_response_seconds",
            "work_handler_time_to_first_response",
        ),
    ]:
        p50 = histogram_quantile(metric_name, 0.50)
        p95 = histogram_quantile(metric_name, 0.95)
        p99 = histogram_quantile(metric_name, 0.99)
        if p50 > 0 or p95 > 0 or p99 > 0:
            setattr(snap, attr_name, {"p50": p50, "p95": p95, "p99": p99})

    poll_times = get_gauge_by_label("dynamo_tokio_worker_mean_poll_time_ns", "worker")
    snap.tokio_worker_mean_poll_time_ns = list(poll_times.values())
    snap.tokio_event_loop_stall_total = (
        get_gauge("dynamo_frontend_event_loop_stall_total") or 0
    )
    snap.tokio_global_queue_depth = get_gauge("dynamo_tokio_global_queue_depth") or 0
    snap.tokio_budget_forced_yield_total = (
        get_gauge("dynamo_tokio_budget_forced_yield_total") or 0
    )

    busy_ratios_raw = get_gauge_by_label("dynamo_tokio_worker_busy_ratio", "worker")
    snap.tokio_worker_busy_ratio = [v / 1000.0 for v in busy_ratios_raw.values()]

    snap.tcp_pool_active = get_gauge("dynamo_transport_tcp_pool_active") or 0
    snap.tcp_pool_idle = get_gauge("dynamo_transport_tcp_pool_idle") or 0
    snap.compute_pool_active = (
        get_gauge("dynamo_compute_compute_pool_active_tasks") or 0
    )

    return snap


# ─── aiperf loading ────────────────────────────────────────────────────────


def load_aiperf(obs_dir: Path) -> Optional[AiperfResult]:
    """Load aiperf results from the aiperf subdir."""
    aiperf_dir = obs_dir / "aiperf"
    for candidate in [
        aiperf_dir / "profile_export_aiperf.json",
        aiperf_dir / "profile_results.json",
    ]:
        if candidate.exists():
            try:
                with open(candidate) as f:
                    data = json.load(f)
                return _extract_aiperf_metrics(data)
            except (json.JSONDecodeError, KeyError):
                continue
    # Try any json file in aiperf dir
    if aiperf_dir.is_dir():
        for jf in sorted(aiperf_dir.glob("*.json")):
            try:
                with open(jf) as f:
                    data = json.load(f)
                if "time_to_first_token" in data or "ttft" in data:
                    return _extract_aiperf_metrics(data)
            except (json.JSONDecodeError, KeyError):
                continue
    return None


def load_prometheus(obs_dir: Path) -> Optional[PrometheusSnapshot]:
    """Load Prometheus snapshot — try final_snapshot.txt first, then aiperf dir."""
    prom_dir = obs_dir / "prometheus"
    final_path = prom_dir / "final_snapshot.txt"

    if final_path.exists() and final_path.stat().st_size > 0:
        return parse_prometheus_text(final_path)

    # Fallback: check aiperf dir
    aiperf_prom = obs_dir / "aiperf" / "prometheus_snapshot.txt"
    if aiperf_prom.exists():
        return parse_prometheus_text(aiperf_prom)

    return None


# ─── perf stat parsing ─────────────────────────────────────────────────────


def parse_perf_stat(obs_dir: Path) -> Optional[dict]:
    """Parse perf stat output into a dict of counter name -> value."""
    path = obs_dir / "perf" / "perf_stat.txt"
    if not path.exists():
        return None

    text = path.read_text()
    counters = {}

    patterns = {
        "task-clock": r"([\d,\.]+)\s+msec\s+task-clock",
        "context-switches": r"([\d,\.]+)\s+context-switches",
        "cpu-migrations": r"([\d,\.]+)\s+cpu-migrations",
        "page-faults": r"([\d,\.]+)\s+page-faults",
        "cycles": r"([\d,\.]+)\s+cycles",
        "instructions": r"([\d,\.]+)\s+instructions",
        "branches": r"([\d,\.]+)\s+branches",
        "branch-misses": r"([\d,\.]+)\s+branch-misses",
        "cache-references": r"([\d,\.]+)\s+cache-references",
        "cache-misses": r"([\d,\.]+)\s+cache-misses",
    }

    for name, pattern in patterns.items():
        m = re.search(pattern, text)
        if m:
            counters[name] = float(m.group(1).replace(",", ""))

    # Extract IPC if present
    ipc_m = re.search(r"([\d,\.]+)\s+insn per cycle", text)
    if ipc_m:
        counters["ipc"] = float(ipc_m.group(1).replace(",", ""))

    # Cache miss rate
    cache_refs = counters.get("cache-references", 0)
    cache_misses = counters.get("cache-misses", 0)
    if cache_refs > 0:
        counters["cache-miss-rate"] = cache_misses / cache_refs * 100

    # Branch miss rate
    branches = counters.get("branches", 0)
    branch_misses = counters.get("branch-misses", 0)
    if branches > 0:
        counters["branch-miss-rate"] = branch_misses / branches * 100

    return counters if counters else None


# ─── bpftrace histogram parsing ────────────────────────────────────────────


def parse_bpftrace_histograms(text: str) -> list[dict]:
    """Parse bpftrace histogram output blocks.

    Each block looks like:
    @label_name[key]:
    [1, 2)        123 |@@@@@@@@@@           |
    [2, 4)        456 |@@@@@@@@@@@@@@@@@@@@|
    """
    histograms = []
    current_label = None
    current_buckets = []

    for line in text.split("\n"):
        # Match label line
        label_m = re.match(r"^@(\w+)(?:\[([^\]]*)\])?:", line)
        if label_m:
            if current_label and current_buckets:
                histograms.append({"label": current_label, "buckets": current_buckets})
            current_label = label_m.group(1)
            if label_m.group(2):
                current_label += f"[{label_m.group(2)}]"
            current_buckets = []
            continue

        # Match bucket line: [lo, hi)  count |bars|
        bucket_m = re.match(r"\s*\[(\d+)[,K]?\s*(\d+)[K]?\)\s+(\d+)", line)
        if bucket_m and current_label:
            lo, hi, count = (
                int(bucket_m.group(1)),
                int(bucket_m.group(2)),
                int(bucket_m.group(3)),
            )
            current_buckets.append({"lo": lo, "hi": hi, "count": count})

    if current_label and current_buckets:
        histograms.append({"label": current_label, "buckets": current_buckets})

    return histograms


def summarize_histogram(buckets: list[dict]) -> dict:
    """Compute basic stats (p50, p99, total, max_bucket) from histogram buckets."""
    total = sum(b["count"] for b in buckets)
    if total == 0:
        return {"total": 0, "p50": 0, "p99": 0, "max_bucket": 0}

    cumulative = 0
    p50 = p99 = 0
    max_bucket = 0
    for b in buckets:
        cumulative += b["count"]
        mid = (b["lo"] + b["hi"]) / 2
        if cumulative >= total * 0.50 and p50 == 0:
            p50 = mid
        if cumulative >= total * 0.99 and p99 == 0:
            p99 = mid
        if b["count"] > 0:
            max_bucket = b["hi"]

    return {"total": total, "p50": p50, "p99": p99, "max_bucket": max_bucket}


# ─── timeseries parsing ────────────────────────────────────────────────────


def parse_timeseries(path: Path, key: str) -> list[tuple[str, float]]:
    """Parse lines like '2025-01-01T00:00:00+00:00 key=value'."""
    if not path.exists():
        return []
    points = []
    for line in path.read_text().strip().split("\n"):
        m = re.match(rf"(\S+)\s+{re.escape(key)}=(\d+)", line)
        if m:
            points.append((m.group(1), float(m.group(2))))
    return points


# ─── nsys SQLite queries ───────────────────────────────────────────────────


def parse_nvtx_stages(
    obs_dir: Path,
) -> Optional[list[dict]]:
    """Parse NVTX_EVENTS from nsys SQLite, return list of stage dicts.

    Each dict has keys: name, count, avg_us, min_us, max_us.
    """
    sqlite_path = obs_dir / "nsys" / "frontend.sqlite"
    if not sqlite_path.exists():
        return None

    try:
        conn = sqlite3.connect(str(sqlite_path))
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        if "NVTX_EVENTS" not in tables:
            conn.close()
            return None

        rows = conn.execute(
            """
            SELECT text, COUNT(*) as cnt,
                   AVG(end - start) as avg_ns,
                   MIN(end - start) as min_ns,
                   MAX(end - start) as max_ns
            FROM NVTX_EVENTS
            WHERE text IS NOT NULL AND end > start
            GROUP BY text
            ORDER BY avg_ns DESC
        """
        ).fetchall()
        conn.close()

        if not rows:
            return None

        return [
            {
                "name": text or "?",
                "count": cnt,
                "avg_us": avg_ns / 1000,
                "min_us": min_ns / 1000,
                "max_us": max_ns / 1000,
            }
            for text, cnt, avg_ns, min_ns, max_ns in rows
        ]

    except Exception:
        return None


def parse_syscall_profile(
    obs_dir: Path,
) -> Optional[list[dict]]:
    """Parse OSRT_API from nsys SQLite (OS runtime API calls).

    Each dict has keys: name, count, avg_us, total_ms.
    """
    sqlite_path = obs_dir / "nsys" / "frontend.sqlite"
    if not sqlite_path.exists():
        return None

    try:
        conn = sqlite3.connect(str(sqlite_path))
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]

        if "OSRT_API" not in tables:
            conn.close()
            return None

        rows = conn.execute(
            """
            SELECT nameId, COUNT(*) as cnt,
                   AVG(end - start) as avg_ns,
                   SUM(end - start) as total_ns
            FROM OSRT_API
            WHERE end > start
            GROUP BY nameId
            ORDER BY total_ns DESC
            LIMIT 20
        """
        ).fetchall()

        # Try to resolve names from StringIds table
        name_map = {}
        if "StringIds" in tables:
            for row in conn.execute("SELECT id, value FROM StringIds").fetchall():
                name_map[row[0]] = row[1]
        conn.close()

        if not rows:
            return None

        return [
            {
                "name": name_map.get(name_id, f"id={name_id}"),
                "count": cnt,
                "avg_us": avg_ns / 1000,
                "total_ms": total_ns / 1e6,
            }
            for name_id, cnt, avg_ns, total_ns in rows
        ]

    except Exception:
        return None


def parse_nsys_context_switches(
    obs_dir: Path,
) -> Optional[dict]:
    """Parse SCHED_EVENTS from nsys SQLite.

    Returns dict with keys: total, avg_duration.
    """
    sqlite_path = obs_dir / "nsys" / "frontend.sqlite"
    if not sqlite_path.exists():
        return None

    try:
        conn = sqlite3.connect(str(sqlite_path))
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]

        if "SCHED_EVENTS" not in tables:
            conn.close()
            return None

        row = conn.execute(
            """
            SELECT COUNT(*) as total,
                   AVG(endGlobalTid - startGlobalTid) as avg_duration
            FROM SCHED_EVENTS
        """
        ).fetchone()
        conn.close()

        if not row or row[0] == 0:
            return None

        return {"total": row[0], "avg_duration": row[1]}

    except Exception:
        return None


# ─── Directory utilities ───────────────────────────────────────────────────


def find_latest_obs_dir(repo_root: Path) -> Optional[Path]:
    """Find the most recent artifacts/obs_* directory."""
    artifacts = repo_root / "artifacts"
    if not artifacts.exists():
        return None
    dirs = sorted(artifacts.glob("obs_*"), reverse=True)
    return dirs[0] if dirs else None
