#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unified observability analysis script.

Ingests the entire output directory from run_perf.sh and
produces a comprehensive report covering:
  - aiperf throughput & latency
  - Server-side Prometheus metrics (stage durations, tokio, transport, compute)
  - NVTX pipeline stages (from nsys SQLite export)
  - Syscall profile (from nsys OSRT_API)
  - Hardware counters (from perf stat)
  - CPU flamegraph pointer
  - BPF insights (runqlat, syscall latency, transport latency, context switches)
  - System resource trends (thread count, FD count)
  - Auto-generated key findings

Usage:
    python create_report.py analyze <obs_directory>
    python create_report.py analyze  # auto-finds latest obs_* dir
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Reuse parsers from the existing analysis module (same directory)
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
from frontend_perf_analysis import AiperfResult, PrometheusSnapshot  # noqa: E402
from parsing_util import (  # noqa: E402
    find_latest_obs_dir,
    load_aiperf,
    load_prometheus,
    parse_bpftrace_histograms,
    parse_nsys_context_switches,
    parse_nvtx_stages,
    parse_perf_stat,
    parse_syscall_profile,
    parse_timeseries,
    summarize_histogram,
)

# ─── Section helpers ─────────────────────────────────────────────────────────


def _section(title: str) -> str:
    """Format a report section header."""
    return f"\n## {title}\n"


def _subsection(title: str) -> str:
    return f"\n### {title}\n"


# ─── 1. Configuration ─────────────────────────────────────────────────────


def section_config(obs_dir: Path) -> Optional[str]:
    config_path = obs_dir / "config.json"
    if not config_path.exists():
        return None

    try:
        with open(config_path) as f:
            config = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    lines = [_section("1. Capture Configuration")]

    key_fields = [
        ("Model", "model"),
        ("Concurrency", "concurrency"),
        ("Num Requests", "num_requests"),
        ("ISL (tokens)", "isl"),
        ("OSL (tokens)", "osl"),
        ("Speedup Ratio", "speedup_ratio"),
        ("Workers", "num_workers"),
        ("Event Plane", "event_plane"),
    ]

    lines.append("| Parameter | Value |")
    lines.append("|---|---|")

    for label, key in key_fields:
        val = config.get(key, "N/A")
        lines.append(f"| {label} | {val} |")

    # Show which profilers were active (keys match run_perf.sh config.json)
    profilers = []
    if config.get("has_nsys"):
        profilers.append("nsys")
    if config.get("has_perf"):
        profilers.append("perf")
    if config.get("has_bpf"):
        profilers.append("bpf")
    lines.append(f"| Profilers | {', '.join(profilers) if profilers else 'none'} |")

    return "\n".join(lines)


# ─── 2. Throughput ─────────────────────────────────────────────────────────


def section_throughput(aiperf: Optional[AiperfResult]) -> Optional[str]:
    if not aiperf:
        return None
    lines = [_section("2. Throughput")]
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Requests/sec | {aiperf.request_throughput_rps:.2f} |")
    lines.append(f"| Output tokens/sec | {aiperf.throughput_tok_s:.2f} |")
    return "\n".join(lines)


# ─── 3. Latency ───────────────────────────────────────────────────────────


def section_latency(aiperf: Optional[AiperfResult]) -> Optional[str]:
    if not aiperf:
        return None
    lines = [_section("3. End-to-End Latency")]
    lines.append("| Metric | p50 (ms) | p95 (ms) | p99 (ms) |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| TTFT | {aiperf.ttft_p50_ms:.1f} | {aiperf.ttft_p95_ms:.1f} | {aiperf.ttft_p99_ms:.1f} |"
    )
    lines.append(
        f"| ITL | {aiperf.itl_p50_ms:.1f} | {aiperf.itl_p95_ms:.1f} | {aiperf.itl_p99_ms:.1f} |"
    )
    return "\n".join(lines)


# ─── 4. Pipeline Stage Durations ──────────────────────────────────────────


def section_lifecycle(prom: Optional[PrometheusSnapshot]) -> Optional[str]:
    if not prom or not prom.stage_durations:
        return None

    lines = [_section("4. Pipeline Stage Durations (Prometheus histograms)")]
    lines.append("| Stage | p50 (ms) | p95 (ms) | p99 (ms) |")
    lines.append("|---|---:|---:|---:|")

    total_p50 = 0
    for stage, vals in prom.stage_durations.items():
        p50_ms = vals["p50"] * 1000
        p95_ms = vals["p95"] * 1000
        p99_ms = vals["p99"] * 1000
        lines.append(f"| {stage} | {p50_ms:.2f} | {p95_ms:.2f} | {p99_ms:.2f} |")
        total_p50 += p50_ms

    lines.append(f"| **TOTAL (sum of p50)** | **{total_p50:.2f}** | | |")

    return "\n".join(lines)


# ─── 4b. Transport Breakdown ─────────────────────────────────────────────


def section_transport_breakdown(prom: Optional[PrometheusSnapshot]) -> Optional[str]:
    if not prom:
        return None

    has_request_plane = (
        prom.request_plane_queue_p50 > 0
        or prom.request_plane_send_p50 > 0
        or prom.request_plane_roundtrip_ttft_p50 > 0
    )

    has_work_handler = hasattr(prom, "work_handler_network_transit") and (
        prom.work_handler_network_transit or prom.work_handler_time_to_first_response
    )

    if not has_request_plane and not has_work_handler:
        return None

    lines = [_section("4b. Transport Breakdown")]

    if has_request_plane:
        lines.append(_subsection("Frontend View (AddressedPushRouter)"))
        queue_ms = prom.request_plane_queue_p50 * 1000
        send_ms = prom.request_plane_send_p50 * 1000
        roundtrip_ttft_ms = prom.request_plane_roundtrip_ttft_p50 * 1000
        lines.append("| Metric | p50 (ms) |")
        lines.append("|---|---:|")
        lines.append(f"| Queue (serialize+encode) | {queue_ms:.2f} |")
        lines.append(f"| Send (network+ack) | {send_ms:.2f} |")
        lines.append(f"| Roundtrip TTFT | {roundtrip_ttft_ms:.2f} |")
        lines.append(f"| Inflight gauge | {prom.request_plane_inflight:.0f} |")

    if has_work_handler:
        lines.append(_subsection("Backend View (WorkHandler)"))

        parts_rows = []
        if prom.work_handler_network_transit:
            t = prom.work_handler_network_transit
            parts_rows.append(
                f"| Part 1 - Network transit (T2-T1) | {t['p50']*1000:.2f} | {t['p95']*1000:.2f} | {t['p99']*1000:.2f} |"
            )

        if prom.work_handler_time_to_first_response:
            t = prom.work_handler_time_to_first_response
            parts_rows.append(
                f"| Part 2 - Processing (T3-T2) | {t['p50']*1000:.2f} | {t['p95']*1000:.2f} | {t['p99']*1000:.2f} |"
            )

        if parts_rows:
            lines.append("| Phase | p50 (ms) | p95 (ms) | p99 (ms) |")
            lines.append("|---|---:|---:|---:|")
            lines.extend(parts_rows)

        # Note: Part 3 (response return) cannot be reliably derived by
        # subtracting p50s because quantiles are not additive across
        # independent distributions.  A distribution-based breakdown
        # would require access to raw sample histograms.

    return "\n".join(lines)


# ─── 5. NVTX Pipeline Stages ────────────────────────────────────────────────


def section_nvtx(obs_dir: Path) -> Optional[str]:
    """Format NVTX_EVENTS from nsys SQLite export."""
    stages = parse_nvtx_stages(obs_dir)
    if not stages:
        return None

    lines = [_section("5. NVTX Pipeline Stages (from nsys SQLite)")]
    lines.append("| Range Name | Count | Avg (us) | Min (us) | Max (us) |")
    lines.append("|---|---:|---:|---:|---:|")

    for s in stages:
        name = s["name"][:40]
        lines.append(
            f"| {name} | {s['count']:d} | {s['avg_us']:.1f} | {s['min_us']:.1f} | {s['max_us']:.1f} |"
        )

    return "\n".join(lines)


# ─── 6. Syscall Profile ─────────────────────────────────────────────────────


def section_syscall_profile(obs_dir: Path) -> Optional[str]:
    """Format OSRT_API from nsys SQLite (OS runtime API calls)."""
    profile = parse_syscall_profile(obs_dir)
    if not profile:
        return None

    lines = [_section("6. Syscall / OS Runtime Profile (from nsys)")]
    lines.append("| API Call | Count | Avg (us) | Total (ms) |")
    lines.append("|---|---:|---:|---:|")

    for entry in profile:
        name = entry["name"][:40]
        lines.append(
            f"| {name} | {entry['count']:d} | {entry['avg_us']:.1f} | {entry['total_ms']:.1f} |"
        )

    return "\n".join(lines)


# ─── 7. Tokio Runtime Health ────────────────────────────────────────────────


def _worker_distribution(values: list, unit: str, warn_threshold: float) -> list:
    """Summarize a per-worker metric list into a compact distribution table."""
    if not values:
        return []
    n = len(values)
    sorted_vals = sorted(values)
    avg = sum(values) / n
    p50 = sorted_vals[n // 2]
    p99 = sorted_vals[min(int(n * 0.99), n - 1)]
    mn, mx = sorted_vals[0], sorted_vals[-1]
    n_warn = sum(1 for v in values if v > warn_threshold)
    n_ok = n - n_warn

    lines = []
    lines.append(f"| Workers | {n} |")
    lines.append(f"| Avg | {avg:,.0f} {unit} |")
    lines.append(f"| p50 | {p50:,.0f} {unit} |")
    lines.append(f"| p99 | {p99:,.0f} {unit} |")
    lines.append(f"| Min / Max | {mn:,.0f} / {mx:,.0f} {unit} |")
    lines.append(
        f"| Health | {n_ok} ok, {n_warn} warn (threshold: {warn_threshold:,.0f} {unit}) |"
    )
    return lines


def section_tokio(prom: Optional[PrometheusSnapshot]) -> Optional[str]:
    if not prom:
        return None
    if not prom.tokio_worker_mean_poll_time_ns and not prom.tokio_worker_busy_ratio:
        return None

    lines = [_section("7. Tokio Runtime Health")]

    # --- Poll Time Summary ---
    if prom.tokio_worker_mean_poll_time_ns:
        lines.append(_subsection("Worker Poll Time"))
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.extend(
            _worker_distribution(prom.tokio_worker_mean_poll_time_ns, "ns", 100_000)
        )

    # --- Busy Ratio Summary ---
    if prom.tokio_worker_busy_ratio:
        n = len(prom.tokio_worker_busy_ratio)
        sorted_br = sorted(prom.tokio_worker_busy_ratio)
        avg_br = sum(prom.tokio_worker_busy_ratio) / n
        p50_br = sorted_br[n // 2]
        p99_br = sorted_br[min(int(n * 0.99), n - 1)]
        n_saturated = sum(1 for v in prom.tokio_worker_busy_ratio if v >= 0.95)
        n_hot = sum(1 for v in prom.tokio_worker_busy_ratio if 0.5 <= v < 0.95)
        n_idle = sum(1 for v in prom.tokio_worker_busy_ratio if v < 0.5)

        lines.append(_subsection("Worker Busy Ratio"))
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| Workers | {n} |")
        lines.append(f"| Avg | {avg_br:.3f} |")
        lines.append(f"| p50 / p99 | {p50_br:.3f} / {p99_br:.3f} |")
        lines.append(f"| Min / Max | {sorted_br[0]:.3f} / {sorted_br[-1]:.3f} |")
        lines.append(
            f"| Distribution | {n_idle} idle (<0.5), {n_hot} hot (0.5-0.95), {n_saturated} saturated (>=0.95) |"
        )

    # --- Event Loop ---
    lines.append(_subsection("Event Loop"))
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Stall count | {prom.tokio_event_loop_stall_total:.0f} |")
    lines.append(f"| Global queue depth | {prom.tokio_global_queue_depth:.0f} |")
    lines.append(
        f"| Budget forced yields | {prom.tokio_budget_forced_yield_total:.0f} |"
    )

    # --- Assessment ---
    lines.append(_subsection("Assessment"))
    issues = []
    if prom.tokio_worker_mean_poll_time_ns:
        avg_pt = sum(prom.tokio_worker_mean_poll_time_ns) / len(
            prom.tokio_worker_mean_poll_time_ns
        )
        if avg_pt > 100_000:
            issues.append(f"High avg poll time: {avg_pt:,.0f}ns (threshold: 100,000ns)")
    if prom.tokio_worker_busy_ratio:
        n_saturated = sum(1 for v in prom.tokio_worker_busy_ratio if v >= 0.95)
        avg_br = sum(prom.tokio_worker_busy_ratio) / len(prom.tokio_worker_busy_ratio)
        if avg_br > 0.8:
            issues.append(f"High avg busy ratio: {avg_br:.3f} (threshold: 0.8)")
        if n_saturated > 0:
            issues.append(f"{n_saturated} worker(s) saturated (busy ratio >= 0.95)")
    if prom.tokio_event_loop_stall_total > 0:
        issues.append(
            f"Event loop stalls detected: {prom.tokio_event_loop_stall_total:.0f}"
        )

    if issues:
        for issue in issues:
            lines.append(f"- **warn** {issue}")
    else:
        lines.append("- **ok** Tokio runtime healthy")

    return "\n".join(lines)


# ─── 8. Transport Gauges ─────────────────────────────────────────────────


def section_transport(prom: Optional[PrometheusSnapshot]) -> Optional[str]:
    if not prom:
        return None
    if prom.tcp_pool_active == 0 and prom.tcp_pool_idle == 0:
        return None

    lines = [_section("8. Transport Layer")]
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| TCP Pool Active | {prom.tcp_pool_active:.0f} |")
    lines.append(f"| TCP Pool Idle | {prom.tcp_pool_idle:.0f} |")

    total = prom.tcp_pool_active + prom.tcp_pool_idle
    if total > 0:
        utilization = prom.tcp_pool_active / total * 100
        lines.append(f"| Utilization | {utilization:.1f}% |")

    return "\n".join(lines)


# ─── 9. Compute Pool ─────────────────────────────────────────────────────


def section_compute(prom: Optional[PrometheusSnapshot]) -> Optional[str]:
    if not prom or prom.compute_pool_active == 0:
        return None
    lines = [_section("9. Compute Pool")]
    lines.append(f"Active tasks: {prom.compute_pool_active:.0f}")
    return "\n".join(lines)


# ─── 10. Hardware Counters ──────────────────────────────────────────────────


def section_hw_counters(obs_dir: Path) -> Optional[str]:
    counters = parse_perf_stat(obs_dir)
    if not counters:
        return None

    lines = [_section("10. Hardware Counters (perf stat)")]
    lines.append("| Counter | Value |")
    lines.append("|---|---:|")

    for key in [
        "task-clock",
        "context-switches",
        "cpu-migrations",
        "page-faults",
        "cycles",
        "instructions",
        "branches",
        "branch-misses",
        "cache-references",
        "cache-misses",
    ]:
        if key in counters:
            lines.append(f"| {key} | {counters[key]:,.0f} |")

    if "ipc" in counters:
        lines.append(f"| IPC | {counters['ipc']:.2f} |")
    if "cache-miss-rate" in counters:
        lines.append(f"| Cache miss rate | {counters['cache-miss-rate']:.2f}% |")
    if "branch-miss-rate" in counters:
        lines.append(f"| Branch miss rate | {counters['branch-miss-rate']:.2f}% |")

    return "\n".join(lines)


# ─── 11. Flamegraph ─────────────────────────────────────────────────────


def section_flamegraph(obs_dir: Path) -> Optional[str]:
    lines = [_section("11. Flamegraphs")]
    found = False

    svg_entries = [
        ("cpu_flamegraph.svg", "CPU Flamegraph"),
        ("offcpu_flamegraph.svg", "Off-CPU Flamegraph"),
    ]

    for filename, label in svg_entries:
        path = obs_dir / "perf" / filename
        if path.exists() and path.stat().st_size > 0:
            # Use relative path from report.md location (obs_dir/)
            rel_path = f"perf/{filename}"
            lines.append(f"### {label}")
            lines.append("")
            lines.append(f'<img src="{rel_path}" alt="{label}" width="100%">')
            lines.append("")
            lines.append(f"*File: `{path}`*")
            lines.append("")
            found = True
    return "\n".join(lines) if found else None


# ─── 12. BPF Insights ───────────────────────────────────────────────────────


def section_bpf(obs_dir: Path) -> Optional[str]:
    bpf_dir = obs_dir / "bpf"
    if not bpf_dir.exists():
        return None

    parts = []

    def _bpf_summary(bpf_path: Path, title: str, desc: str) -> list:
        """Parse a BPF histogram file and return a compact summary."""
        if not bpf_path.exists() or bpf_path.stat().st_size == 0:
            return []
        text = bpf_path.read_text()
        hists = parse_bpftrace_histograms(text)
        if not hists:
            return []

        # Aggregate across all histogram labels for a single summary
        total_samples = 0
        weighted_p50_sum = 0.0
        weighted_p99_sum = 0.0
        max_p99 = 0.0
        max_bucket_seen = 0

        for h in hists[:10]:
            stats = summarize_histogram(h["buckets"])
            n = stats["total"]
            if n == 0:
                continue
            total_samples += n
            weighted_p50_sum += stats["p50"] * n
            weighted_p99_sum += stats["p99"] * n
            max_p99 = max(max_p99, stats["p99"])
            max_bucket_seen = max(max_bucket_seen, stats["max_bucket"])

        if total_samples == 0:
            return []

        avg_p50 = weighted_p50_sum / total_samples
        avg_p99 = weighted_p99_sum / total_samples

        rows = [_subsection(title)]
        rows.append(f"> {desc}")
        rows.append("")
        rows.append("| Metric | Value |")
        rows.append("|---|---|")
        rows.append(f"| Total samples | {total_samples:,d} |")
        rows.append(f"| p50 (weighted avg) | {avg_p50:,.0f} us |")
        rows.append(f"| p99 (weighted avg) | {avg_p99:,.0f} us |")
        rows.append(f"| Worst p99 | {max_p99:,.0f} us |")
        rows.append(f"| Max bucket | {max_bucket_seen:,d} us |")

        return rows

    parts.extend(
        _bpf_summary(
            bpf_dir / "runqlat.txt",
            "Run Queue Latency",
            "Time tasks spent waiting in the CPU run queue before being scheduled. "
            "High values indicate CPU contention or oversubscription.",
        )
    )
    parts.extend(
        _bpf_summary(
            bpf_dir / "syscall_latency.txt",
            "Syscall Latency",
            "Latency of key syscalls (futex, epoll_wait, read, write, recvmsg). "
            "High futex latency suggests lock contention; high epoll_wait means idle polling.",
        )
    )
    parts.extend(
        _bpf_summary(
            bpf_dir / "transport_latency.txt",
            "Transport Latency (BPF)",
            "Kernel-observed latency for TCP send/recv operations on the transport plane. "
            "High values may indicate network congestion or small-buffer stalls.",
        )
    )
    parts.extend(
        _bpf_summary(
            bpf_dir / "context_switches.txt",
            "Context Switch Overhead",
            "Voluntary and involuntary context switch overhead per event. "
            "High involuntary switches suggest CPU pressure.",
        )
    )

    if not parts:
        return None

    return _section("12. BPF Insights") + "\n".join(parts)


# ─── 13. System Resources ───────────────────────────────────────────────────


def section_system_resources(obs_dir: Path) -> Optional[str]:
    system_dir = obs_dir / "system"
    if not system_dir.exists():
        return None

    rows = []

    # Thread count
    thread_data = parse_timeseries(system_dir / "thread_count.txt", "threads")
    if thread_data:
        values = [v for _, v in thread_data]
        rows.append(
            f"| Threads | {min(values):.0f} | {max(values):.0f} | {sum(values)/len(values):.0f} | {len(values)} |"
        )

    # FD count
    fd_data = parse_timeseries(system_dir / "fd_count.txt", "fds")
    if fd_data:
        values = [v for _, v in fd_data]
        rows.append(
            f"| FDs | {min(values):.0f} | {max(values):.0f} | {sum(values)/len(values):.0f} | {len(values)} |"
        )

    if not rows:
        return None

    lines = [_section("13. System Resources")]
    lines.append("| Resource | Min | Max | Avg | Samples |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.extend(rows)
    return "\n".join(lines)


# ─── 14. Context Switches (nsys) ────────────────────────────────────────────


def section_nsys_context_switches(obs_dir: Path) -> Optional[str]:
    data = parse_nsys_context_switches(obs_dir)
    if not data:
        return None

    lines = [_section("14. Context Switches (nsys SCHED_EVENTS)")]
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Total Events | {data['total']:,d} |")
    if data["avg_duration"]:
        lines.append(f"| Avg Duration | {data['avg_duration']:.0f} (tid units) |")

    return "\n".join(lines)


# ─── 15. Key Findings ───────────────────────────────────────────────────────


def section_key_findings(
    aiperf: Optional[AiperfResult],
    prom: Optional[PrometheusSnapshot],
    hw: Optional[dict],
) -> str:
    """Auto-generate insights based on thresholds."""
    findings = []

    if aiperf:
        if aiperf.ttft_p99_ms > 500:
            findings.append(f"High TTFT p99: {aiperf.ttft_p99_ms:.0f}ms (> 500ms)")

    if prom:
        # Check transport breakdown
        if prom.request_plane_roundtrip_ttft_p50 > 0.1:
            findings.append(
                f"High roundtrip TTFT p50: {prom.request_plane_roundtrip_ttft_p50*1000:.0f}ms (> 100ms)"
            )

        # Tokio health
        if prom.tokio_worker_mean_poll_time_ns:
            avg_pt = sum(prom.tokio_worker_mean_poll_time_ns) / len(
                prom.tokio_worker_mean_poll_time_ns
            )
            if avg_pt > 100_000:
                findings.append(
                    f"Tokio avg poll time elevated: {avg_pt:.0f}ns (> 100μs)"
                )
        if prom.tokio_event_loop_stall_total > 10:
            findings.append(
                f"Multiple event loop stalls: {prom.tokio_event_loop_stall_total:.0f}"
            )

    if hw:
        if hw.get("cache-miss-rate", 0) > 5:
            findings.append(f"Cache miss rate: {hw['cache-miss-rate']:.1f}% (> 5%)")
        if hw.get("ipc", 999) < 0.5:
            findings.append(f"Low IPC: {hw['ipc']:.2f} (< 0.5)")

    lines = [_section("15. Key Findings")]
    if not findings:
        lines.append("No anomalies detected - all metrics within expected ranges.")
    else:
        lines.append(f"Found {len(findings)} notable item(s):")
        lines.append("")
        for i, f in enumerate(findings, 1):
            lines.append(f"{i}. {f}")

    return "\n".join(lines)


# ─── Main: assemble report ──────────────────────────────────────────────────


def run_analysis(obs_dir: Path) -> str:
    """Run full analysis and return the report as a string."""
    print(f"Analyzing: {obs_dir}")
    print("")

    # Load data sources
    aiperf = load_aiperf(obs_dir)
    prom = load_prometheus(obs_dir)
    hw = parse_perf_stat(obs_dir)

    # Build report
    sections = [
        section_config(obs_dir),
        section_throughput(aiperf),
        section_latency(aiperf),
        section_lifecycle(prom),
        section_transport_breakdown(prom),
        section_nvtx(obs_dir),
        section_syscall_profile(obs_dir),
        section_tokio(prom),
        section_transport(prom),
        section_compute(prom),
        section_hw_counters(obs_dir),
        section_flamegraph(obs_dir),
        section_bpf(obs_dir),
        section_system_resources(obs_dir),
        section_nsys_context_switches(obs_dir),
        section_key_findings(aiperf, prom, hw),
    ]

    # Filter out None sections
    report_parts = [s for s in sections if s is not None]

    # Header
    header = (
        "# Unified Observability Report\n\n"
        f"**Directory:** `{obs_dir}`\n\n"
        f"**Generated:** {__import__('datetime').datetime.now().isoformat()}\n"
    )

    report = header + "\n".join(report_parts)

    # Summary of what's missing
    missing = []
    if aiperf is None:
        missing.append("aiperf results")
    if prom is None:
        missing.append("Prometheus snapshot")
    if hw is None:
        missing.append("perf stat")
    if not (obs_dir / "nsys" / "frontend.sqlite").exists():
        missing.append("nsys SQLite")
    if not (obs_dir / "bpf").exists() or not any((obs_dir / "bpf").glob("*.txt")):
        missing.append("BPF data")

    if missing:
        report += f"\n\n[Skipped sections — missing data: {', '.join(missing)}]\n"

    return report


def cmd_analyze(args: argparse.Namespace) -> None:
    """Analyze an observability capture directory."""
    if args.obs_dir:
        obs_dir = Path(args.obs_dir)
    else:
        # Auto-find latest
        script_dir = Path(__file__).resolve().parent  # .../analysis/
        repo_root = script_dir.parents[3]  # .../dynamo/ (repo root)
        obs_dir_found = find_latest_obs_dir(repo_root)
        if obs_dir_found is None:
            print("ERROR: No artifacts/obs_* directory found. Specify path explicitly.")
            sys.exit(1)
        obs_dir = obs_dir_found
        print(f"Auto-detected: {obs_dir}")

    if not obs_dir.exists():
        print(f"ERROR: Directory not found: {obs_dir}")
        sys.exit(1)

    report = run_analysis(obs_dir)

    # Print to stdout
    print(report)

    # Also write to report.md
    report_path = obs_dir / "report.md"
    report_path.write_text(report)
    print(f"\nReport saved to: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified observability analysis for dynamo frontend"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_analyze = subparsers.add_parser(
        "analyze", help="Analyze an observability capture directory"
    )
    p_analyze.add_argument(
        "obs_dir",
        nargs="?",
        default=None,
        help="Path to obs_* directory (default: auto-find latest)",
    )
    p_analyze.set_defaults(func=cmd_analyze)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
