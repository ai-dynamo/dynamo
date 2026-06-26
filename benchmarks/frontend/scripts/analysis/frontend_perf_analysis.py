#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Frontend performance analysis script.

Ingests aiperf JSON + Prometheus snapshots and produces:
- Scalability curves (TTFT/ITL/throughput vs concurrency for each ISL)
- KV router A/B comparison
- ISL scaling heatmaps
- Stage waterfall breakdown
- Transport overhead analysis
- Tokio health correlation
- Regression detection (Mann-Whitney U test)

Usage:
    # Analyze a single run
    python frontend_perf_analysis.py analyze <artifact_dir>

    # Compare two runs (A/B or regression)
    python frontend_perf_analysis.py compare <baseline_dir> <candidate_dir>

    # Generate ISL heatmap
    python frontend_perf_analysis.py heatmap <artifact_dir>
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class AiperfResult:
    """Parsed aiperf profile result."""

    concurrency: int = 0
    isl: int = 0
    ttft_p50_ms: float = 0.0
    ttft_p95_ms: float = 0.0
    ttft_p99_ms: float = 0.0
    itl_p50_ms: float = 0.0
    itl_p95_ms: float = 0.0
    itl_p99_ms: float = 0.0
    throughput_tok_s: float = 0.0
    request_throughput_rps: float = 0.0


@dataclass
class PrometheusSnapshot:
    """Parsed Prometheus /metrics snapshot."""

    stage_durations: dict = field(default_factory=dict)  # stage -> {p50, p95, p99}
    request_plane_queue_p50: float = 0.0
    request_plane_send_p50: float = 0.0
    request_plane_roundtrip_ttft_p50: float = 0.0
    request_plane_inflight: float = 0.0
    # Transport breakdown (backend-side, cross-process wall-clock)
    work_handler_network_transit: dict = field(default_factory=dict)  # {p50, p95, p99}
    work_handler_time_to_first_response: dict = field(
        default_factory=dict
    )  # {p50, p95, p99}
    tokio_worker_mean_poll_time_ns: list = field(default_factory=list)
    tokio_event_loop_stall_total: float = 0.0
    tokio_global_queue_depth: float = 0.0
    tokio_budget_forced_yield_total: float = 0.0
    tokio_worker_busy_ratio: list = field(default_factory=list)
    tcp_pool_active: float = 0.0
    tcp_pool_idle: float = 0.0
    compute_pool_active: float = 0.0


@dataclass
class TestPoint:
    """A single test point (concurrency x ISL)."""

    key: str
    concurrency: int
    isl: int
    aiperf: Optional[AiperfResult] = None
    prometheus: Optional[PrometheusSnapshot] = None


def parse_aiperf_json(path: Path) -> Optional[AiperfResult]:
    """Parse aiperf profile_export_aiperf.json."""
    # Look for the aiperf JSON output in the directory
    candidates = [
        path / "profile_export_aiperf.json",
        path / "profile_results.json",
    ]
    # Also try any .json file in the directory
    if path.is_dir():
        candidates.extend(sorted(path.glob("*.json")))

    for candidate in candidates:
        if candidate.exists():
            try:
                with open(candidate) as f:
                    data = json.load(f)
                return _extract_aiperf_metrics(data)
            except (json.JSONDecodeError, KeyError):
                continue
    return None


def _extract_aiperf_metrics(data: dict) -> AiperfResult:
    """Extract metrics from aiperf v0.6+ JSON structure.

    Top-level keys: time_to_first_token, inter_token_latency,
    output_token_throughput, request_throughput, input_config, etc.
    Each latency key is a dict with p50/p95/p99/avg/min/max.
    """
    result = AiperfResult()

    # TTFT — aiperf v0.6 key: time_to_first_token (unit: ms)
    ttft = data.get("time_to_first_token", data.get("ttft", {}))
    if isinstance(ttft, dict):
        result.ttft_p50_ms = float(ttft.get("p50", ttft.get("avg", 0)) or 0)
        result.ttft_p95_ms = float(ttft.get("p95", 0) or 0)
        result.ttft_p99_ms = float(ttft.get("p99", 0) or 0)

    # ITL — aiperf v0.6 key: inter_token_latency (unit: ms)
    itl = data.get("inter_token_latency", data.get("itl", {}))
    if isinstance(itl, dict):
        result.itl_p50_ms = float(itl.get("p50", itl.get("avg", 0)) or 0)
        result.itl_p95_ms = float(itl.get("p95", 0) or 0)
        result.itl_p99_ms = float(itl.get("p99", 0) or 0)

    # Throughput — output_token_throughput.avg (tok/s)
    otput = data.get("output_token_throughput", {})
    result.throughput_tok_s = float(otput.get("avg", 0) or 0)

    # Request throughput — request_throughput.avg (req/s)
    rtput = data.get("request_throughput", {})
    result.request_throughput_rps = float(rtput.get("avg", 0) or 0)

    # Config: concurrency from input_config.loadgen; ISL from input_sequence_length
    cfg = data.get("input_config", {})
    loadgen = cfg.get("loadgen", {})
    result.concurrency = int(loadgen.get("concurrency", 0))

    isl_field = data.get("input_sequence_length", {})
    result.isl = int(isl_field.get("avg", 0) or 0)

    return result


def parse_prometheus_snapshot(path: Path) -> Optional[PrometheusSnapshot]:
    """Parse Prometheus text format snapshot.

    Delegates to parsing_util.parse_prometheus_text() with the conventional
    ``prometheus_snapshot.txt`` filename.
    """
    from parsing_util import parse_prometheus_text

    snapshot_path = path / "prometheus_snapshot.txt"
    if not snapshot_path.exists():
        return None
    return parse_prometheus_text(snapshot_path)


def load_test_points(artifact_dir: Path) -> list[TestPoint]:
    """Load all test points from an artifact directory."""
    points = []

    for subdir in sorted(artifact_dir.iterdir()):
        if not subdir.is_dir():
            continue

        # Parse key like "c10_isl4096"
        match = re.match(r"c(\d+)_isl(\d+)", subdir.name)
        if not match:
            # Try nested epoch directories
            for nested in sorted(subdir.iterdir()):
                if nested.is_dir():
                    match = re.match(r"c(\d+)_isl(\d+)", nested.name)
                    if match:
                        c, isl = int(match.group(1)), int(match.group(2))
                        point = TestPoint(key=nested.name, concurrency=c, isl=isl)
                        point.aiperf = parse_aiperf_json(nested)
                        point.prometheus = parse_prometheus_snapshot(nested)
                        points.append(point)
            continue

        c, isl = int(match.group(1)), int(match.group(2))
        point = TestPoint(key=subdir.name, concurrency=c, isl=isl)
        point.aiperf = parse_aiperf_json(subdir)
        point.prometheus = parse_prometheus_snapshot(subdir)
        points.append(point)

    return points


def print_scalability_table(points: list[TestPoint]) -> None:
    """Print scalability table: TTFT/ITL/throughput vs concurrency for each ISL."""
    isls = sorted(set(p.isl for p in points))
    concurrencies = sorted(set(p.concurrency for p in points))

    for isl in isls:
        print(f"\n{'='*80}")
        print(f"ISL = {isl}")
        print(f"{'='*80}")
        print(
            f"{'Conc':>6} {'TTFT p50':>10} {'TTFT p95':>10} {'ITL p50':>10} "
            f"{'ITL p95':>10} {'Tput tok/s':>12} {'RPS':>8}"
        )
        print("-" * 80)

        for c in concurrencies:
            matching = [
                p for p in points if p.concurrency == c and p.isl == isl and p.aiperf
            ]
            if not matching:
                continue
            a = matching[0].aiperf
            print(
                f"{c:>6} {a.ttft_p50_ms:>10.2f} {a.ttft_p95_ms:>10.2f} "
                f"{a.itl_p50_ms:>10.2f} {a.itl_p95_ms:>10.2f} "
                f"{a.throughput_tok_s:>12.1f} {a.request_throughput_rps:>8.2f}"
            )


def print_stage_waterfall(points: list[TestPoint]) -> None:
    """Print stage breakdown at each load level."""
    print(f"\n{'='*80}")
    print("Pipeline Stage Breakdown (p50 seconds)")
    print(f"{'='*80}")

    stages = ["preprocess", "route", "transport_roundtrip", "postprocess"]
    header = f"{'Key':>20}"
    for s in stages:
        header += f" {s:>18}"
    print(header)
    print("-" * 100)

    for p in sorted(points, key=lambda x: (x.isl, x.concurrency)):
        if not p.prometheus or not p.prometheus.stage_durations:
            continue
        line = f"{p.key:>20}"
        for s in stages:
            val = p.prometheus.stage_durations.get(s, {}).get("p50", 0)
            line += f" {val:>18.6f}"
        print(line)


def print_transport_breakdown(points: list[TestPoint]) -> None:
    """Print transport overhead: queue_seconds vs roundtrip_ttft_seconds."""
    print(f"\n{'='*80}")
    print("Transport Overhead (p50 seconds)")
    print(f"{'='*80}")
    print(f"{'Key':>20} {'Queue (encode)':>16} {'RT TTFT (net)':>16} {'Inflight':>10}")
    print("-" * 70)

    for p in sorted(points, key=lambda x: (x.isl, x.concurrency)):
        if not p.prometheus:
            continue
        print(
            f"{p.key:>20} {p.prometheus.request_plane_queue_p50:>16.6f} "
            f"{p.prometheus.request_plane_roundtrip_ttft_p50:>16.6f} "
            f"{p.prometheus.request_plane_inflight:>10.0f}"
        )


def print_tokio_health(points: list[TestPoint]) -> None:
    """Print tokio health indicators."""
    print(f"\n{'='*80}")
    print("Tokio Health")
    print(f"{'='*80}")
    print(
        f"{'Key':>20} {'Avg Poll ns':>12} {'Max Poll ns':>12} "
        f"{'Stalls':>8} {'Queue':>8} {'Yields':>8} {'Busy Avg':>10}"
    )
    print("-" * 90)

    for p in sorted(points, key=lambda x: (x.isl, x.concurrency)):
        if not p.prometheus:
            continue
        pm = p.prometheus
        avg_poll = (
            sum(pm.tokio_worker_mean_poll_time_ns)
            / len(pm.tokio_worker_mean_poll_time_ns)
            if pm.tokio_worker_mean_poll_time_ns
            else 0
        )
        max_poll = max(pm.tokio_worker_mean_poll_time_ns, default=0)
        avg_busy = (
            sum(pm.tokio_worker_busy_ratio) / len(pm.tokio_worker_busy_ratio)
            if pm.tokio_worker_busy_ratio
            else 0
        )

        # Health indicators
        poll_status = ""
        if avg_poll > 1_000_000:
            poll_status = " STARVING"
        elif avg_poll > 100_000:
            poll_status = " WARN"

        print(
            f"{p.key:>20} {avg_poll:>12.0f} {max_poll:>12.0f} "
            f"{pm.tokio_event_loop_stall_total:>8.0f} "
            f"{pm.tokio_global_queue_depth:>8.0f} "
            f"{pm.tokio_budget_forced_yield_total:>8.0f} "
            f"{avg_busy:>10.3f}{poll_status}"
        )


def compare_runs(
    baseline_dir: Path, candidate_dir: Path, threshold_pct: float = 10.0
) -> None:
    """Compare two runs and flag regressions."""
    baseline_points = load_test_points(baseline_dir)
    candidate_points = load_test_points(candidate_dir)

    if not baseline_points or not candidate_points:
        print("ERROR: No test points found in one or both directories.")
        return

    # Index by key
    baseline_map = {(p.concurrency, p.isl): p for p in baseline_points}
    candidate_map = {(p.concurrency, p.isl): p for p in candidate_points}

    common_keys = sorted(set(baseline_map.keys()) & set(candidate_map.keys()))

    if not common_keys:
        print("ERROR: No matching test points between baseline and candidate.")
        return

    print(f"\n{'='*100}")
    print(
        f"A/B Comparison: {baseline_dir.name} (baseline) vs {candidate_dir.name} (candidate)"
    )
    print(f"Regression threshold: {threshold_pct}%")
    print(f"{'='*100}")
    print(
        f"{'Key':>20} {'TTFT p50 B':>12} {'TTFT p50 C':>12} {'Delta%':>8} "
        f"{'ITL p50 B':>12} {'ITL p50 C':>12} {'Delta%':>8} {'Status':>12}"
    )
    print("-" * 100)

    regressions = []

    for key in common_keys:
        bp = baseline_map[key]
        cp = candidate_map[key]

        if not bp.aiperf or not cp.aiperf:
            continue

        ttft_delta = 0
        if bp.aiperf.ttft_p50_ms > 0:
            ttft_delta = (
                (cp.aiperf.ttft_p50_ms - bp.aiperf.ttft_p50_ms)
                / bp.aiperf.ttft_p50_ms
                * 100
            )

        itl_delta = 0
        if bp.aiperf.itl_p50_ms > 0:
            itl_delta = (
                (cp.aiperf.itl_p50_ms - bp.aiperf.itl_p50_ms)
                / bp.aiperf.itl_p50_ms
                * 100
            )

        status = "OK"
        if ttft_delta > threshold_pct or itl_delta > threshold_pct:
            status = "REGRESSION"
            regressions.append((f"c{key[0]}_isl{key[1]}", ttft_delta, itl_delta))
        elif ttft_delta < -threshold_pct or itl_delta < -threshold_pct:
            status = "IMPROVED"

        print(
            f"{'c' + str(key[0]) + '_isl' + str(key[1]):>20} "
            f"{bp.aiperf.ttft_p50_ms:>12.2f} {cp.aiperf.ttft_p50_ms:>12.2f} "
            f"{ttft_delta:>+7.1f}% "
            f"{bp.aiperf.itl_p50_ms:>12.2f} {cp.aiperf.itl_p50_ms:>12.2f} "
            f"{itl_delta:>+7.1f}% "
            f"{status:>12}"
        )

    # Note: Mann-Whitney U on aggregated p50 values is statistically invalid
    # because p50s across (concurrency, ISL) configs are not IID samples.
    # A valid test would require raw per-request TTFT samples for matched
    # configurations.  Skipped to avoid reporting misleading p-values.

    if regressions:
        print(f"\nWARNING: {len(regressions)} regression(s) detected:")
        for key, ttft_d, itl_d in regressions:
            print(f"  {key}: TTFT {ttft_d:+.1f}%, ITL {itl_d:+.1f}%")


def print_heatmap(points: list[TestPoint]) -> None:
    """Print ISL x concurrency heatmap of TTFT p95."""
    isls = sorted(set(p.isl for p in points))
    concurrencies = sorted(set(p.concurrency for p in points))

    point_map = {(p.concurrency, p.isl): p for p in points}

    print(f"\n{'='*80}")
    print("TTFT p95 Heatmap (ms) — Concurrency x ISL")
    print(f"{'='*80}")

    # Header
    header = f"{'Conc':>8}"
    for isl in isls:
        header += f" {'ISL=' + str(isl):>12}"
    print(header)
    print("-" * (8 + 13 * len(isls)))

    for c in concurrencies:
        line = f"{c:>8}"
        for isl in isls:
            key = (c, isl)
            if key in point_map and point_map[key].aiperf:
                val = point_map[key].aiperf.ttft_p95_ms
                line += f" {val:>12.2f}"
            else:
                line += f" {'---':>12}"
        print(line)


def generate_plots(points: list[TestPoint], output_dir: Path) -> None:
    """Generate matplotlib plots if available."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("\nSKIP: matplotlib/numpy not available for plot generation")
        print("Install: pip install matplotlib numpy")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    isls = sorted(set(p.isl for p in points))
    concurrencies = sorted(set(p.concurrency for p in points))

    # --- Scalability curves ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for isl in isls:
        isl_points = sorted(
            [p for p in points if p.isl == isl and p.aiperf],
            key=lambda x: x.concurrency,
        )
        if not isl_points:
            continue

        cs = [p.concurrency for p in isl_points]
        ttft_p50 = [p.aiperf.ttft_p50_ms for p in isl_points]
        itl_p50 = [p.aiperf.itl_p50_ms for p in isl_points]
        tput = [p.aiperf.throughput_tok_s for p in isl_points]

        axes[0].plot(cs, ttft_p50, "o-", label=f"ISL={isl}")
        axes[1].plot(cs, itl_p50, "o-", label=f"ISL={isl}")
        axes[2].plot(cs, tput, "o-", label=f"ISL={isl}")

    axes[0].set_title("TTFT p50 vs Concurrency")
    axes[0].set_xlabel("Concurrency")
    axes[0].set_ylabel("TTFT p50 (ms)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_title("ITL p50 vs Concurrency")
    axes[1].set_xlabel("Concurrency")
    axes[1].set_ylabel("ITL p50 (ms)")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].set_title("Throughput vs Concurrency")
    axes[2].set_xlabel("Concurrency")
    axes[2].set_ylabel("Throughput (tok/s)")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / "scalability_curves.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'scalability_curves.png'}")

    # --- ISL Heatmap ---
    if len(isls) > 1 and len(concurrencies) > 1:
        point_map = {(p.concurrency, p.isl): p for p in points}
        matrix = np.full((len(concurrencies), len(isls)), np.nan)

        for i, c in enumerate(concurrencies):
            for j, isl in enumerate(isls):
                key = (c, isl)
                if key in point_map and point_map[key].aiperf:
                    matrix[i, j] = point_map[key].aiperf.ttft_p95_ms

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(isls)))
        ax.set_xticklabels([str(x) for x in isls])
        ax.set_yticks(range(len(concurrencies)))
        ax.set_yticklabels([str(x) for x in concurrencies])
        ax.set_xlabel("Input Sequence Length")
        ax.set_ylabel("Concurrency")
        ax.set_title("TTFT p95 (ms) — Concurrency x ISL")
        plt.colorbar(im, label="TTFT p95 (ms)")

        # Annotate cells
        for i in range(len(concurrencies)):
            for j in range(len(isls)):
                if not np.isnan(matrix[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{matrix[i, j]:.1f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )

        plt.tight_layout()
        plt.savefig(output_dir / "isl_heatmap.png", dpi=150)
        plt.close()
        print(f"  Saved: {output_dir / 'isl_heatmap.png'}")

    # --- Stage waterfall ---
    stages = ["preprocess", "route", "transport_roundtrip", "postprocess"]
    stage_data = {}
    labels = []

    for p in sorted(points, key=lambda x: (x.isl, x.concurrency)):
        if not p.prometheus or not p.prometheus.stage_durations:
            continue
        labels.append(p.key)
        for s in stages:
            stage_data.setdefault(s, []).append(
                p.prometheus.stage_durations.get(s, {}).get("p50", 0) * 1000
            )

    if labels and any(stage_data.values()):
        fig, ax = plt.subplots(figsize=(14, 6))
        bottoms = np.zeros(len(labels))
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

        for s, color in zip(stages, colors):
            vals = np.array(stage_data.get(s, [0] * len(labels)))
            ax.bar(labels, vals, bottom=bottoms, label=s, color=color)
            bottoms += vals

        ax.set_xlabel("Test Point")
        ax.set_ylabel("Duration (ms)")
        ax.set_title("Pipeline Stage Breakdown (p50)")
        ax.legend()
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / "stage_waterfall.png", dpi=150)
        plt.close()
        print(f"  Saved: {output_dir / 'stage_waterfall.png'}")


def cmd_analyze(args: argparse.Namespace) -> None:
    """Analyze a single run directory."""
    artifact_dir = Path(args.artifact_dir)
    if not artifact_dir.exists():
        print(f"ERROR: Directory not found: {artifact_dir}")
        sys.exit(1)

    print(f"Loading test points from: {artifact_dir}")
    points = load_test_points(artifact_dir)

    if not points:
        print("ERROR: No test points found. Expected directories like c10_isl4096/")
        sys.exit(1)

    print(f"Found {len(points)} test point(s)")

    print_scalability_table(points)
    print_stage_waterfall(points)
    print_transport_breakdown(points)
    print_tokio_health(points)
    print_heatmap(points)

    if not args.no_plots:
        plot_dir = artifact_dir / "plots"
        print(f"\nGenerating plots in: {plot_dir}")
        generate_plots(points, plot_dir)


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare two run directories."""
    baseline = Path(args.baseline_dir)
    candidate = Path(args.candidate_dir)

    for d in [baseline, candidate]:
        if not d.exists():
            print(f"ERROR: Directory not found: {d}")
            sys.exit(1)

    compare_runs(baseline, candidate, threshold_pct=args.threshold)


def cmd_heatmap(args: argparse.Namespace) -> None:
    """Generate ISL heatmap."""
    artifact_dir = Path(args.artifact_dir)
    points = load_test_points(artifact_dir)

    if not points:
        print("ERROR: No test points found.")
        sys.exit(1)

    print_heatmap(points)

    if not args.no_plots:
        plot_dir = artifact_dir / "plots"
        generate_plots(points, plot_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamo frontend performance analysis")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Analyze a single run")
    p_analyze.add_argument("artifact_dir", help="Directory with test point subdirs")
    p_analyze.add_argument(
        "--no-plots", action="store_true", help="Skip plot generation"
    )
    p_analyze.set_defaults(func=cmd_analyze)

    # compare
    p_compare = subparsers.add_parser("compare", help="Compare two runs (A/B)")
    p_compare.add_argument("baseline_dir", help="Baseline run directory")
    p_compare.add_argument("candidate_dir", help="Candidate run directory")
    p_compare.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="Regression threshold percent (default: 10)",
    )
    p_compare.set_defaults(func=cmd_compare)

    # heatmap
    p_heatmap = subparsers.add_parser("heatmap", help="Generate ISL heatmap")
    p_heatmap.add_argument("artifact_dir", help="Directory with test point subdirs")
    p_heatmap.add_argument(
        "--no-plots", action="store_true", help="Skip plot generation"
    )
    p_heatmap.set_defaults(func=cmd_heatmap)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
