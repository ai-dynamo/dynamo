# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Plot aiperf timeslice metrics comparing MX P2P scale-up vs disk loading.
#
# Usage (auto-discover from artifact directory):
#   python plot_scaleup.py \
#       --artifacts-dir artifacts/Qwen3-32B/staircase_trace/baseline \
#       --slice-duration 60 --scaleup-time 120
#
# Usage (explicit paths):
#   python plot_scaleup.py \
#       --p2p artifacts/.../baseline/p2p/profile_export_aiperf_timeslices.json \
#       --disk artifacts/.../baseline/disk/profile_export_aiperf_timeslices.json \
#       --slice-duration 60
#
# Optional: mark the scale-up trigger time with --scaleup-time 90
# Optional: use --stat p99 to plot p99 instead of avg

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

TIMESLICE_FILENAME = "profile_export_aiperf_timeslices.json"
SERVER_METRICS_FILENAME = "server_metrics_export.jsonl"


@dataclass
class MetricConfig:
    """Definition of a metric to plot."""

    key: str
    label: str
    unit: str
    higher_is_better: bool = False


# Client-side metrics from timeslice export
TIMESLICE_METRICS: list[MetricConfig] = [
    MetricConfig("time_to_first_token", "TTFT", "ms"),
    MetricConfig("inter_token_latency", "ITL", "ms"),
    MetricConfig("request_latency", "Request Latency", "ms"),
    MetricConfig(
        "output_token_throughput",
        "Output Token Throughput",
        "tokens/sec",
        higher_is_better=True,
    ),
    MetricConfig(
        "request_throughput", "Request Throughput", "req/sec", higher_is_better=True
    ),
    MetricConfig("goodput", "Goodput", "req/sec", higher_is_better=True),
]

# Server-side gauge metrics from server_metrics_export.jsonl
# aiperf scrapes the Dynamo frontend's /metrics endpoint, which exposes
# dynamo_frontend_* metrics (not vllm:* which are on the worker pods).
SERVER_METRICS: list[MetricConfig] = [
    MetricConfig("dynamo_frontend_queued_requests", "Queue Depth", "requests"),
    MetricConfig("dynamo_frontend_inflight_requests", "Inflight Requests", "requests"),
]


def load_timeslices(path: Path) -> list[dict]:
    """Load timeslice data from aiperf JSON export."""
    with open(path) as f:
        data = json.load(f)
    timeslices = data.get("timeslices", data)
    if isinstance(timeslices, dict):
        raise ValueError(
            f"Unexpected format in {path}. Expected 'timeslices' key with a list."
        )
    return sorted(timeslices, key=lambda t: t.get("timeslice_index", 0))


def extract_timeslice_series(
    timeslices: list[dict],
    metric_key: str,
    stat: str,
    slice_duration: float,
) -> tuple[list[float], list[float]]:
    """Extract (time, value) series from timeslice data for a given metric."""
    times: list[float] = []
    values: list[float] = []
    for ts in timeslices:
        metric = ts.get(metric_key)
        if metric is None:
            continue
        val = metric.get(stat)
        if val is None:
            continue
        idx = ts.get("timeslice_index", 0)
        times.append(idx * slice_duration + slice_duration / 2)
        values.append(float(val))
    return times, values


def load_server_metrics(
    path: Path, metric_name: str, bin_seconds: float
) -> tuple[list[float], list[float]]:
    """Load a gauge metric from server_metrics_export.jsonl and bin by time.

    The JSONL format has one line per scrape with all metrics nested:
      {"timestamp_ns": ..., "metrics": {"metric_name": [{"value": ..., "labels": ...}], ...}}

    For gauge metrics with labels, sums the values across all label sets
    (e.g., multiple model labels).

    Returns:
        Tuple of (bin_midpoints_seconds, bin_avg_values).
    """
    timestamps_ns: list[int] = []
    values: list[float] = []
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            ts = record.get("timestamp_ns")
            if ts is None:
                continue
            metrics = record.get("metrics", {})
            entries = metrics.get(metric_name)
            if not entries:
                continue
            # Sum across all label sets (usually just one)
            total = sum(e.get("value", 0) for e in entries if "value" in e)
            timestamps_ns.append(ts)
            values.append(float(total))

    if not timestamps_ns:
        return [], []

    t0 = min(timestamps_ns)
    rel_seconds = [(t - t0) / 1e9 for t in timestamps_ns]
    max_t = max(rel_seconds)

    # Bin into intervals
    n_bins = max(1, int(max_t / bin_seconds) + 1)
    bin_sums = [0.0] * n_bins
    bin_counts = [0] * n_bins
    for t, v in zip(rel_seconds, values):
        idx = min(int(t / bin_seconds), n_bins - 1)
        bin_sums[idx] += v
        bin_counts[idx] += 1

    times: list[float] = []
    avgs: list[float] = []
    for i in range(n_bins):
        if bin_counts[i] > 0:
            times.append(i * bin_seconds + bin_seconds / 2)
            avgs.append(bin_sums[i] / bin_counts[i])
    return times, avgs


def _plot_single(
    ax: plt.Axes,
    p2p_t: list[float],
    p2p_v: list[float],
    disk_t: list[float],
    disk_v: list[float],
    metric: MetricConfig,
    stat_label: str,
    scaleup_time: Optional[float],
    p2p_label: str,
    disk_label: str,
) -> None:
    """Plot a single metric on the given axes."""
    if p2p_t:
        ax.plot(
            p2p_t,
            p2p_v,
            "-o",
            markersize=3,
            linewidth=1.5,
            label=p2p_label,
            color="#1f77b4",
        )
    if disk_t:
        ax.plot(
            disk_t,
            disk_v,
            "-s",
            markersize=3,
            linewidth=1.5,
            label=disk_label,
            color="#ff7f0e",
        )
    if scaleup_time is not None:
        ax.axvline(
            scaleup_time,
            color="red",
            linestyle="--",
            alpha=0.7,
            label="Scale-up trigger",
        )
    ax.set_title(f"{metric.label} ({stat_label})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"{metric.label} ({metric.unit})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_comparison(
    p2p_timeslice_path: Path,
    disk_timeslice_path: Path,
    slice_duration: float,
    stat: str,
    output_path: Path,
    scaleup_time: Optional[float] = None,
    p2p_label: str = "MX P2P",
    disk_label: str = "Disk",
    p2p_server_metrics_path: Optional[Path] = None,
    disk_server_metrics_path: Optional[Path] = None,
) -> None:
    """Generate comparison plots for all metrics."""
    p2p_slices = load_timeslices(p2p_timeslice_path)
    disk_slices = load_timeslices(disk_timeslice_path)

    # Collect available timeslice metrics
    available_timeslice: list[MetricConfig] = []
    for m in TIMESLICE_METRICS:
        p2p_t, _ = extract_timeslice_series(p2p_slices, m.key, stat, slice_duration)
        disk_t, _ = extract_timeslice_series(disk_slices, m.key, stat, slice_duration)
        if p2p_t or disk_t:
            available_timeslice.append(m)

    # Collect available server metrics
    available_server: list[MetricConfig] = []
    if p2p_server_metrics_path and disk_server_metrics_path:
        for m in SERVER_METRICS:
            p2p_t, _ = load_server_metrics(
                p2p_server_metrics_path, m.key, slice_duration
            )
            disk_t, _ = load_server_metrics(
                disk_server_metrics_path, m.key, slice_duration
            )
            if p2p_t or disk_t:
                available_server.append(m)

    all_metrics = available_timeslice + available_server
    if not all_metrics:
        print(f"No metrics found with stat '{stat}'. Try --stat avg or --stat p50.")
        return

    n_metrics = len(all_metrics)
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    idx = 0
    # Plot timeslice metrics
    for m in available_timeslice:
        p2p_t, p2p_v = extract_timeslice_series(p2p_slices, m.key, stat, slice_duration)
        disk_t, disk_v = extract_timeslice_series(
            disk_slices, m.key, stat, slice_duration
        )
        _plot_single(
            axes[idx],
            p2p_t,
            p2p_v,
            disk_t,
            disk_v,
            m,
            stat,
            scaleup_time,
            p2p_label,
            disk_label,
        )
        idx += 1

    # Plot server metrics
    for m in available_server:
        p2p_t, p2p_v = load_server_metrics(
            p2p_server_metrics_path, m.key, slice_duration
        )
        disk_t, disk_v = load_server_metrics(
            disk_server_metrics_path, m.key, slice_duration
        )
        _plot_single(
            axes[idx],
            p2p_t,
            p2p_v,
            disk_t,
            disk_v,
            m,
            "avg",
            scaleup_time,
            p2p_label,
            disk_label,
        )
        idx += 1

    # Hide unused subplots
    for j in range(n_metrics, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"MX P2P vs Disk Loading: Inference Metrics Over Time ({stat})",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot aiperf timeslice comparison: MX P2P vs disk loading"
    )

    # Auto-discover mode
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Artifact root containing p2p/ and disk/ subdirs "
        "(e.g., artifacts/Llama-3.3-70B-Instruct/staircase_trace)",
    )

    # Explicit path mode
    parser.add_argument(
        "--p2p",
        type=Path,
        default=None,
        help="Explicit path to P2P timeslice JSON (overrides --artifacts-dir)",
    )
    parser.add_argument(
        "--disk",
        type=Path,
        default=None,
        help="Explicit path to disk timeslice JSON (overrides --artifacts-dir)",
    )

    parser.add_argument(
        "--slice-duration",
        type=float,
        required=True,
        help="Timeslice duration in seconds (must match aiperf's --slice-duration)",
    )
    parser.add_argument(
        "--stat",
        type=str,
        default="avg",
        help="Statistic to plot: avg, p50, p90, p99, min, max (default: avg)",
    )
    parser.add_argument(
        "--scaleup-time",
        type=float,
        default=None,
        help="Time in seconds when scale-up was triggered (draws vertical line)",
    )
    parser.add_argument(
        "--p2p-label",
        type=str,
        default="MX P2P",
        help="Legend label for P2P run (default: 'MX P2P')",
    )
    parser.add_argument(
        "--disk-label",
        type=str,
        default="Disk",
        help="Legend label for disk run (default: 'Disk')",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path (default: <artifacts-dir>/scaleup_comparison.png)",
    )
    args = parser.parse_args()

    # Resolve paths
    p2p_path = args.p2p
    disk_path = args.disk
    p2p_server = None
    disk_server = None

    if args.artifacts_dir is not None:
        if p2p_path is None:
            p2p_path = args.artifacts_dir / "p2p" / TIMESLICE_FILENAME
        if disk_path is None:
            disk_path = args.artifacts_dir / "disk" / TIMESLICE_FILENAME
        # Auto-discover server metrics JSONL
        p2p_candidate = args.artifacts_dir / "p2p" / SERVER_METRICS_FILENAME
        disk_candidate = args.artifacts_dir / "disk" / SERVER_METRICS_FILENAME
        if p2p_candidate.exists() and disk_candidate.exists():
            p2p_server = p2p_candidate
            disk_server = disk_candidate

    if p2p_path is None or disk_path is None:
        parser.error("Provide --artifacts-dir or both --p2p and --disk")

    for label, path in [("P2P", p2p_path), ("Disk", disk_path)]:
        if not path.exists():
            parser.error(f"{label} timeslice file not found: {path}")

    output_path = args.output
    if output_path is None:
        if args.artifacts_dir is not None:
            output_path = args.artifacts_dir / "scaleup_comparison.png"
        else:
            output_path = Path("scaleup_comparison.png")

    if p2p_server:
        print(
            f"Server metrics found: {p2p_server.parent.name}/, {disk_server.parent.name}/"
        )

    plot_comparison(
        p2p_timeslice_path=p2p_path,
        disk_timeslice_path=disk_path,
        slice_duration=args.slice_duration,
        stat=args.stat,
        output_path=output_path,
        scaleup_time=args.scaleup_time,
        p2p_label=args.p2p_label,
        disk_label=args.disk_label,
        p2p_server_metrics_path=p2p_server,
        disk_server_metrics_path=disk_server,
    )


if __name__ == "__main__":
    main()
