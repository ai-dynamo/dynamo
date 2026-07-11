# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize aiperf results and CustomEncoder stage/graph selection logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

TIMING_MARKER = "custom_encoder_timing"
GRAPH_MARKER = "custom_encoder_graph"
FIELD_PATTERN = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)=([^\s]+)")
PREFIX_CACHE_PATTERN = re.compile(r"Prefix cache hit rate: ([0-9.]+)%")
REPEAT_PATTERN = re.compile(r"^rep(?:eat)?[-_]?([0-9]+)$")
T_CRITICAL_95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}


def _fields_after_marker(line: str, marker: str) -> dict[str, str] | None:
    marker_index = line.find(marker)
    if marker_index < 0:
        return None
    return dict(FIELD_PATTERN.findall(line[marker_index + len(marker) :]))


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    rank = (len(ordered) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _metric(document: dict[str, Any], name: str, statistic: str = "avg") -> Any:
    value = document.get(name, {})
    return value.get(statistic, "") if isinstance(value, dict) else ""


def _profiling_concurrency(document: dict[str, Any]) -> Any:
    phases = document.get("input_config", {}).get("phases", [])
    for phase in phases:
        if isinstance(phase, dict) and phase.get("name") == "profiling":
            return phase.get("concurrency", "")
    return ""


def _resource_peaks(path: Path) -> dict[str, float | int | str]:
    peaks: dict[str, float | int | str] = {
        "peak_gpu_memory_mib": "",
        "peak_process_count": "",
        "peak_total_rss_kib": "",
    }
    if not path.is_file():
        return peaks

    gpu_memory: list[float] = []
    process_counts: list[int] = []
    total_rss: list[int] = []
    with path.open("r", encoding="utf-8", newline="") as resource_file:
        for row in csv.DictReader(resource_file):
            try:
                gpu_value = float(row["gpu_memory_mib"])
                process_value = int(row["process_count"])
                rss_value = int(row["total_rss_kib"])
            except (KeyError, TypeError, ValueError):
                continue
            gpu_memory.append(gpu_value)
            process_counts.append(process_value)
            total_rss.append(rss_value)
    if gpu_memory:
        peaks["peak_gpu_memory_mib"] = max(gpu_memory)
        peaks["peak_process_count"] = max(process_counts)
        peaks["peak_total_rss_kib"] = max(total_rss)
    return peaks


def _run_name(root: Path, result_path: Path) -> str:
    try:
        return str(result_path.parent.relative_to(root))
    except ValueError:
        return str(result_path.parent)


def _run_dimensions(root: Path, result_path: Path) -> tuple[str, str]:
    try:
        parts = result_path.parent.relative_to(root).parts
    except ValueError:
        parts = result_path.parent.parts
    mode = next((part for part in parts if part in {"async-mp", "sync-inproc"}), "")
    repeat = ""
    for part in parts:
        match = REPEAT_PATTERN.match(part)
        if match is not None:
            repeat = match.group(1)
            break
    return mode, repeat


def _log_paths(run_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in run_dir.rglob("*.log")
        if path.is_file() and path.name != "aiperf.log"
    )


def _parse_logs(
    paths: Iterable[Path],
) -> tuple[
    dict[tuple[str, str, str, str], list[float]],
    Counter[tuple[str, str, str]],
    float | None,
]:
    stage_samples: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    graph_picks: Counter[tuple[str, str, str]] = Counter()
    prefix_cache_hit_rate = None
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as log_file:
            for line in log_file:
                prefix_match = PREFIX_CACHE_PATTERN.search(line)
                if prefix_match is not None:
                    prefix_cache_hit_rate = float(prefix_match.group(1))
                timing_fields = _fields_after_marker(line, TIMING_MARKER)
                if timing_fields is not None:
                    stage = timing_fields.get("stage")
                    elapsed_ms = timing_fields.get("elapsed_ms")
                    if stage is not None and elapsed_ms is not None:
                        try:
                            sample_key = (
                                stage,
                                timing_fields.get("batch_size", "unknown"),
                                timing_fields.get("bucket", "unknown"),
                                timing_fields.get("cost", "unknown"),
                            )
                            stage_samples[sample_key].append(float(elapsed_ms))
                        except ValueError:
                            pass

                graph_fields = _fields_after_marker(line, GRAPH_MARKER)
                if graph_fields is not None:
                    selected_bucket = graph_fields.get("selected_bucket", "unknown")
                    batch_size = graph_fields.get("batch_size", "unknown")
                    actual_cost = graph_fields.get("actual_cost", "unknown")
                    graph_picks[(selected_bucket, batch_size, actual_cost)] += 1
    return stage_samples, graph_picks, prefix_cache_hit_rate


def summarize(
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    aiperf_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []

    for result_path in sorted(root.rglob("profile_export_aiperf.json")):
        with result_path.open("r", encoding="utf-8") as result_file:
            document = json.load(result_file)
        run = _run_name(root, result_path)
        mode, repeat = _run_dimensions(root, result_path)
        aiperf_rows.append(
            {
                "run": run,
                "mode": mode,
                "repeat": repeat,
                "concurrency": _profiling_concurrency(document),
                "request_count": _metric(document, "request_count"),
                "mean_isl": _metric(document, "input_sequence_length"),
                "mean_osl": _metric(document, "output_sequence_length"),
                "request_throughput_rps": _metric(document, "request_throughput"),
                "output_throughput_tps": _metric(document, "output_token_throughput"),
                "latency_mean_ms": _metric(document, "request_latency"),
                "latency_p50_ms": _metric(document, "request_latency", "p50"),
                "latency_p95_ms": _metric(document, "request_latency", "p95"),
                "latency_p99_ms": _metric(document, "request_latency", "p99"),
                "ttft_p50_ms": _metric(document, "time_to_first_token", "p50"),
                "ttft_p95_ms": _metric(document, "time_to_first_token", "p95"),
                "ttft_p99_ms": _metric(document, "time_to_first_token", "p99"),
                "itl_p50_ms": _metric(document, "inter_token_latency", "p50"),
                "itl_p95_ms": _metric(document, "inter_token_latency", "p95"),
                "itl_p99_ms": _metric(document, "inter_token_latency", "p99"),
                "errors": len(document.get("error_summary", [])),
                **_resource_peaks(result_path.parent.parent / "resources.csv"),
            }
        )

        stage_samples, graph_picks, prefix_cache_hit_rate = _parse_logs(
            _log_paths(result_path.parent)
        )
        aiperf_rows[-1]["prefix_cache_hit_rate_pct"] = (
            "" if prefix_cache_hit_rate is None else prefix_cache_hit_rate
        )
        for (stage, batch_size, bucket, cost), samples in sorted(stage_samples.items()):
            timing_rows.append(
                {
                    "run": run,
                    "stage": stage,
                    "batch_size": batch_size,
                    "bucket": bucket,
                    "cost": cost,
                    "count": len(samples),
                    "mean_ms": sum(samples) / len(samples),
                    "p50_ms": _percentile(samples, 0.50),
                    "p95_ms": _percentile(samples, 0.95),
                }
            )
        for (selected_bucket, batch_size, actual_cost), count in sorted(
            graph_picks.items()
        ):
            graph_rows.append(
                {
                    "run": run,
                    "selected_bucket": selected_bucket,
                    "batch_size": batch_size,
                    "actual_cost": actual_cost,
                    "count": count,
                }
            )
    return aiperf_rows, timing_rows, graph_rows


def paired_comparisons(aiperf_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "request_throughput_rps",
        "output_throughput_tps",
        "latency_p99_ms",
        "ttft_p99_ms",
        "itl_p99_ms",
        "peak_gpu_memory_mib",
        "peak_total_rss_kib",
    ]
    by_pair: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in aiperf_rows:
        mode = str(row.get("mode", ""))
        repeat = str(row.get("repeat", ""))
        concurrency = str(row.get("concurrency", ""))
        if mode and repeat and concurrency:
            by_pair[(repeat, concurrency)][mode] = row

    samples: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (_, concurrency), modes in by_pair.items():
        if "async-mp" not in modes or "sync-inproc" not in modes:
            continue
        for metric in metrics:
            async_value = modes["async-mp"].get(metric)
            sync_value = modes["sync-inproc"].get(metric)
            if not isinstance(async_value, (int, float)) or not isinstance(
                sync_value, (int, float)
            ):
                continue
            if async_value == 0:
                continue
            samples[(concurrency, metric)].append(
                (sync_value / async_value - 1.0) * 100.0
            )

    rows: list[dict[str, Any]] = []
    for (concurrency, metric), deltas in sorted(samples.items()):
        sample_count = len(deltas)
        mean_delta = statistics.mean(deltas)
        if sample_count >= 2:
            critical = T_CRITICAL_95.get(sample_count, 1.96)
            half_width = critical * statistics.stdev(deltas) / math.sqrt(sample_count)
        else:
            half_width = math.nan
        rows.append(
            {
                "concurrency": concurrency,
                "metric": metric,
                "pairs": sample_count,
                "mean_delta_pct": mean_delta,
                "median_delta_pct": statistics.median(deltas),
                "ci95_low_pct": mean_delta - half_width,
                "ci95_high_pct": mean_delta + half_width,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _print_table(title: str, rows: list[dict[str, Any]], fields: list[str]) -> None:
    print(f"\n{title}")
    if not rows:
        print("(no rows)")
        return
    widths = {
        field: max(len(field), *(len(str(row.get(field, ""))) for row in rows))
        for field in fields
    }
    print("  ".join(field.ljust(widths[field]) for field in fields))
    print("  ".join("-" * widths[field] for field in fields))
    for row in rows:
        print(
            "  ".join(str(row.get(field, "")).ljust(widths[field]) for field in fields)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Sweep or single-run result directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Also write aiperf, timing, and graph summary CSV files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aiperf_rows, timing_rows, graph_rows = summarize(args.root.resolve())
    aiperf_fields = [
        "run",
        "mode",
        "repeat",
        "concurrency",
        "request_count",
        "mean_isl",
        "mean_osl",
        "request_throughput_rps",
        "output_throughput_tps",
        "latency_mean_ms",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_p99_ms",
        "ttft_p50_ms",
        "ttft_p95_ms",
        "ttft_p99_ms",
        "itl_p50_ms",
        "itl_p95_ms",
        "itl_p99_ms",
        "prefix_cache_hit_rate_pct",
        "peak_gpu_memory_mib",
        "peak_process_count",
        "peak_total_rss_kib",
        "errors",
    ]
    timing_fields = [
        "run",
        "stage",
        "batch_size",
        "bucket",
        "cost",
        "count",
        "mean_ms",
        "p50_ms",
        "p95_ms",
    ]
    graph_fields = [
        "run",
        "selected_bucket",
        "batch_size",
        "actual_cost",
        "count",
    ]
    _print_table("aiperf", aiperf_rows, aiperf_fields)
    paired_rows = paired_comparisons(aiperf_rows)
    paired_fields = [
        "concurrency",
        "metric",
        "pairs",
        "mean_delta_pct",
        "median_delta_pct",
        "ci95_low_pct",
        "ci95_high_pct",
    ]
    _print_table("paired sync-inproc vs async-mp", paired_rows, paired_fields)
    _print_table("custom encoder stage timings", timing_rows, timing_fields)
    _print_table("CUDA graph bucket histogram", graph_rows, graph_fields)

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(args.output_dir / "aiperf_summary.csv", aiperf_rows, aiperf_fields)
        _write_csv(args.output_dir / "paired_summary.csv", paired_rows, paired_fields)
        _write_csv(
            args.output_dir / "stage_timing_summary.csv", timing_rows, timing_fields
        )
        _write_csv(args.output_dir / "cuda_graph_summary.csv", graph_rows, graph_fields)


if __name__ == "__main__":
    main()
