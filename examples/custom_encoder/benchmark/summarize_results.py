# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize aiperf results and CustomEncoder stage/graph selection logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

TIMING_MARKER = "custom_encoder_timing"
GRAPH_MARKER = "custom_encoder_graph"
FIELD_PATTERN = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)=([^\s]+)")
PREFIX_CACHE_PATTERN = re.compile(r"Prefix cache hit rate: ([0-9.]+)%")


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


def _run_name(root: Path, result_path: Path) -> str:
    try:
        return str(result_path.parent.relative_to(root))
    except ValueError:
        return str(result_path.parent)


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
        loadgen = document.get("input_config", {}).get("loadgen", {})
        aiperf_rows.append(
            {
                "run": run,
                "concurrency": loadgen.get("concurrency", ""),
                "request_count": _metric(document, "request_count"),
                "mean_isl": _metric(document, "input_sequence_length"),
                "mean_osl": _metric(document, "output_sequence_length"),
                "request_throughput_rps": _metric(document, "request_throughput"),
                "output_throughput_tps": _metric(document, "output_token_throughput"),
                "latency_mean_ms": _metric(document, "request_latency"),
                "latency_p50_ms": _metric(document, "request_latency", "p50"),
                "latency_p95_ms": _metric(document, "request_latency", "p95"),
                "errors": len(document.get("error_summary", [])),
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
        "concurrency",
        "request_count",
        "mean_isl",
        "mean_osl",
        "request_throughput_rps",
        "output_throughput_tps",
        "latency_mean_ms",
        "latency_p50_ms",
        "latency_p95_ms",
        "prefix_cache_hit_rate_pct",
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
    _print_table("custom encoder stage timings", timing_rows, timing_fields)
    _print_table("CUDA graph bucket histogram", graph_rows, graph_fields)

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(args.output_dir / "aiperf_summary.csv", aiperf_rows, aiperf_fields)
        _write_csv(
            args.output_dir / "stage_timing_summary.csv", timing_rows, timing_fields
        )
        _write_csv(args.output_dir / "cuda_graph_summary.csv", graph_rows, graph_fields)


if __name__ == "__main__":
    main()
