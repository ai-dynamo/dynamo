# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Any

from .common import finite_number, linear_percentile

def metric_samples(
    samples: Iterable[Mapping[str, Any]], metric_name: str
) -> list[float]:
    values: list[float] = []
    for sample in samples:
        metrics = sample.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        value = finite_number(metrics.get(metric_name))
        if value is not None:
            values.append(value)
    return values


def summarize_samples(
    samples: Sequence[Mapping[str, Any]], counts: Sequence[int], repetitions: int
) -> list[dict[str, Any]]:
    """Calculate median/IQR throughput and latency summaries for each count."""

    valid_by_count: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for sample in samples:
        count = sample.get("frontend_count")
        if sample.get("valid") is True and isinstance(count, int):
            valid_by_count[count].append(sample)

    summaries: list[dict[str, Any]] = []
    baseline_median: float | None = None
    baseline_count = counts[0]
    for count in counts:
        valid_samples = valid_by_count[count]
        rps_values = metric_samples(valid_samples, "request_throughput_rps")
        rps_median = float(median(rps_values)) if rps_values else None
        if count == baseline_count:
            baseline_median = rps_median
        metric_medians = {
            metric_name: (
                float(median(values))
                if (values := metric_samples(valid_samples, metric_name))
                else None
            )
            for metric_name in (
                "ttft_ms_p50",
                "ttft_ms_p95",
                "itl_ms_p50",
                "itl_ms_p95",
                "request_latency_ms_p50",
                "request_latency_ms_p95",
                "isl_tokens_avg",
                "osl_tokens_avg",
                "output_token_throughput_tps",
            )
        }
        summaries.append(
            {
                "frontend_count": count,
                "planned_samples": repetitions,
                "valid_samples": len(valid_samples),
                "rps_samples": rps_values,
                "request_throughput_rps_median": rps_median,
                "request_throughput_rps_p25": linear_percentile(rps_values, 0.25),
                "request_throughput_rps_p75": linear_percentile(rps_values, 0.75),
                "request_throughput_rps_min": min(rps_values) if rps_values else None,
                "request_throughput_rps_max": max(rps_values) if rps_values else None,
                "median_metrics": metric_medians,
            }
        )

    if baseline_median is not None and baseline_median > 0.0:
        for summary in summaries:
            value = summary["request_throughput_rps_median"]
            if isinstance(value, float):
                speedup = value / baseline_median
                summary["speedup_vs_baseline"] = speedup
                summary["improvement_percent_vs_baseline"] = (speedup - 1.0) * 100.0
                summary["rps_per_frontend"] = value / summary["frontend_count"]
            else:
                summary["speedup_vs_baseline"] = None
                summary["improvement_percent_vs_baseline"] = None
                summary["rps_per_frontend"] = None
    else:
        for summary in summaries:
            summary["speedup_vs_baseline"] = None
            summary["improvement_percent_vs_baseline"] = None
            summary["rps_per_frontend"] = None
    return summaries


def sample_csv_rows(samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        metrics = sample.get("metrics")
        metrics = metrics if isinstance(metrics, Mapping) else {}
        rows.append(
            {
                "sample_index": sample.get("sample_index"),
                "repetition": sample.get("repetition"),
                "frontend_count": sample.get("frontend_count"),
                "valid": sample.get("valid"),
                "child_returncode": sample.get("child_returncode"),
                "child_status": sample.get("child_status"),
                "elapsed_seconds": sample.get("elapsed_seconds"),
                "request_throughput_rps": metrics.get("request_throughput_rps"),
                "ttft_ms_p50": metrics.get("ttft_ms_p50"),
                "ttft_ms_p95": metrics.get("ttft_ms_p95"),
                "itl_ms_p50": metrics.get("itl_ms_p50"),
                "itl_ms_p95": metrics.get("itl_ms_p95"),
                "request_latency_ms_p50": metrics.get("request_latency_ms_p50"),
                "request_latency_ms_p95": metrics.get("request_latency_ms_p95"),
                "isl_tokens_avg": metrics.get("isl_tokens_avg"),
                "osl_tokens_avg": metrics.get("osl_tokens_avg"),
                "output_token_throughput_tps": metrics.get(
                    "output_token_throughput_tps"
                ),
                "completed_profiling_records": sample.get(
                    "completed_profiling_records"
                ),
                "errored_profiling_records": sample.get("errored_profiling_records"),
                "observed_aiperf_errors": sample.get("observed_aiperf_errors"),
                "aiperf_input_sha256": sample.get("aiperf_input_sha256"),
                "valkey_final_replica_synchronized": sample.get(
                    "valkey_final_replica_synchronized"
                ),
                "valkey_final_connected_replicas": sample.get(
                    "valkey_final_connected_replicas"
                ),
                "valkey_final_good_replicas": sample.get("valkey_final_good_replicas"),
                "valkey_final_primary_replid": sample.get(
                    "valkey_final_primary_replid"
                ),
                "valkey_final_replica_replid": sample.get(
                    "valkey_final_replica_replid"
                ),
                "valkey_final_primary_replication_offset": sample.get(
                    "valkey_final_primary_replication_offset"
                ),
                "valkey_final_replica_replication_offset": sample.get(
                    "valkey_final_replica_replication_offset"
                ),
                "valkey_final_replication_offset_delta": sample.get(
                    "valkey_final_replication_offset_delta"
                ),
                "valkey_final_replica_at_least_primary_snapshot": sample.get(
                    "valkey_final_replica_at_least_primary_snapshot"
                ),
                "valkey_final_renew_failed_calls": sample.get(
                    "valkey_final_renew_failed_calls"
                ),
                "valkey_final_renew_commandstats": sample.get(
                    "valkey_final_renew_commandstats"
                ),
                "validation_errors": "; ".join(sample.get("validation_errors", [])),
                "child_output_dir": sample.get("child_output_dir"),
                "log_path": sample.get("log_path"),
            }
        )
    return rows


def summary_csv_rows(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        metrics = summary.get("median_metrics")
        metrics = metrics if isinstance(metrics, Mapping) else {}
        rows.append(
            {
                "frontend_count": summary.get("frontend_count"),
                "planned_samples": summary.get("planned_samples"),
                "valid_samples": summary.get("valid_samples"),
                "request_throughput_rps_median": summary.get(
                    "request_throughput_rps_median"
                ),
                "request_throughput_rps_p25": summary.get("request_throughput_rps_p25"),
                "request_throughput_rps_p75": summary.get("request_throughput_rps_p75"),
                "request_throughput_rps_min": summary.get("request_throughput_rps_min"),
                "request_throughput_rps_max": summary.get("request_throughput_rps_max"),
                "speedup_vs_baseline": summary.get("speedup_vs_baseline"),
                "improvement_percent_vs_baseline": summary.get(
                    "improvement_percent_vs_baseline"
                ),
                "rps_per_frontend": summary.get("rps_per_frontend"),
                "ttft_ms_p50_median": metrics.get("ttft_ms_p50"),
                "ttft_ms_p95_median": metrics.get("ttft_ms_p95"),
                "itl_ms_p50_median": metrics.get("itl_ms_p50"),
                "itl_ms_p95_median": metrics.get("itl_ms_p95"),
                "request_latency_ms_p50_median": metrics.get("request_latency_ms_p50"),
                "request_latency_ms_p95_median": metrics.get("request_latency_ms_p95"),
                "isl_tokens_avg_median": metrics.get("isl_tokens_avg"),
                "osl_tokens_avg_median": metrics.get("osl_tokens_avg"),
                "output_token_throughput_tps_median": metrics.get(
                    "output_token_throughput_tps"
                ),
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
