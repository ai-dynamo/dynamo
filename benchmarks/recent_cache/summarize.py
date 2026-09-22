# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export explicitly selected aggregate replay metrics as a portable CSV.

Example::

    python benchmarks/recent_cache/summarize.py \
        --runs /path/to/comparison-a /path/to/comparison-b --output results.csv

Each input may be one runner group or a parent containing multiple groups.
Only manifest.json and row.json are read. Raw traces, per-request reports, logs,
host details, arbitrary configuration, and original error text are never exported.
Missing optional diagnostics remain blank; an observed zero remains zero.
Names with paths, URLs, UUIDs, or long hexadecimal identifiers are redacted.
Review the resulting aggregate CSV before publishing it.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

LABEL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}\Z")
IDENTIFIER_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}" r"|[0-9a-f]{24,}",
    re.IGNORECASE,
)
TRACE_PREFIXES = (
    "mooncake",
    "toolagent",
    "weka",
    "opus48",
    "behavior-smoke",
    "workload",
    "trace",
    "shared",
    "private",
    "repeat",
)
ROW_METRICS = (
    "num_requests",
    "completed_requests",
    "duration_ms",
    "output_throughput_tok_s",
    "request_throughput_rps",
    "output_throughput_tok_s_per_gpu",
    "request_throughput_rps_per_gpu",
    "mean_ttft_ms",
    "p90_ttft_ms",
    "mean_e2e_latency_ms",
    "p90_e2e_latency_ms",
    "p99_e2e_latency_ms",
    "total_trajectories",
    "completed_trajectories",
    "p90_trajectory_e2e_latency_ms",
    "first_admission_prefix_cache_reused_ratio",
    "route_overpredicted_fraction",
    "route_overpredicted_input_fraction",
    "requests_readmitted",
    "requests_readmitted_fraction",
    "readmission_events",
    "max_readmissions_per_request",
    "vllm_preemption_log_capture_configured",
    "vllm_preemptions_total",
    "logged_vllm_preemption_events",
    "logged_vllm_preempted_requests",
    "max_logged_vllm_preemptions_per_request",
)
HOT_METRICS = (
    "decisions",
    "mean_decision_pressure",
    "decision_fraction_above_threshold",
    "boosted_decision_fraction",
    "pressure_upward_crossings",
    "pressure_downward_crossings",
    "peak_footprint",
    "final_count",
    "final_capacity",
)
RESIDENT_FINAL_METRICS = (
    "resident_copies",
    "unique_blocks",
    "excess_copies",
    "replication_factor",
    "total_capacity",
    "occupancy_ratio",
)
RESIDENT_COUNTERS = (
    "store_events",
    "added_copies",
    "duplicate_stores",
    "removal_events",
    "removed_copies",
    "unknown_removals",
    "global_final_copy_removals",
    "clear_events",
    "cleared_copies",
    "worker_removals",
    "topology_removed_copies",
)
FIELDS = (
    "group",
    "run_name",
    "source_family",
    "trace_basename",
    "trace_path_count",
    "status",
    "error_category",
    "expected_requests",
    "workers",
    "tp_size",
    "kv_blocks_per_worker",
    "engine_block_size",
    "trace_block_size",
    "arrival_speedup",
    "engine_speedup",
    "ttl_seconds",
    "threshold",
    "base_overlap_credit",
    "boost_credit",
    "mode",
    "predict",
    "smg",
    *ROW_METRICS,
    *(f"hot_{key}" for key in HOT_METRICS),
    *(f"resident_final_{key}" for key in RESIDENT_FINAL_METRICS),
    "resident_peak_copies",
    "resident_peak_excess_copies",
    *(f"resident_{key}" for key in RESIDENT_COUNTERS),
)


def safe_label(value: Any) -> str:
    if not isinstance(value, str):
        return "redacted"
    if not LABEL_PATTERN.fullmatch(value) or IDENTIFIER_PATTERN.search(value):
        return "redacted"
    return value


def number(value: Any) -> int | float | str:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    return ""


def mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def error_category(row: dict[str, Any] | None) -> str:
    if row is None:
        return "missing_result"
    if row.get("status") == "ok":
        return ""
    error = str(row.get("error", "")).lower()
    if "timeout" in error or "timed out" in error:
        return "timeout"
    if "trace_format" in error or "trace file" in error:
        return "trace_input"
    if "config" in error or "manifest" in error:
        return "configuration"
    return "replay_failure"


def export_row(
    case: dict[str, Any], row: dict[str, Any] | None, group_name: str
) -> dict[str, Any]:
    """Select only aggregate primitives; never serialize an arbitrary input value."""
    result: dict[str, Any] = {field: "" for field in FIELDS}
    engine = mapping(case.get("engine_args"))
    recent = mapping(case.get("recent_cache"))
    router = mapping(case.get("router_config"))
    traces = case.get("trace_files", [])
    if isinstance(traces, str):
        traces = [traces]
    if not isinstance(traces, list) or not all(isinstance(p, str) for p in traces):
        raise ValueError("trace_files must be a path or a list of paths")
    basenames = [Path(path).name for path in traces]
    source = case.get("trace_format")
    family = source if source in ("mooncake", "weka") else "other"
    if any("toolagent" in basename.lower() for basename in basenames):
        family = "toolagent"
    trace_labels = [
        safe_label(basename)
        if basename.lower().startswith(TRACE_PREFIXES)
        else "redacted"
        for basename in basenames
    ]
    mode = recent.get("mode", "adaptive") if recent else "disabled"
    status = "missing_result" if row is None else row.get("status")
    status = status if status in ("ok", "failed", "missing_result") else "unknown"
    result.update(
        group=safe_label(group_name),
        run_name=safe_label(case.get("name")),
        source_family=family,
        trace_basename=";".join(sorted(set(trace_labels))),
        trace_path_count=len(traces),
        status=status,
        error_category=error_category(row),
        mode=mode
        if mode in ("adaptive", "fixed", "observe", "disabled")
        else "unknown",
        predict=number(recent.get("predict", True)) if recent else "",
        smg=int("smg" in str(case.get("name", "")).lower()),
    )
    parameters = {
        "expected_requests": case.get("expected_requests"),
        "workers": case.get("num_workers"),
        "tp_size": engine.get("aic_tp_size"),
        "kv_blocks_per_worker": engine.get("num_gpu_blocks"),
        "engine_block_size": engine.get("block_size"),
        "trace_block_size": case.get("trace_block_size"),
        "arrival_speedup": case.get("arrival_speedup_ratio", 1),
        "engine_speedup": engine.get("speedup_ratio", 1),
        "ttl_seconds": recent.get("ttl_secs"),
        "threshold": recent.get("threshold"),
        "base_overlap_credit": router.get("overlap_score_credit", 1),
        "boost_credit": recent.get("boost_credit"),
    }
    result.update({key: number(value) for key, value in parameters.items()})
    if row is None:
        return result
    if status == "ok":
        total, completed = row.get("num_requests"), row.get("completed_requests")
        expected = case.get("expected_requests", total)
        if (
            any(
                type(value) is not int or value <= 0
                for value in (expected, total, completed)
            )
            or total != expected
            or completed != expected
        ):
            result.update(status="invalid", error_category="completion_count")
        trajectory_keys = (
            "total_trajectories",
            "completed_trajectories",
            "incomplete_trajectories",
        )
        if result["status"] == "ok" and any(key in row for key in trajectory_keys):
            counts = tuple(row.get(key) for key in trajectory_keys)
            if (
                any(type(value) is not int or value < 0 for value in counts)
                or counts[0] != counts[1]
                or counts[2] != 0
            ):
                result.update(status="invalid", error_category="trajectory_count")
    result.update({key: number(row.get(key)) for key in ROW_METRICS})
    hot = mapping(row.get("recent_cache"))
    result.update({f"hot_{key}": number(hot.get(key)) for key in HOT_METRICS})
    resident = mapping(hot.get("resident_cache"))
    final = mapping(resident.get("final"))
    counters = mapping(final.get("counters"))
    result.update(
        {
            f"resident_final_{key}": number(final.get(key))
            for key in RESIDENT_FINAL_METRICS
        }
    )
    result["resident_peak_copies"] = number(resident.get("peak_resident_copies"))
    result["resident_peak_excess_copies"] = number(resident.get("peak_excess_copies"))
    result.update(
        {f"resident_{key}": number(counters.get(key)) for key in RESIDENT_COUNTERS}
    )
    return result


def collect(run_directories: list[Path]) -> list[dict[str, Any]]:
    manifests: set[Path] = set()
    for directory in run_directories:
        if not directory.is_dir():
            raise ValueError("each --runs input must be an existing directory")
        manifests.update(path.resolve() for path in directory.rglob("manifest.json"))
    rows = []
    for manifest in sorted(manifests):
        document = json.loads(manifest.read_text())
        cases = document.get("cases", []) if isinstance(document, dict) else document
        if not isinstance(cases, list):
            raise ValueError("manifest cases must be a list")
        for case in cases:
            if not isinstance(case, dict):
                raise ValueError("each manifest case must be an object")
            name = case.get("name")
            if not isinstance(name, str) or not LABEL_PATTERN.fullmatch(name):
                raise ValueError("each case name must be a simple directory name")
            row_path = manifest.parent / name / "row.json"
            row = json.loads(row_path.read_text()) if row_path.is_file() else None
            if row is not None and not isinstance(row, dict):
                raise ValueError("row.json must contain an object")
            rows.append(export_row(case, row, manifest.parent.name))
    if not rows:
        raise ValueError("no runner manifest cases found")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", action="extend", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        parser.error("output already exists; pass --overwrite to replace it")
    rows = collect(args.runs)
    with args.output.open("w" if args.overwrite else "x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} aggregate rows.")


if __name__ == "__main__":
    main()
