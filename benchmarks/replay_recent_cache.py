# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Run independent recent-cache replay cases in isolated, bounded subprocesses.

Use the selected worktree's Python environment, for example::

    .venv/bin/python benchmarks/replay_recent_cache.py \
        --manifest /tmp/recent-cache/cases.json \
        --output /tmp/recent-cache/results --parallel 4

The manifest is a JSON list, or an object with a ``cases`` list. Each case has a
unique ``name`` and ``trace_files`` (one path or a list), and may include
``engine_args``, ``router_config``, ``recent_cache``, ``expected_requests``, and
the replay options listed in REPLAY_OPTIONS. Relative trace paths resolve from
the manifest's directory. No trace timestamps or gaps are rewritten here.
Weka takes exactly one path, which may be a file or a trace directory.
``recent_cache: null`` unsets the experimental environment variable for a case.
Existing output directories are rejected so failed evidence cannot be replaced.

``request_windows.json`` groups arrivals, first admissions, readmissions, and
completions into ten equal simulated-time windows. These are request events,
not executed prefill/recompute measurements. For exact native vLLM preemption
log counts, unset ``DYNAMO_SKIP_PYTHON_LOG_INIT`` and set before launching::

    DYN_LOG=warn,aisimulate_core::engine::scheduler::vllm::core=debug
    DYN_LOGGING_CONSOLE_FORMAT=jsonl

Logged preemption timestamps are host wall time, not simulated time.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import resource
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPLAY_OPTIONS = (
    "trace_format",
    "trace_block_size",
    "num_workers",
    "agentic_lanes",
    "arrival_speedup_ratio",
    "replay_concurrency",
    "max_sim_time_ms",
)
SUMMARY_METRICS = (
    "duration_ms",
    "first_admission_prefix_cache_reused_ratio",
    "mean_ttft_ms",
    "p90_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "mean_e2e_latency_ms",
    "p90_e2e_latency_ms",
    "p95_e2e_latency_ms",
    "p99_e2e_latency_ms",
    "request_throughput_rps",
    "input_throughput_tok_s",
    "output_throughput_tok_s",
)
OPTIONAL_SUMMARY_METRICS = (
    "gpu_hours",
    "prefill_worker_seconds",
    "decode_worker_seconds",
    "prefill_gpus_per_worker",
    "decode_gpus_per_worker",
    "total_trajectories",
    "completed_trajectories",
    "incomplete_trajectories",
    "mean_trajectory_e2e_latency_ms",
    "p90_trajectory_e2e_latency_ms",
    "p95_trajectory_e2e_latency_ms",
    "p99_trajectory_e2e_latency_ms",
)
DIAGNOSTIC_PREFIX = "recent_cache_experiment="
PREEMPTION_TARGET = "aisimulate_core::engine::scheduler::vllm::core"
PREEMPTION_MESSAGE = "vLLM scheduler preempted and requeued request"
LOGGING_ENVIRONMENT = (
    "DYN_LOG",
    "DYN_LOGGING_CONSOLE_FORMAT",
    "DYNAMO_SKIP_PYTHON_LOG_INIT",
)


def write_json(path: Path, value: Any) -> None:
    with path.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def run_child(case_path: Path) -> None:
    case = json.loads(case_path.read_text())
    config = case.get("recent_cache")
    if config is None:
        os.environ.pop("DYN_REPLAY_RECENT_CACHE", None)
    else:
        os.environ["DYN_REPLAY_RECENT_CACHE"] = json.dumps(config, allow_nan=False)

    started = time.perf_counter()
    cpu_started = time.process_time()
    try:
        # Load the native extension only in the child, after setting its config.
        from dynamo.llm import KvRouterConfig
        from dynamo.replay.api import run_trace_replay
        from dynamo.replay.config import load_engine_args

        router = case.get("router_config")
        report = run_trace_replay(
            case["trace_files"],
            replay_mode="offline",
            router_mode="kv_router",
            capture_per_request=True,
            capture_planner_details=False,
            extra_engine_args=load_engine_args(case.get("engine_args")),
            router_config=(
                None if router is None else KvRouterConfig.from_json(json.dumps(router))
            ),
            **{key: case[key] for key in REPLAY_OPTIONS if key in case},
        )
        write_json(case_path.parent / "report.json", report.to_dict())
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        write_json(
            case_path.parent / "host.json",
            {
                "wall_seconds": time.perf_counter() - started,
                "process_cpu_seconds": time.process_time() - cpu_started,
                "user_cpu_seconds": usage.ru_utime,
                "system_cpu_seconds": usage.ru_stime,
                "max_rss_bytes": usage.ru_maxrss
                * (1 if sys.platform == "darwin" else 1024),
                "python": sys.executable,
                "pid": os.getpid(),
                "logging_environment": {
                    key: os.environ.get(key) for key in LOGGING_ENVIRONMENT
                },
            },
        )


def pressure_metrics(diagnostic: dict[str, Any]) -> dict[str, Any]:
    decisions = diagnostic["decisions"]
    metrics = {
        key: diagnostic[key]
        for key in (
            "decisions",
            "boosted_decisions",
            "mode_transitions",
            "mean_decision_pressure",
            "peak_footprint",
            "peak_per_worker_pressure",
            "final_count",
            "final_capacity",
            "decisions_above_threshold",
            "decisions_at_or_below_threshold",
            "pressure_upward_crossings",
            "pressure_downward_crossings",
        )
        if key in diagnostic
    }
    if decisions > 0:
        metrics["boosted_decision_fraction"] = (
            diagnostic["boosted_decisions"] / decisions
        )
        if "decisions_above_threshold" in diagnostic:
            metrics["decision_fraction_above_threshold"] = (
                diagnostic["decisions_above_threshold"] / decisions
            )
    series = diagnostic["series"]
    threshold = diagnostic["config"]["threshold"]
    metrics["pressure_comparison"] = ">"
    metrics["threshold"] = threshold
    if "resident_cache" in diagnostic:
        # Preserve scope/limitations: these are event-visible rank copies,
        # not physical pages or a complete inventory of engine KV memory.
        metrics["resident_cache"] = {
            key: value
            for key, value in diagnostic["resident_cache"].items()
            if key != "series"
        }
    if series:
        ratios = [point["ratio"] for point in series]
        metrics["sampled_min_pressure"] = min(ratios)
        metrics["sampled_max_pressure"] = max(ratios)
        metrics["sampled_fraction_above_threshold"] = sum(
            ratio > threshold for ratio in ratios
        ) / len(ratios)
        # This is a held-sample estimate, not an integral of expiry events.
        span = series[-1]["time_secs"] - series[0]["time_secs"]
        metrics["sampled_span_seconds"] = span
        if span > 0:
            above_seconds = sum(
                right["time_secs"] - left["time_secs"]
                for left, right in zip(series, series[1:])
                if left["ratio"] > threshold
            )
            metrics["approx_sample_hold_time_fraction_above_threshold"] = (
                above_seconds / span
            )
    return metrics


def request_metrics(
    records: list[dict[str, Any]], case: dict[str, Any]
) -> dict[str, Any]:
    routed = 0
    compared = 0
    overpredicted = 0
    overpredicted_tokens = 0
    first_admission_reused_tokens = 0
    first_admission_input_tokens = 0
    first_admission_records = 0
    compared_input_tokens = 0
    overpredicted_more_than_block = 0
    readmission_events = 0
    requests_readmitted = 0
    max_readmissions_per_request = 0
    engine_args = case.get("engine_args") or {}
    if isinstance(engine_args, str):
        engine_args = json.loads(engine_args)
    block_size = engine_args.get("block_size")
    routes_per_worker: dict[str, int] = {}
    for record in records:
        readmissions = sum(
            admission["pool"] == "agg" and admission["is_readmission"]
            for admission in record["admission_history"]
        )
        readmission_events += readmissions
        requests_readmitted += readmissions > 0
        max_readmissions_per_request = max(max_readmissions_per_request, readmissions)
        routes = [
            route for route in record["routing_history"] if route["pool"] == "agg"
        ]
        routed += len(routes)
        admissions = [
            admission
            for admission in record["admission_history"]
            if admission["pool"] == "agg" and admission["pool_admission_ordinal"] == 0
        ]
        if not admissions:
            continue
        actual = admissions[0]["reused_input_tokens"]
        first_admission_records += 1
        first_admission_reused_tokens += actual
        first_admission_input_tokens += record["input_length"]
        if not routes:
            continue
        route = routes[0]
        worker = f"{route['logical_worker_id']}:{route['dp_rank']}"
        routes_per_worker[worker] = routes_per_worker.get(worker, 0) + 1
        predicted = route["reported_overlap_tokens"]
        if predicted is None:
            continue
        compared += 1
        compared_input_tokens += record["input_length"]
        excess = max(0, min(predicted, record["input_length"]) - actual)
        overpredicted += excess > 0
        overpredicted_tokens += excess
        if block_size is not None:
            overpredicted_more_than_block += excess > block_size
    return {
        "routing_records": routed,
        "readmission_events": readmission_events,
        "requests_readmitted": requests_readmitted,
        "requests_readmitted_fraction": (
            requests_readmitted / len(records) if records else None
        ),
        "max_readmissions_per_request": max_readmissions_per_request,
        "first_admission_records": first_admission_records,
        "first_admission_reused_tokens": first_admission_reused_tokens,
        "first_admission_input_tokens": first_admission_input_tokens,
        "first_route_counts_by_worker_dp": routes_per_worker,
        "route_actual_comparable_requests": compared,
        "route_overpredicted_requests": overpredicted,
        "route_overpredicted_tokens": overpredicted_tokens,
        "route_overpredicted_fraction": overpredicted / compared if compared else None,
        "route_overpredicted_input_fraction": (
            overpredicted_tokens / compared_input_tokens
            if compared_input_tokens
            else None
        ),
        # One-block discrepancies may be the engine's final logits recomputation.
        "route_overpredicted_more_than_one_block_requests": (
            overpredicted_more_than_block if block_size is not None else None
        ),
        "route_overprediction_block_size": block_size,
    }


def request_windows(report: dict[str, Any]) -> dict[str, Any]:
    duration_ms = max(
        report["summary"]["duration_ms"],
        max(record["terminal_time_ms"] for record in report["per_request"]),
    )
    count = 10 if duration_ms > 0 else 1
    width = duration_ms / count
    windows = [
        {
            "start_ms": index * width,
            "end_ms": (index + 1) * width,
            "arrivals": 0,
            "first_admissions": 0,
            "first_admission_input_tokens": 0,
            "first_admission_reused_input_tokens": 0,
            "readmission_events": 0,
            "completed_requests": 0,
            "completed_output_tokens": 0,
        }
        for index in range(count)
    ]
    latencies: list[list[float]] = [[] for _ in windows]

    def index_for(at_ms: float) -> int:
        if not math.isfinite(at_ms) or at_ms < 0 or at_ms > duration_ms + 1e-6:
            raise ValueError(f"request event outside replay duration: {at_ms}")
        return min(count - 1, int(at_ms / width)) if width > 0 else 0

    for record in report["per_request"]:
        windows[index_for(record["arrival_time_ms"])]["arrivals"] += 1
        for admission in record["admission_history"]:
            if admission["pool"] != "agg":
                continue
            window = windows[index_for(admission["at_ms"])]
            if admission["is_readmission"]:
                window["readmission_events"] += 1
                continue
            window["first_admissions"] += 1
            window["first_admission_input_tokens"] += record["input_length"]
            window["first_admission_reused_input_tokens"] += admission[
                "reused_input_tokens"
            ]
        if record["terminal_status"] != "completed":
            continue
        index = index_for(record["terminal_time_ms"])
        windows[index]["completed_requests"] += 1
        windows[index]["completed_output_tokens"] += record["output_length"]
        if record["e2e_latency_ms"] is not None:
            latencies[index].append(record["e2e_latency_ms"])
    for window, values in zip(windows, latencies):
        tokens = window["first_admission_input_tokens"]
        window["first_admission_reused_ratio"] = (
            window["first_admission_reused_input_tokens"] / tokens if tokens else None
        )
        values.sort()
        window["p90_completed_request_e2e_ms"] = (
            values[int((len(values) - 1) * 0.9 + 0.5)] if values else None
        )
    return {
        "clock": "simulated",
        "duration_ms": duration_ms,
        "intervals": "left-closed right-open; final window includes endpoint",
        "output_token_attribution": "all request output credited at completion",
        "windows": windows,
    }


def preemption_metrics(
    events: list[dict[str, Any]], host: dict[str, Any]
) -> dict[str, Any]:
    environment = host.get("logging_environment", {})
    filter_levels = {}
    for item in (environment.get("DYN_LOG") or "").split(","):
        target, separator, level = item.partition("=")
        if separator:
            filter_levels[target.strip()] = level.strip()
    filter_enabled = filter_levels.get(PREEMPTION_TARGET) in ("debug", "trace")
    complete_capture = (
        filter_enabled
        and environment.get("DYN_LOGGING_CONSOLE_FORMAT") == "jsonl"
        and environment.get("DYNAMO_SKIP_PYTHON_LOG_INIT") is None
    )
    counts: dict[str, int] = {}
    for event in events:
        request_id = event["request_id"]
        counts[request_id] = counts.get(request_id, 0) + 1
    return {
        "logged_vllm_preemption_events": len(events),
        "vllm_preemption_log_capture_configured": complete_capture,
        "vllm_preemptions_total": len(events) if complete_capture else None,
        "logged_vllm_preempted_requests": len(counts),
        "max_logged_vllm_preemptions_per_request": max(counts.values(), default=0),
    }


def validate_report(report: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    summary = report["summary"]
    expected = case.get("expected_requests", summary["num_requests"])
    total = summary["num_requests"]
    completed = summary["completed_requests"]
    if (
        any(
            type(value) is not int or value <= 0
            for value in (expected, total, completed)
        )
        or total != expected
        or completed != expected
    ):
        raise ValueError(
            f"request counts disagree: expected={expected}, "
            f"num_requests={total}, completed_requests={completed}"
        )
    records = report["per_request"]
    if not isinstance(records, list) or len(records) != summary["num_requests"]:
        raise ValueError("per-request records do not cover all replay requests")
    terminal_completions = sum(
        record["terminal_status"] == "completed" for record in records
    )
    if terminal_completions != completed:
        raise ValueError("per-request completions disagree with summary")
    trajectory_keys = (
        "total_trajectories",
        "completed_trajectories",
        "incomplete_trajectories",
    )
    if any(key in summary for key in trajectory_keys):
        counts = tuple(summary.get(key) for key in trajectory_keys)
        if (
            any(type(value) is not int or value < 0 for value in counts)
            or counts[0] != counts[1]
            or counts[2] != 0
        ):
            raise ValueError("trajectory counts must describe complete trajectories")
    metric_keys = SUMMARY_METRICS + tuple(
        key for key in OPTIONAL_SUMMARY_METRICS if key in summary
    )
    for key in metric_keys:
        value = summary[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"summary metric {key} is not numeric: {value!r}")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"summary metric {key} is invalid: {value!r}")
    metrics = {
        "num_requests": summary["num_requests"],
        "completed_requests": completed,
        **{key: summary[key] for key in metric_keys},
        **request_metrics(records, case),
    }
    if summary.get("gpu_hours", 0) > 0 and summary["duration_ms"] > 0:
        gpu_seconds = summary["gpu_hours"] * 3600
        average_gpus = gpu_seconds / (summary["duration_ms"] / 1000)
        metrics["average_provisioned_gpus"] = average_gpus
        metrics["output_throughput_tok_s_per_gpu"] = (
            summary["output_throughput_tok_s"] / average_gpus
        )
        metrics["request_throughput_rps_per_gpu"] = (
            summary["request_throughput_rps"] / average_gpus
        )
    return metrics


def run_case(case: dict[str, Any], output: Path, timeout: float) -> dict[str, Any]:
    directory = output / case["name"]
    directory.mkdir()
    case_path = directory / "case.json"
    write_json(case_path, case)
    row: dict[str, Any] = {
        "name": case["name"],
        "status": "failed",
        "directory": str(directory),
    }
    started = time.perf_counter()
    try:
        with (
            (directory / "stdout.log").open("w") as stdout,
            (directory / "stderr.log").open("w") as stderr,
        ):
            process = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--case",
                    str(case_path),
                ],
                stdout=stdout,
                stderr=stderr,
                timeout=timeout,
                check=False,
            )
        row["returncode"] = process.returncode
        diagnostics = []
        preemptions = []
        for line in (directory / "stderr.log").read_text().splitlines():
            if DIAGNOSTIC_PREFIX in line:
                diagnostics.append(json.loads(line.split(DIAGNOSTIC_PREFIX, 1)[1]))
            elif line.startswith("{") and PREEMPTION_MESSAGE in line:
                event = json.loads(line)
                if (
                    event.get("target") == PREEMPTION_TARGET
                    and event.get("message") == PREEMPTION_MESSAGE
                    and event.get("level") == "DEBUG"
                ):
                    preemptions.append(event)
        if diagnostics:
            write_json(directory / "recent_cache.json", diagnostics)
        if process.returncode != 0:
            raise ValueError(f"replay exited {process.returncode}; inspect stderr.log")
        if case.get("recent_cache") is not None and len(diagnostics) != 1:
            raise ValueError(
                f"expected one recent-cache diagnostic, got {len(diagnostics)}"
            )
        report = json.loads((directory / "report.json").read_text())
        row.update(validate_report(report, case))
        row["host"] = json.loads((directory / "host.json").read_text())
        row.update(preemption_metrics(preemptions, row["host"]))
        windows = request_windows(report)
        row["readmission_events_by_time_decile"] = [
            window["readmission_events"] for window in windows["windows"]
        ]
        write_json(directory / "request_windows.json", windows)
        write_json(directory / "preemption_events.json", preemptions)
        if diagnostics:
            row["recent_cache"] = pressure_metrics(diagnostics[0])
        row["status"] = "ok"
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.TimeoutExpired,
    ) as error:
        # Failure is evidence: persist it and make the campaign exit nonzero.
        row["error"] = f"{type(error).__name__}: {error}"
    row["subprocess_wall_seconds"] = time.perf_counter() - started
    write_json(directory / "row.json", row)
    return row


def load_cases(manifest: Path) -> list[dict[str, Any]]:
    payload = json.loads(manifest.read_text())
    cases = payload["cases"] if isinstance(payload, dict) else payload
    if not isinstance(cases, list) or not cases:
        raise ValueError("manifest must contain a nonempty cases list")
    names: set[str] = set()
    for case in cases:
        name = case["name"]
        if not isinstance(name, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*", name
        ):
            raise ValueError(f"invalid case name: {name!r}")
        if name in names:
            raise ValueError(f"duplicate case name: {name}")
        names.add(name)
        paths = case["trace_files"]
        paths = [paths] if isinstance(paths, str) else paths
        if not isinstance(paths, list) or not paths:
            raise ValueError(f"{name}: trace_files must be a path or nonempty list")
        is_weka = case.get("trace_format") == "weka"
        if is_weka and len(paths) != 1:
            raise ValueError(f"{name}: Weka requires exactly one file or directory")
        case["trace_files"] = [
            str((manifest.parent / path).resolve()) for path in paths
        ]
        for path in case["trace_files"]:
            trace_path = Path(path)
            if trace_path.is_file() or (is_weka and trace_path.is_dir()):
                continue
            raise ValueError(f"{name}: trace path missing or unsupported: {path}")
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=900)
    parser.add_argument("--case", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.case is not None:
        run_child(args.case.resolve())
        return 0
    if args.manifest is None or args.output is None:
        parser.error("--manifest and --output are required")
    if (
        args.parallel < 1
        or not math.isfinite(args.timeout_seconds)
        or args.timeout_seconds <= 0
    ):
        parser.error("--parallel and --timeout-seconds must be positive")
    cases = load_cases(args.manifest.resolve())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", {"cases": cases})
    rows = []
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = [
            executor.submit(run_case, case, output, args.timeout_seconds)
            for case in cases
        ]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            rows.sort(key=lambda item: item["name"])
            write_json(output / "rows.json", rows)
            if row["status"] != "ok":
                print(f"FAIL {row['name']}: {row['error']}", flush=True)
                continue
            pressure = row.get("recent_cache", {})
            print(
                f"OK {row['name']} requests={row['completed_requests']} "
                f"output_tok_s={row['output_throughput_tok_s']:.2f} "
                f"rps={row['request_throughput_rps']:.2f} "
                f"p90_e2e_ms={row['p90_e2e_latency_ms']:.2f} "
                f"reuse={row['first_admission_prefix_cache_reused_ratio']:.3f} "
                f"above={pressure.get('decision_fraction_above_threshold', 'n/a')} "
                f"up/down={pressure.get('pressure_upward_crossings', 'n/a')}/"
                f"{pressure.get('pressure_downward_crossings', 'n/a')}",
                flush=True,
            )
    return int(any(row["status"] != "ok" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
