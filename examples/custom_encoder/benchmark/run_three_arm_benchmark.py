# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the mixed-shape aggregated, encoder-only, and parallel benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
from collections import Counter
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

from examples.custom_encoder.benchmark.run_parallel_encoder_sweep import (
    COMBINED_PORT,
    COMBINED_ROLE,
    ENCODER_ONLY_PORT,
    ENCODER_ONLY_ROLE,
    QUEUE_DELAY_US,
    REQUESTS,
    WARMUP_REQUESTS,
    GpuSampler,
    ProcessResult,
    _aiperf_command,
    _combined_service,
    _command_output,
    _encoder_service,
    _package_version,
    _patch_sweep_dispatch_summary,
    _run_single,
    _write_or_check_metadata,
    run_barrier_pair,
    validate_aiperf,
)
from examples.custom_encoder.benchmark.safeguard_proxy_workload import (
    BENCHMARK_IMAGE_SIZE_COUNTS,
    DECODER_MODEL,
    ENCODER_MODEL,
    TARGET_ISL,
    TARGET_OSL,
    generate_workload,
    validate_workload,
)

AGGREGATED_ARM = "aggregated"
STANDALONE_ARM = "encoder_only"
PARALLEL_ARM = "parallel"
ARM_ORDER = (AGGREGATED_ARM, STANDALONE_ARM, PARALLEL_ARM)
DEFAULT_REPETITIONS = 5
DEFAULT_CONCURRENCY = 64
DEFAULT_MAX_BATCH_PATCHES = 32 * 36 * 36
DEFAULT_MAX_BATCH_ITEMS = 64
EXPECTED_MEASURED_PATCHES = 890_000
EXPECTED_WARMUP_PATCHES = 17_800


def _input_path(root: Path, requests: int) -> Path:
    return root / f"image_custom_{requests}_isl{TARGET_ISL}.jsonl"


def _schedule_audit(root: Path, requests: int) -> dict[str, Any]:
    manifest = json.loads((root / "workload_manifest.json").read_text(encoding="utf-8"))
    path_to_size = {
        str(record["path"]): f"{record['width']}x{record['height']}"
        for record in manifest["images"]
    }
    rows = [
        json.loads(line)
        for line in _input_path(root, requests).read_text(encoding="utf-8").splitlines()
    ]
    sizes = [path_to_size[str(row["image"])] for row in rows]
    longest_run = 0
    current_run = 0
    previous: str | None = None
    for size in sizes:
        current_run = current_run + 1 if size == previous else 1
        longest_run = max(longest_run, current_run)
        previous = size
    transitions = sum(left != right for left, right in pairwise(sizes))
    counts = Counter(sizes)
    if counts != Counter({"300x300": requests // 2, "500x500": requests // 2}):
        raise AssertionError(f"unexpected mixed-shape schedule: {counts}")
    if transitions == 0:
        raise AssertionError("mixed-shape schedule was not shuffled")
    encoded_order = "\n".join(sizes).encode("utf-8")
    return {
        "requests": len(sizes),
        "counts": dict(sorted(counts.items())),
        "transitions": transitions,
        "longest_same_size_run": longest_run,
        "size_order_sha256": hashlib.sha256(encoded_order).hexdigest(),
        "first_20_sizes": sizes[:20],
    }


def validate_three_arm_workloads(root: Path) -> dict[str, Any]:
    measured_root = root / "measured"
    warmup_root = root / "warmup"
    measured = validate_workload(
        measured_root,
        expected_unique_images=REQUESTS,
        expected_image_size_counts=BENCHMARK_IMAGE_SIZE_COUNTS,
    )
    warmup_counts = ((300, 300, 10), (500, 500, 10))
    warmup = validate_workload(
        warmup_root,
        expected_unique_images=WARMUP_REQUESTS,
        expected_image_size_counts=warmup_counts,
    )
    if measured["raw_patch_rows"] != EXPECTED_MEASURED_PATCHES:
        raise AssertionError("measured workload patch total changed")
    if warmup["raw_patch_rows"] != EXPECTED_WARMUP_PATCHES:
        raise AssertionError("warmup workload patch total changed")
    audit = {
        "measured": measured,
        "warmup": warmup,
        "measured_schedule": _schedule_audit(measured_root, REQUESTS),
        "warmup_schedule": _schedule_audit(warmup_root, WARMUP_REQUESTS),
        "sharing_policy": (
            "the same measured JSONL and order are used by every arm and by both "
            "parallel clients"
        ),
    }
    manifest_path = root / "three_arm_workload_manifest.json"
    if manifest_path.is_file():
        expected = json.loads(manifest_path.read_text(encoding="utf-8"))
        if expected != audit:
            raise AssertionError("three-arm workload audit changed")
    print(
        "WORKLOAD_AUDIT=PASS "
        f"measured_images={measured['images']} transitions="
        f"{audit['measured_schedule']['transitions']}"
    )
    return audit


def generate_three_arm_workloads(
    root: Path, concurrency: int = DEFAULT_CONCURRENCY
) -> dict[str, Any]:
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    generate_workload(
        root / "measured",
        concurrencies=(concurrency,),
        requests=REQUESTS,
        unique_images=REQUESTS,
        target_isl=TARGET_ISL,
        seed=42,
        image_size_counts=BENCHMARK_IMAGE_SIZE_COUNTS,
    )
    generate_workload(
        root / "warmup",
        concurrencies=(concurrency,),
        requests=WARMUP_REQUESTS,
        unique_images=WARMUP_REQUESTS,
        target_isl=TARGET_ISL,
        seed=900_000,
        image_size_counts=((300, 300, 10), (500, 500, 10)),
    )
    audit = validate_three_arm_workloads(root)
    (root / "three_arm_workload_manifest.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def _duration_s(result: ProcessResult) -> float:
    return (result.finished_ns - result.released_ns) / 1_000_000_000


def _run_client(
    role: str,
    port: int,
    concurrency: int,
    requests: int,
    input_path: Path,
    artifact_dir: Path,
) -> ProcessResult:
    command = _aiperf_command(
        role=role,
        port=port,
        concurrency=concurrency,
        requests=requests,
        input_file=input_path,
        artifact_dir=artifact_dir,
    )
    return _run_single(command, artifact_dir, role=role)


def _run_parallel_clients(
    concurrency: int,
    requests: int,
    input_path: Path,
    artifact_root: Path,
) -> dict[str, ProcessResult]:
    artifacts = {
        COMBINED_ROLE: artifact_root / COMBINED_ROLE,
        ENCODER_ONLY_ROLE: artifact_root / ENCODER_ONLY_ROLE,
    }
    commands = {
        COMBINED_ROLE: _aiperf_command(
            role=COMBINED_ROLE,
            port=COMBINED_PORT,
            concurrency=concurrency,
            requests=requests,
            input_file=input_path,
            artifact_dir=artifacts[COMBINED_ROLE],
        ),
        ENCODER_ONLY_ROLE: _aiperf_command(
            role=ENCODER_ONLY_ROLE,
            port=ENCODER_ONLY_PORT,
            concurrency=concurrency,
            requests=requests,
            input_file=input_path,
            artifact_dir=artifacts[ENCODER_ONLY_ROLE],
        ),
    }
    return run_barrier_pair(commands, artifacts)


def _validated_client(
    result: ProcessResult, concurrency: int, mixed_shape: bool
) -> dict[str, Any]:
    validation = validate_aiperf(
        result.artifact_dir / "profile_export_aiperf.json",
        result.role,
        concurrency,
        require_fixed_client_isl=not mixed_shape or result.role == COMBINED_ROLE,
    )
    if not validation["accepted"]:
        raise AssertionError(
            f"AIPerf validation failed for {result.role}: {validation['failures']}"
        )
    return {
        "duration_s": _duration_s(result),
        "wall_throughput_request_s": REQUESTS / _duration_s(result),
        "released_ns": result.released_ns,
        "finished_ns": result.finished_ns,
        "artifact_dir": str(result.artifact_dir),
        "aiperf": validation,
    }


def _server_dispatch(log_path: Path) -> dict[str, Any]:
    dispatch = _patch_sweep_dispatch_summary(log_path)
    expected = EXPECTED_MEASURED_PATCHES + EXPECTED_WARMUP_PATCHES
    if dispatch["actual_patches_including_warmup"] != expected:
        raise AssertionError(
            f"server dispatched {dispatch['actual_patches_including_warmup']} "
            f"patches; expected {expected}"
        )
    grids = {entry["grid"] for entry in dispatch["distribution"]}
    if grids != {"1x22x22", "1x36x36"}:
        raise AssertionError(f"unexpected dispatched grids: {sorted(grids)}")
    if dispatch["capture_memory_delta_gib"] is None:
        raise AssertionError("server log did not record CUDA graph memory")
    return dispatch


def _execute_arm(
    arm: str,
    repetition: int,
    order_index: int,
    workload_root: Path,
    output_root: Path,
    concurrency: int,
    max_batch_patches: int,
    max_batch_items: int,
) -> dict[str, Any]:
    cell_dir = output_root / arm / f"run{repetition}"
    result_path = cell_dir / "result.json"
    if result_path.is_file():
        return json.loads(result_path.read_text(encoding="utf-8"))
    if cell_dir.exists() and any(cell_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite partial benchmark cell {cell_dir}")

    measured_input = _input_path(workload_root / "measured", REQUESTS)
    warmup_input = _input_path(workload_root / "warmup", WARMUP_REQUESTS)
    clients: dict[str, ProcessResult]
    server_logs: dict[str, Path]
    if arm == AGGREGATED_ARM:
        combined_log = cell_dir / "combined_server.log"
        with _combined_service(
            cell_dir,
            arm,
            max_batch_patches=max_batch_patches,
            max_batch_items=max_batch_items,
            log_path=combined_log,
        ):
            _run_client(
                COMBINED_ROLE,
                COMBINED_PORT,
                min(concurrency, WARMUP_REQUESTS),
                WARMUP_REQUESTS,
                warmup_input,
                cell_dir / "warmup" / COMBINED_ROLE,
            )
            measured = _run_client(
                COMBINED_ROLE,
                COMBINED_PORT,
                concurrency,
                REQUESTS,
                measured_input,
                cell_dir / "measured" / COMBINED_ROLE,
            )
        clients = {COMBINED_ROLE: measured}
        server_logs = {COMBINED_ROLE: combined_log}
    elif arm == STANDALONE_ARM:
        encoder_log = cell_dir / "encoder_only_server.log"
        with _encoder_service(
            cell_dir,
            arm,
            max_batch_patches=max_batch_patches,
            max_batch_items=max_batch_items,
            log_path=encoder_log,
        ):
            _run_client(
                ENCODER_ONLY_ROLE,
                ENCODER_ONLY_PORT,
                min(concurrency, WARMUP_REQUESTS),
                WARMUP_REQUESTS,
                warmup_input,
                cell_dir / "warmup" / ENCODER_ONLY_ROLE,
            )
            measured = _run_client(
                ENCODER_ONLY_ROLE,
                ENCODER_ONLY_PORT,
                concurrency,
                REQUESTS,
                measured_input,
                cell_dir / "measured" / ENCODER_ONLY_ROLE,
            )
        clients = {ENCODER_ONLY_ROLE: measured}
        server_logs = {ENCODER_ONLY_ROLE: encoder_log}
    elif arm == PARALLEL_ARM:
        combined_log = cell_dir / "combined_server.log"
        encoder_log = cell_dir / "encoder_only_server.log"
        with (
            _combined_service(
                cell_dir,
                arm,
                max_batch_patches=max_batch_patches,
                max_batch_items=max_batch_items,
                log_path=combined_log,
            ),
            _encoder_service(
                cell_dir,
                arm,
                max_batch_patches=max_batch_patches,
                max_batch_items=max_batch_items,
                log_path=encoder_log,
            ),
        ):
            _run_parallel_clients(
                min(concurrency, WARMUP_REQUESTS),
                WARMUP_REQUESTS,
                warmup_input,
                cell_dir / "warmup",
            )
            clients = _run_parallel_clients(
                concurrency,
                REQUESTS,
                measured_input,
                cell_dir / "measured",
            )
        server_logs = {
            COMBINED_ROLE: combined_log,
            ENCODER_ONLY_ROLE: encoder_log,
        }
    else:
        raise ValueError(f"unknown benchmark arm {arm!r}")

    validated = {
        role: _validated_client(result, concurrency, mixed_shape=True)
        for role, result in clients.items()
    }
    durations = [float(client["duration_s"]) for client in validated.values()]
    releases = [int(client["released_ns"]) for client in validated.values()]
    finishes = [int(client["finished_ns"]) for client in validated.values()]
    row = {
        "arm": arm,
        "repetition": repetition,
        "order_index": order_index,
        "concurrency_per_client": concurrency,
        "requests_per_client": REQUESTS,
        "total_requests": REQUESTS * len(validated),
        "min_wall_time_s": min(durations),
        "max_wall_time_s": max(durations),
        "start_skew_ms": (max(releases) - min(releases)) / 1_000_000,
        "completion_skew_ms": (max(finishes) - min(finishes)) / 1_000_000,
        "clients": validated,
        "dispatch": {
            role: _server_dispatch(path) for role, path in server_logs.items()
        },
    }
    if arm == PARALLEL_ARM and row["start_skew_ms"] > 100:
        raise AssertionError(
            f"parallel start skew exceeded 100ms: {row['start_skew_ms']}"
        )
    result_path.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    return row


def _metadata(
    workload_audit: dict[str, Any],
    repetitions: int,
    concurrency: int,
    max_batch_patches: int,
    max_batch_items: int,
) -> dict[str, Any]:
    required_env = (
        "DYNAMO_BENCHMARK_COMMIT",
        "DYNAMO_BENCHMARK_BRANCH",
        "DYNAMO_BENCHMARK_IMAGE",
        "DYNAMO_BASE_IMAGE_COMMIT",
    )
    missing = [name for name in required_env if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"missing benchmark provenance: {', '.join(missing)}")
    return {
        "dynamo_commit": os.environ["DYNAMO_BENCHMARK_COMMIT"],
        "dynamo_branch": os.environ["DYNAMO_BENCHMARK_BRANCH"],
        "container_image": os.environ["DYNAMO_BENCHMARK_IMAGE"],
        "base_image_dynamo_commit": os.environ["DYNAMO_BASE_IMAGE_COMMIT"],
        "gpu": _command_output(
            [
                "nvidia-smi",
                "--id=0",
                "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "versions": {
            "python": platform.python_version(),
            "aiperf": _package_version("aiperf")
            or _command_output(["aiperf", "--version"]),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "vllm": _package_version("vllm"),
        },
        "models": {"decoder": DECODER_MODEL, "encoder": ENCODER_MODEL},
        "arms": list(ARM_ORDER),
        "arm_order_policy": "rotate by repetition",
        "repetitions": repetitions,
        "concurrency_per_client": concurrency,
        "requests_per_client": REQUESTS,
        "warmup_requests_per_client": WARMUP_REQUESTS,
        "target_isl": TARGET_ISL,
        "combined_target_osl": TARGET_OSL,
        "encoder_only_target_osl": 1,
        "max_batch_patches": max_batch_patches,
        "max_batch_items": max_batch_items,
        "queue_delay_us": QUEUE_DELAY_US,
        "graph_batch_buckets": [1, 2, 4, 8, 16, 32, 64],
        "graph_image_sizes": [[300, 300], [500, 500]],
        "preprocess_cache_size": 0,
        "workload": workload_audit,
    }


def _mean_stdev(values: Sequence[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def summarize_three_arm(output_root: Path) -> dict[str, Any]:
    metadata = json.loads(
        (output_root / "benchmark_metadata.json").read_text(encoding="utf-8")
    )
    repetitions = int(metadata["repetitions"])
    rows = [
        json.loads(
            (output_root / arm / f"run{repetition}" / "result.json").read_text(
                encoding="utf-8"
            )
        )
        for repetition in range(1, repetitions + 1)
        for arm in ARM_ORDER
    ]
    expected_aiperf = repetitions * 4
    actual_aiperf = sum(len(row["clients"]) for row in rows)
    if actual_aiperf != expected_aiperf:
        raise AssertionError(
            f"expected {expected_aiperf} measured AIPerf results, got {actual_aiperf}"
        )

    summaries: list[dict[str, Any]] = []
    for arm in ARM_ORDER:
        selected = [row for row in rows if row["arm"] == arm]
        minima = [float(row["min_wall_time_s"]) for row in selected]
        maxima = [float(row["max_wall_time_s"]) for row in selected]
        avg_min, std_min = _mean_stdev(minima)
        avg_max, std_max = _mean_stdev(maxima)
        role_names = sorted({role for row in selected for role in row["clients"]})
        aiperf_means = {
            role: statistics.mean(
                float(row["clients"][role]["aiperf"]["request_throughput"])
                for row in selected
                if role in row["clients"]
            )
            for role in role_names
        }
        summaries.append(
            {
                "arm": arm,
                "runs": len(selected),
                "requests_per_client": REQUESTS,
                "avg_min_wall_time_s": avg_min,
                "avg_max_wall_time_s": avg_max,
                "stdev_min_wall_time_s": std_min,
                "stdev_max_wall_time_s": std_max,
                "wall_throughput_lower_request_s": REQUESTS / avg_max,
                "wall_throughput_upper_request_s": REQUESTS / avg_min,
                "aiperf_request_s_mean_by_client": aiperf_means,
                "client_avg_wall_time_s": {
                    role: statistics.mean(
                        float(row["clients"][role]["duration_s"])
                        for row in selected
                        if role in row["clients"]
                    )
                    for role in role_names
                },
            }
        )

    report = {"summaries": summaries, "runs": rows}
    (output_root / "three_arm_results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (output_root / "benchmark.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "arm",
                "repetition",
                "order_index",
                "total_requests",
                "min_wall_time_s",
                "max_wall_time_s",
                "start_skew_ms",
                "completion_skew_ms",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in writer.fieldnames})

    lines = [
        "# Three-arm mixed-image custom-encoder benchmark",
        "",
        (
            f"One H100, concurrency {metadata['concurrency_per_client']} per client, "
            f"{REQUESTS:,} requests/client, {repetitions} runs. Every client uses "
            "the same shuffled set of 500 unique 300x300 and 500 unique 500x500 "
            "images."
        ),
        "",
        "## Average wall-time throughput",
        "",
        "| Experiment | Avg min wall | Avg max wall | Wall throughput | Runs |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        lower = float(summary["wall_throughput_lower_request_s"])
        upper = float(summary["wall_throughput_upper_request_s"])
        throughput = (
            f"{lower:.2f} req/s" if lower == upper else f"{lower:.2f}-{upper:.2f} req/s"
        )
        lines.append(
            f"| {summary['arm']} | {summary['avg_min_wall_time_s']:.3f} s | "
            f"{summary['avg_max_wall_time_s']:.3f} s | {throughput} | "
            f"{summary['runs']} |"
        )
    lines.extend(
        [
            "",
            (
                "Parallel throughput is per-client completion throughput: "
                "`1000 / avg_max_wall_time` through `1000 / avg_min_wall_time`; "
                "it does not use 2,000 as the numerator."
            ),
            "",
            "## Raw wall times",
            "",
            "| Repetition | Order | Experiment | Min wall | Max wall |",
            "| ---: | ---: | --- | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['repetition']} | {row['order_index']} | {row['arm']} | "
            f"{row['min_wall_time_s']:.3f} s | {row['max_wall_time_s']:.3f} s |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- [Full JSON](three_arm_results.json)",
            "- [CSV](benchmark.csv)",
            "- [Metadata](benchmark_metadata.json)",
            "- [GPU samples](gpu_samples.csv)",
        ]
    )
    report_path = output_root / "benchmark.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"BENCHMARK_AUDIT=PASS aiperf_results={actual_aiperf}")
    print(f"benchmark={report_path} csv={output_root / 'benchmark.csv'}")
    return report


def run_three_arm_benchmark(
    workload_root: Path,
    output_root: Path,
    repetitions: int = DEFAULT_REPETITIONS,
    concurrency: int = DEFAULT_CONCURRENCY,
    max_batch_patches: int = DEFAULT_MAX_BATCH_PATCHES,
    max_batch_items: int = DEFAULT_MAX_BATCH_ITEMS,
) -> dict[str, Any]:
    if min(repetitions, concurrency, max_batch_patches, max_batch_items) < 1:
        raise ValueError("benchmark limits must be positive")
    workload_audit = validate_three_arm_workloads(workload_root)
    metadata = _metadata(
        workload_audit,
        repetitions,
        concurrency,
        max_batch_patches,
        max_batch_items,
    )
    _write_or_check_metadata(output_root, metadata)

    with GpuSampler(output_root / "gpu_samples.csv"):
        for repetition in range(1, repetitions + 1):
            offset = (repetition - 1) % len(ARM_ORDER)
            order = ARM_ORDER[offset:] + ARM_ORDER[:offset]
            for order_index, arm in enumerate(order, start=1):
                _execute_arm(
                    arm,
                    repetition,
                    order_index,
                    workload_root,
                    output_root,
                    concurrency,
                    max_batch_patches,
                    max_batch_items,
                )
    return summarize_three_arm(output_root)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--workload-dir", type=Path, required=True)
    generate.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    validate = subparsers.add_parser("validate-workload")
    validate.add_argument("--workload-dir", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--workload-dir", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    run.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    run.add_argument("--max-batch-patches", type=int, default=DEFAULT_MAX_BATCH_PATCHES)
    run.add_argument("--max-batch-items", type=int, default=DEFAULT_MAX_BATCH_ITEMS)
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("output_dir", type=Path)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "generate":
        generate_three_arm_workloads(
            args.workload_dir.resolve(), concurrency=args.concurrency
        )
    elif args.command == "validate-workload":
        validate_three_arm_workloads(args.workload_dir.resolve())
    elif args.command == "run":
        run_three_arm_benchmark(
            args.workload_dir.resolve(),
            args.output_dir.resolve(),
            repetitions=args.repetitions,
            concurrency=args.concurrency,
            max_batch_patches=args.max_batch_patches,
            max_batch_items=args.max_batch_items,
        )
    else:
        summarize_three_arm(args.output_dir.resolve())


if __name__ == "__main__":
    main()
