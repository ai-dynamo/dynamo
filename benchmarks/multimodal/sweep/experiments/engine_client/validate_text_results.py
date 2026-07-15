# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit the complete Qwen2.5 text engine-client result matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.multimodal.sweep.experiments.engine_client.generate_text_workload import (
    sha256,
)
from benchmarks.multimodal.sweep.experiments.engine_client.text_config import (
    EXPECTED_RUNTIMES,
    TextSweepConfig,
)


def metric(document: dict[str, Any], name: str, statistic: str = "avg") -> float:
    value = document.get(name)
    if not isinstance(value, dict) or statistic not in value:
        raise ValueError(f"missing metric {name}.{statistic}")
    return float(value[statistic])


def profiling_concurrency(document: dict[str, Any]) -> int:
    phases = document.get("input_config", {}).get("phases", [])
    for phase in phases:
        if isinstance(phase, dict) and phase.get("name") == "profiling":
            return int(phase["concurrency"])
    loadgen = document.get("input_config", {}).get("loadgen", {})
    if "concurrency" in loadgen:
        return int(loadgen["concurrency"])
    raise ValueError("missing profiling concurrency")


def validate_result(
    path: Path,
    config: TextSweepConfig,
    expected_config_sha: str,
    expected_dataset_sha: str,
) -> dict[str, Any]:
    runtime = path.parents[1].name
    trial_dir = path.parents[2].name
    trial = int(trial_dir.removeprefix("trial-"))
    document = json.loads(path.read_text(encoding="utf-8"))
    failures: list[str] = []

    if runtime not in EXPECTED_RUNTIMES:
        failures.append("runtime")
    if trial not in range(1, config.repeats + 1):
        failures.append("trial")
    if metric(document, "request_count") != float(config.request_count):
        failures.append("request_count")
    if document.get("error_summary"):
        failures.append("errors")
    if document.get("was_cancelled"):
        failures.append("cancelled")
    if not document.get("input_config", {}).get("endpoint", {}).get("streaming", False):
        failures.append("streaming")
    if profiling_concurrency(document) != config.concurrency:
        failures.append("concurrency")

    for name, expected in (
        ("input_sequence_length", config.target_isl),
        ("output_sequence_length", config.osl),
    ):
        for statistic in ("min", "avg", "max"):
            if metric(document, name, statistic) != float(expected):
                failures.append(f"{name}_{statistic}")

    for name in (
        "request_throughput",
        "output_token_throughput",
        "time_to_first_token",
        "inter_token_latency",
        "request_latency",
    ):
        metric(document, name)

    command_path = path.parent / "command.txt"
    command = command_path.read_text(encoding="utf-8")
    required_command_parts = (
        f"--concurrency {config.concurrency}",
        f"--request-count {config.request_count}",
        f"--warmup-request-count {config.warmup_count}",
        "--use-server-token-count",
        "--random-seed 42",
    )
    if any(part not in command for part in required_command_parts):
        failures.append("command")
    if "--request-rate" in command:
        failures.append("request_rate")

    run_metadata = json.loads(
        (path.parents[1] / "run_metadata.json").read_text(encoding="utf-8")
    )
    if run_metadata.get("config_sha256") != expected_config_sha:
        failures.append("config_provenance")
    if run_metadata.get("dataset_sha256") != expected_dataset_sha:
        failures.append("dataset_provenance")

    return {
        "trial": trial,
        "runtime": runtime,
        "path": str(path),
        "accepted": not failures,
        "failures": failures,
    }


def validate_matrix(
    root: Path,
    config: TextSweepConfig,
    dataset_path: Path,
    manifest_path: Path,
) -> list[dict[str, Any]]:
    metadata = json.loads(
        (root / "benchmark_metadata.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_sha = sha256(dataset_path)
    if manifest.get("model") != config.model or metadata.get("model") != config.model:
        raise AssertionError("benchmark model provenance mismatch")
    if manifest.get("media_fields") != []:
        raise AssertionError("pure-text workload must not contain media fields")
    if manifest.get("dataset_sha256") != dataset_sha:
        raise AssertionError("workload manifest hash mismatch")
    if metadata.get("dataset_sha256") != dataset_sha:
        raise AssertionError("benchmark dataset hash mismatch")
    if metadata.get("config_sha256") != config.source_sha256:
        raise AssertionError("benchmark config hash mismatch")

    results = [
        validate_result(path, config, config.source_sha256, dataset_sha)
        for path in sorted(root.rglob("profile_export_aiperf.json"))
    ]
    expected = {
        (trial, runtime)
        for trial in range(1, config.repeats + 1)
        for runtime in EXPECTED_RUNTIMES
    }
    observed = {(result["trial"], result["runtime"]) for result in results}
    if observed != expected or len(results) != len(expected):
        raise AssertionError(
            f"expected {len(expected)} trials, found {len(results)}; "
            f"missing={sorted(expected - observed)} extra={sorted(observed - expected)}"
        )
    rejected = [result for result in results if not result["accepted"]]
    if rejected:
        raise AssertionError(f"rejected benchmark artifacts: {rejected}")

    output = root / "validation.json"
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("BENCHMARK_AUDIT=PASS cells=3 trials=15")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_matrix(
        root=args.root.resolve(),
        config=TextSweepConfig.load(args.config.resolve()),
        dataset_path=args.dataset.resolve(),
        manifest_path=args.manifest.resolve(),
    )


if __name__ == "__main__":
    main()
