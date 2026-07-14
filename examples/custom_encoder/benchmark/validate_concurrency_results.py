# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the three measured Qwen2.5 custom-encoder concurrency cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Collection

CONCURRENCIES = (8, 16, 32)
RUNTIME = "dynamo-custom-encoder"


def _metric(data: dict[str, Any], name: str, statistic: str = "avg") -> float | None:
    value = data.get(name)
    if not isinstance(value, dict) or statistic not in value:
        return None
    return float(value[statistic])


def validate_result(
    path: Path,
    expected_requests: int = 1000,
    expected_isl: int = 515,
    expected_osl: int = 70,
    expected_concurrencies: Collection[int] = CONCURRENCIES,
) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    failures: list[str] = []
    runtime = path.parents[1].name
    loadgen = data.get("input_config", {}).get("loadgen", {})
    concurrency = int(loadgen.get("concurrency", -1))

    if runtime != RUNTIME:
        failures.append("runtime")
    if concurrency not in expected_concurrencies:
        failures.append("concurrency")
    if _metric(data, "request_count") != float(expected_requests):
        failures.append("request_count")
    if data.get("error_summary"):
        failures.append("errors")
    if data.get("was_cancelled"):
        failures.append("cancelled")
    if not data.get("input_config", {}).get("endpoint", {}).get("streaming", False):
        failures.append("streaming")
    cli_command = data.get("input_config", {}).get("cli_command", "")
    if "--random-seed 42" not in cli_command:
        failures.append("random_seed")
    if f"--concurrency {concurrency}" not in cli_command:
        failures.append("concurrency_cli")

    for name, expected in (
        ("input_sequence_length", float(expected_isl)),
        ("output_sequence_length", float(expected_osl)),
    ):
        for statistic in ("min", "avg", "max"):
            if _metric(data, name, statistic) != expected:
                failures.append(f"{name}_{statistic}")

    for metric_name in ("time_to_first_token", "request_latency"):
        if data.get(metric_name, {}).get("unit") != "ms":
            failures.append(f"{metric_name}_unit")
        for statistic in ("avg", "p50", "p90", "p99"):
            if _metric(data, metric_name, statistic) is None:
                failures.append(f"{metric_name}_{statistic}")
    if _metric(data, "request_throughput") is None:
        failures.append("request_throughput")

    return {
        "path": str(path),
        "runtime": runtime,
        "concurrency": concurrency,
        "accepted": not failures,
        "failures": failures,
    }


def validate_matrix(
    root: Path, expected_concurrencies: Collection[int] = CONCURRENCIES
) -> list[dict[str, Any]]:
    expected_values = set(expected_concurrencies)
    results = [
        validate_result(path, expected_concurrencies=expected_values)
        for path in sorted(root.rglob("profile_export_aiperf.json"))
        if path.parents[1].name == RUNTIME
    ]
    observed = {result["concurrency"] for result in results}
    if observed != expected_values or len(results) != len(expected_values):
        raise AssertionError(
            f"expected {len(expected_values)} unique concurrency cells; "
            f"found {len(results)}, missing={sorted(expected_values - observed)}, "
            f"extra={sorted(observed - expected_values)}"
        )
    rejected = [result for result in results if not result["accepted"]]
    if rejected:
        details = "; ".join(
            f"concurrency{item['concurrency']}={','.join(item['failures'])}"
            for item in rejected
        )
        raise AssertionError(f"rejected benchmark artifacts: {details}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    results = validate_matrix(root)
    output = root / "validation.json"
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"BENCHMARK_AUDIT=PASS cells={len(results)} validation={output}")


if __name__ == "__main__":
    main()
