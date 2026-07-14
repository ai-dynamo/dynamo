# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate all nine AIPerf artifacts from the custom-encoder image sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Collection

RUNTIMES = ("vllm-serve", "dynamo-native", "dynamo-custom-encoder")
RATES = (16, 24, 32)


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
    expected_runtimes: Collection[str] = RUNTIMES,
    expected_rates: Collection[int] = RATES,
) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    failures: list[str] = []
    runtime = path.parents[1].name
    loadgen = data.get("input_config", {}).get("loadgen", {})
    rate = int(loadgen.get("request_rate", -1))

    if runtime not in expected_runtimes:
        failures.append("runtime")
    if rate not in expected_rates:
        failures.append("request_rate")
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

    for name in ("input_sequence_length", "output_sequence_length"):
        expected = float(
            expected_isl if name == "input_sequence_length" else expected_osl
        )
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
        "rate": rate,
        "accepted": not failures,
        "failures": failures,
    }


def validate_matrix(root: Path) -> list[dict[str, Any]]:
    results = [
        validate_result(path)
        for path in sorted(root.rglob("profile_export_aiperf.json"))
        if path.parents[1].name in RUNTIMES
    ]
    expected = {(runtime, rate) for runtime in RUNTIMES for rate in RATES}
    observed = {(result["runtime"], result["rate"]) for result in results}
    if observed != expected or len(results) != len(expected):
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise AssertionError(
            f"expected nine unique cells; found {len(results)}, "
            f"missing={missing}, extra={extra}"
        )
    rejected = [result for result in results if not result["accepted"]]
    if rejected:
        details = "; ".join(
            f"{item['runtime']}@{item['rate']}={','.join(item['failures'])}"
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
