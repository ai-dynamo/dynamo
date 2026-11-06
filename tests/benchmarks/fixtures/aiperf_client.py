# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fault-injected AIPerf executable for benchmark wrapper tests."""

import json
import os
import sys
from pathlib import Path


def write_artifacts(directory: Path, count: int, mode: str) -> None:
    """Write healthy or fault-injected aggregate exports for one benchmark."""
    if mode == "missing":
        return
    summary = {"request_count": {"avg": count - 1 if mode == "count" else count}}
    if mode == "mixed-errors":
        summary["request_count"]["avg"] = count - 1
        summary["error_request_count"] = {"avg": 1}
    if mode == "metric-null":
        summary["request_throughput"] = None
    content = "{" if mode == "invalid" else json.dumps(summary)
    (directory / "profile_export_aiperf.json").write_text(content)
    if mode == "csv-missing":
        return
    csv_count = "" if mode == "csv-empty-count" else str(summary["request_count"]["avg"])
    csv_content = {
        "csv-invalid": 'Metric,avg\n"unterminated',
        "csv-truncated": "Metric,avg,p99\nLatency\n",
    }.get(
        mode,
        f"Metric,avg,p99\nLatency,1,2\n\nMetric,Value\nRequest Count,{csv_count}\n",
    )
    (directory / "profile_export_aiperf.csv").write_text(csv_content)


def main() -> None:
    mode = os.environ["PROOF_MODE"]
    args = sys.argv
    if calls := os.environ.get("MODEL_CALLS"):
        model = args[args.index("--model") + 1]
        with Path(calls).open("a") as stream:
            stream.write(model + "\n")
    if mode == "exit":
        sys.exit(23)
    if mode == "missing":
        return

    directory = Path(args[args.index("--artifact-dir") + 1])
    count = int(args[args.index("--request-count") + 1])
    write_artifacts(directory, count, mode)


if __name__ == "__main__":
    main()
