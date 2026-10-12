# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate the GPU shard timing snapshot from analyzed per-test medians."""

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source", type=Path, help="Analyzed test-durations.json")
parser.add_argument(
    "--output",
    type=Path,
    default=Path(__file__).resolve().parents[1] / "ci/pre-merge-test-durations.json",
)
parser.add_argument("--check", action="store_true")
parser.add_argument("--start", default="2026-10-02T04:00:00Z")
parser.add_argument("--end", default="2026-10-09T04:00:00Z")
args = parser.parse_args()
source = json.loads(args.source.read_text())
data = {
    "_meta": {
        "generated": "Do not edit; use .github/scripts/generate_test_shard_durations.py",
        "window_utc": [args.start, args.end],
        "estimator": "Median successful duration; JUnit for sequential tests, scheduler logs for profiled subprocesses",
    }
}
for backend in ("vllm", "sglang", "trtllm"):
    for stage in ("pre_merge_gpu", "pre_merge_gpu_parallel"):
        key = f"{backend}/{stage}/amd64"
        weights = source[key]
        if not weights or not any(value > 0 for value in weights.values()):
            raise SystemExit(f"No positive timing evidence for {key}")
        data[key] = weights
output = json.dumps(data, indent=2, sort_keys=True) + "\n"
if args.check:
    if args.output.read_text() != output:
        raise SystemExit("GPU test timing snapshot is stale")
else:
    args.output.write_text(output)
