#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fresh processes, alternating A/B order; JSONL contains every raw measurement."""
import argparse
import json
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--baseline-bin", required=True)
p.add_argument("--final-bin", required=True)
p.add_argument("--output", required=True)
p.add_argument("--duration-ms", type=int, default=1000)
p.add_argument("--repeats", type=int, default=5)
p.add_argument("--raw", default="raw-results.jsonl")
a = p.parse_args()
root = Path(a.output).resolve()
root.mkdir(parents=True, exist_ok=True)
binaries = {
    "baseline": Path(a.baseline_bin).resolve(),
    "final": Path(a.final_bin).resolve(),
}
raw = root / a.raw
if raw.exists():
    raise SystemExit("Raw results already exist: " + str(raw))
measurements = []
fixtures = [
    (case, readers)
    for case in ["sparse", "incomplete", "dense", "populated_short"]
    for readers in [1, 16]
]
with raw.open("w", buffering=1) as f:
    for repeat in range(a.repeats):
        # Reverse fixture order as well as arm order to distribute warm/late runs.
        cases = fixtures if repeat % 2 == 0 else list(reversed(fixtures))
        arms = ["baseline", "final"] if repeat % 2 == 0 else ["final", "baseline"]
        for case, readers in cases:
            for order, arm in enumerate(arms):
                command = [str(binaries[arm]), case, str(readers), str(a.duration_ms)]
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=max(60, a.duration_ms / 1000 + 30),
                )
                row = json.loads(completed.stdout)
                row.update(
                    arm=arm,
                    repeat=repeat + 1,
                    order_in_pair=order + 1,
                    requested_duration_ms=a.duration_ms,
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                )
                if completed.stderr:
                    row["stderr"] = completed.stderr
                f.write(json.dumps(row, sort_keys=True) + "\n")
                measurements.append(row)
                print(
                    arm,
                    case,
                    readers,
                    "repeat",
                    repeat + 1,
                    round(row["lookups_per_second"], 1),
                    "lookup/s",
                    flush=True,
                )
                time.sleep(0.2)
summary = []
for case, readers in fixtures:
    row = dict(case=case, readers=readers)
    for arm in ["baseline", "final"]:
        runs = [
            m
            for m in measurements
            if m["case"] == case and m["readers"] == readers and m["arm"] == arm
        ]
        for key in [
            "lookups_per_second",
            "sampled_latency_p50_us",
            "sampled_latency_p95_us",
            "fixture_ingest_seconds",
        ]:
            row[arm + "_" + key] = statistics.median(m[key] for m in runs)
    row["throughput_ratio"] = (
        row["final_lookups_per_second"] / row["baseline_lookups_per_second"]
    )
    summary.append(row)
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
