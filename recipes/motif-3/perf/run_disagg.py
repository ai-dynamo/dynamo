#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run a 32-request warm-cache burst, then replay the complete C64 trace."""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path

import yaml

TRACE_SHA256 = "b1221bca72b69f842897f339624306a84857f1b55ea0d866525f94d9ceb9b871"
WARMUP_REQUESTS = 32


def prepare_configs(trace: Path, root: Path) -> tuple[Path, Path]:
    """Keep the complete source trace so both runs synthesize identical prefixes."""
    data = trace.read_bytes()
    if hashlib.sha256(data).hexdigest() != TRACE_SHA256:
        raise ValueError("Expected the complete, unchanged 1,805-row chat trace")
    rows = [json.loads(line) for line in data.splitlines()]
    # The pinned trace's first 32 rows fit comfortably within the 262,144 context.
    # The margin also accommodates chat-template tokens and synthesis differences.
    eligible = [row for row in rows if row["input_length"] <= 256000]
    if len(rows) != 1805 or len(eligible) < WARMUP_REQUESTS:
        raise ValueError("Trace does not contain the expected requests")
    if any(
        row["input_length"] + row["output_length"] > 256000
        for row in eligible[:WARMUP_REQUESTS]
    ):
        raise ValueError("Warmup requests exceed the context safety margin")

    measured = yaml.safe_load(Path(__file__).with_name("disagg.yaml").read_text())
    measured["benchmark"]["dataset"]["path"] = str(trace.resolve())
    measured["benchmark"]["artifacts"]["dir"] = str(root / "measurement")
    warmup = copy.deepcopy(measured)
    warmup["benchmark"]["dataset"].update(
        entries=WARMUP_REQUESTS, synthesis={"max_isl": 256000}
    )
    warmup["benchmark"]["profiling"].update(
        concurrency=WARMUP_REQUESTS, requests=WARMUP_REQUESTS
    )
    warmup["benchmark"]["runtime"]["workers"] = WARMUP_REQUESTS
    warmup["benchmark"]["artifacts"]["dir"] = str(root / "warmup")

    root.mkdir(parents=True, exist_ok=False)
    paths = (root / "warmup.yaml", root / "measurement.yaml")
    for path, config in zip(paths, (warmup, measured), strict=True):
        path.write_text(yaml.safe_dump(config, sort_keys=False))
    (root / "protocol.json").write_text(
        json.dumps(
            {
                "trace_sha256": TRACE_SHA256,
                "warmup_requests": WARMUP_REQUESTS,
                "warmup_concurrency": WARMUP_REQUESTS,
                "measured_requests": 1805,
                "measured_concurrency": measured["variables"]["concurrency"],
                "cache_protocol": "preserve worker and router caches after warmup",
                "source_trace_unchanged": True,
            },
            indent=2,
        )
        + "\n"
    )
    return paths


def check_warmup(report_path: Path) -> None:
    report = json.loads(report_path.read_text())
    if (
        report.get("is_complete") is not True
        or report.get("was_cancelled") is not False
        or report.get("request_count", {}).get("avg") != WARMUP_REQUESTS
        or report.get("error_request_count", {}).get("avg", 0) != 0
    ):
        raise RuntimeError("Warmup must complete all 32 requests without errors")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-file",
        type=Path,
        default=Path(
            "/shared-model-cache/traces/"
            "8k_1k_70kv_chat_new_noschedule_short_15perc.jsonl"
        ),
    )
    parser.add_argument(
        "--artifact-dir", type=Path, default=os.environ.get("AIPERF_ARTIFACT_DIR")
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write configs without sending requests",
    )
    args = parser.parse_args()
    if args.artifact_dir is None:
        parser.error("Set AIPERF_ARTIFACT_DIR or --artifact-dir to a new directory")
    root = args.artifact_dir.resolve()
    if not args.prepare_only and importlib.metadata.version("aiperf") != "0.13.0":
        raise RuntimeError("This benchmark requires AIPerf 0.13.0")
    warmup, measured = prepare_configs(args.trace_file, root)
    if args.prepare_only:
        print(f"Prepared {warmup} and {measured}; no requests sent")
        return
    for config in (warmup, measured):
        subprocess.run(["aiperf", "config", "validate", str(config)], check=True)
    subprocess.run(["aiperf", "profile", "--config", str(warmup)], check=True)
    check_warmup(root / "warmup" / "report_aiperf.json")
    # Do not clear KV blocks or restart workers/frontend here: measurement is warm-cache.
    subprocess.run(["aiperf", "profile", "--config", str(measured)], check=True)


if __name__ == "__main__":
    main()
