#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""2-way comparison of vllm-serve vs dynamo-fd aiperf runs.

Reads each config's profile_export_aiperf.json from the retrieved
artifact tree and prints request throughput, TTFT (avg/p50/p90/p99),
and ITL (avg/p50/p90/p99) side-by-side.

Usage:
  python3 compare.py <results_dir>

Where <results_dir> is the directory written by run-all-benchmarks.sh,
e.g. ~/workspace/dynamo-tmp/logs/05-12/qwen35-fp8-h100/

Expects:
  <results_dir>/vllm-serve/.../profile_export_aiperf.json
  <results_dir>/dynamo-fd/.../profile_export_aiperf.json
"""
import json
import sys
from pathlib import Path

CONFIGS = ["vllm-serve", "dynamo-fd"]


def find_profile_json(config_dir: Path) -> Path | None:
    matches = list(config_dir.rglob("profile_export_aiperf.json"))
    if not matches:
        return None
    # Prefer the one nested under artifacts/qwen35_fp8/...
    matches.sort(key=lambda p: (0 if "qwen35_fp8" in str(p) else 1, len(str(p))))
    return matches[0]


def load_metrics(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def extract(metrics: dict, key: str, sub: str) -> float | None:
    return metrics.get(key, {}).get(sub)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    root = Path(sys.argv[1]).expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: not a directory: {root}", file=sys.stderr)
        return 2

    loaded: dict[str, dict | None] = {}
    for cfg in CONFIGS:
        cfg_dir = root / cfg
        path = find_profile_json(cfg_dir) if cfg_dir.is_dir() else None
        loaded[cfg] = load_metrics(path) if path else None
        if path:
            print(f"[{cfg}] {path.relative_to(root)}")
        else:
            print(f"[{cfg}] MISSING profile_export_aiperf.json under {cfg_dir}")

    print()
    header = f"{'metric':<28}" + "".join(f"{c:>14}" for c in CONFIGS)
    print(header)
    print("-" * len(header))

    rows = [
        ("request_throughput (rps)", ("request_throughput", "avg"), 3),
        ("TTFT avg (ms)", ("time_to_first_token", "avg"), 1),
        ("TTFT p50 (ms)", ("time_to_first_token", "p50"), 1),
        ("TTFT p90 (ms)", ("time_to_first_token", "p90"), 1),
        ("TTFT p99 (ms)", ("time_to_first_token", "p99"), 1),
        ("ITL avg (ms)", ("inter_token_latency", "avg"), 2),
        ("ITL p50 (ms)", ("inter_token_latency", "p50"), 2),
        ("ITL p90 (ms)", ("inter_token_latency", "p90"), 2),
        ("ITL p99 (ms)", ("inter_token_latency", "p99"), 2),
    ]

    for label, (k, sub), decimals in rows:
        vals = [extract(loaded[c], k, sub) if loaded[c] else None for c in CONFIGS]
        line = f"{label:<28}" + "".join(
            f"{(f'{v:.{decimals}f}' if v is not None else 'n/a'):>14}" for v in vals
        )
        print(line)

    print()
    # Pct delta vs vllm-serve for the headline metrics.
    if loaded["vllm-serve"]:
        print("Delta dynamo-fd vs vllm-serve baseline (negative = faster / higher):")
        print(f"{'metric':<28}{'dynamo-fd':>14}")
        for label, (k, sub), _ in rows:
            base = extract(loaded["vllm-serve"], k, sub)
            if base is None or base == 0:
                continue
            v = extract(loaded["dynamo-fd"], k, sub) if loaded["dynamo-fd"] else None
            if v is None:
                delta = "n/a"
            else:
                delta = f"{(v - base) / base * 100.0:+.1f}%"
            print(f"{label:<28}{delta:>14}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
