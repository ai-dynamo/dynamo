# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Extract TTFT attribution data from server.log [PERF] lines and aiperf JSON.

Parses one or more config directories (e.g. vllm-serve/, dynamo-fd/) that each
contain request_rate*/server.log and request_rate*/profile_export_aiperf.json.

Outputs a JSON file with per-config, per-rate component statistics.

Usage:
    python benchmarks/multimodal/sweep/scripts/extract_attribution.py \
        --results-dir logs/04-09/h100-fd-pass5-breakdown/60req_4img_240pool_400word_base64 \
        -o attribution.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

# [PERF] component key=value ...
PERF_RE = re.compile(r"\[PERF\]\s+(\S+)\s+(.*)")
TIME_MS_RE = re.compile(r"time_ms=([\d.]+)")
TOTAL_MS_RE = re.compile(r"total_ms=([\d.]+)")
FETCH_MS_RE = re.compile(r"fetch_ms=([\d.]+)")
DECODE_MS_RE = re.compile(r"decode_ms=([\d.]+)")
REGISTER_MS_RE = re.compile(r"register_ms=([\d.]+)")


def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = (len(s) - 1) * p / 100
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def parse_perf_lines(path: Path) -> dict[str, list[float]]:
    """Parse [PERF] lines from server.log, returning component -> list of ms values."""
    components: dict[str, list[float]] = {}
    with open(path) as f:
        for line in f:
            m = PERF_RE.search(line)
            if not m:
                continue
            component = m.group(1)
            rest = m.group(2)

            # Skip "Patched" setup lines
            if "Patched" in rest:
                continue

            for regex, suffix in [
                (TIME_MS_RE, ""),
                (TOTAL_MS_RE, ".total"),
                (FETCH_MS_RE, ".fetch"),
                (DECODE_MS_RE, ".decode"),
                (REGISTER_MS_RE, ".register"),
            ]:
                match = regex.search(rest)
                if match:
                    key = f"{component}{suffix}" if suffix else component
                    components.setdefault(key, []).append(float(match.group(1)))
    return components


def compute_stats(values: list[float]) -> dict[str, float]:
    return {
        "count": len(values),
        "avg": round(statistics.mean(values), 2),
        "p50": round(percentile(values, 50), 2),
        "p90": round(percentile(values, 90), 2),
        "p99": round(percentile(values, 99), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
    }


def load_aiperf_ttft(path: Path) -> dict[str, float] | None:
    """Extract time_to_first_token stats from profile_export_aiperf.json."""
    try:
        data = json.loads(path.read_text())
        ttft = data.get("time_to_first_token", {})
        return {
            "avg": round(ttft["avg"], 2),
            "p50": round(ttft["p50"], 2),
            "p90": round(ttft["p90"], 2),
            "p99": round(ttft["p99"], 2),
        }
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return None


def extract_config(config_dir: Path) -> dict[str, Any]:
    """Extract attribution data for one config (e.g. vllm-serve/)."""
    result: dict[str, Any] = {"rates": {}}

    for rate_dir in sorted(config_dir.iterdir()):
        if not rate_dir.is_dir() or not rate_dir.name.startswith("request_rate"):
            continue

        rate_str = rate_dir.name.replace("request_rate", "")
        rate_data: dict[str, Any] = {}

        # Parse [PERF] from server.log
        server_log = rate_dir / "server.log"
        if server_log.exists():
            components = parse_perf_lines(server_log)
            rate_data["components"] = {
                k: compute_stats(v) for k, v in sorted(components.items())
            }
        else:
            rate_data["components"] = {}

        # Parse TTFT from aiperf
        aiperf_json = rate_dir / "profile_export_aiperf.json"
        ttft = load_aiperf_ttft(aiperf_json)
        if ttft:
            rate_data["ttft"] = ttft

        result["rates"][rate_str] = rate_data

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract TTFT attribution data from [PERF] logs + aiperf JSON"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing config subdirs (vllm-serve/, dynamo-fd/, ...)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: <results-dir>/attribution.json)",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    if not results_dir.exists():
        print(f"ERROR: {results_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    output: dict[str, Any] = {"configs": {}}

    for config_dir in sorted(results_dir.iterdir()):
        if not config_dir.is_dir() or config_dir.name in ("plots",):
            continue
        config_name = config_dir.name
        config_data = extract_config(config_dir)
        if config_data["rates"]:
            output["configs"][config_name] = config_data

    out_path = args.output or (results_dir / "attribution.json")
    out_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Written: {out_path}")
    print(f"Configs: {list(output['configs'].keys())}")
    for cfg, data in output["configs"].items():
        for rate, rd in data["rates"].items():
            n_comp = len(rd.get("components", {}))
            has_ttft = "ttft" in rd
            print(f"  {cfg} rate={rate}: {n_comp} components, ttft={'yes' if has_ttft else 'no'}")


if __name__ == "__main__":
    main()
