#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate cache_on vs cache_off sweep cells into a single comparison table.

Reads each cell's aiperf JSON output and prometheus metrics, extracts
request_throughput / latency / cache stats, and prints a formatted table.

Usage:
    tokenizer_cache_aggregate.py <sweep_root_dir>
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def parse_metric(text: str, name: str) -> float | None:
    """Extract the value of a Prometheus counter/gauge by name from raw text."""
    pat = re.compile(rf"^{re.escape(name)}\s+([\d.eE+-]+)\s*$", re.MULTILINE)
    m = pat.search(text)
    return float(m.group(1)) if m else None


def parse_cell(cell_dir: Path) -> dict | None:
    """Pull RPS, latency p50/p99, and cache hit/miss counts from one cell directory."""
    aiperf_json = cell_dir / "aiperf" / "profile_export_aiperf.json"
    if not aiperf_json.exists():
        return None
    data = json.loads(aiperf_json.read_text())

    def get_avg(key: str) -> float | None:
        v = data.get(key)
        return v.get("avg") if isinstance(v, dict) else None

    rps = get_avg("request_throughput")
    p50 = get_avg("request_latency")
    e2e_tput = get_avg("total_token_throughput")
    rcount = (data.get("request_count") or {}).get("avg")
    duration = data.get("benchmark_duration")

    # Cache metric deltas
    initial_text = (
        (cell_dir / "metrics_initial.txt").read_text()
        if (cell_dir / "metrics_initial.txt").exists()
        else ""
    )
    final_text = (
        (cell_dir / "metrics_final.txt").read_text()
        if (cell_dir / "metrics_final.txt").exists()
        else ""
    )
    hits_i = (
        parse_metric(initial_text, "dynamo_frontend_tokenizer_cache_hits_total") or 0.0
    )
    hits_f = (
        parse_metric(final_text, "dynamo_frontend_tokenizer_cache_hits_total") or 0.0
    )
    miss_i = (
        parse_metric(initial_text, "dynamo_frontend_tokenizer_cache_misses_total")
        or 0.0
    )
    miss_f = (
        parse_metric(final_text, "dynamo_frontend_tokenizer_cache_misses_total") or 0.0
    )
    hits = hits_f - hits_i
    misses = miss_f - miss_i
    hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0.0

    return dict(
        rps=rps,
        p50_latency_s=p50,
        total_tok_per_sec=e2e_tput,
        requests=rcount,
        duration_s=duration,
        hits=hits,
        misses=misses,
        hit_rate=hit_rate,
    )


def main(root: Path) -> int:
    cells: dict[tuple[str, int], dict] = {}
    for cell_dir in sorted(root.glob("cache=*_conc=*")):
        m = re.match(r"cache=(off|on)_conc=(\d+)", cell_dir.name)
        if not m:
            continue
        cache = m.group(1)
        conc = int(m.group(2))
        parsed = parse_cell(cell_dir)
        if parsed is None:
            print(f"[skip] {cell_dir.name}: no aiperf output")
            continue
        cells[(cache, conc)] = parsed

    # Find concurrency levels exercised
    concurrencies = sorted({c for (_, c) in cells.keys()})
    if not concurrencies:
        print("No completed cells found.")
        return 1

    # Header
    cols = [
        ("conc", 6),
        ("rps_off", 10),
        ("rps_on", 10),
        ("speedup", 9),
        ("p50_off_ms", 12),
        ("p50_on_ms", 12),
        ("hit_rate_on", 13),
        ("hits_on", 10),
        ("misses_on", 10),
    ]
    print("\nTokenizer L1 cache sweep — Qwen3-0.6B, 2 mockers, frontend taskset 0-3")
    print("  shared system prompt: 48000 tokens   user context: 12000 tokens")
    print("  output tokens: 500   benchmark: 60s   speedup-ratio: 1000000")
    print()
    print(" | ".join(f"{name:>{w}}" for name, w in cols))
    print("-+-".join("-" * w for _, w in cols))

    for conc in concurrencies:
        off = cells.get(("off", conc))
        on = cells.get(("on", conc))
        rps_off = off.get("rps") if off else None
        rps_on = on.get("rps") if on else None
        speedup = (rps_on / rps_off) if (rps_off and rps_on) else None
        p50_off = off.get("p50_latency_s") if off else None
        p50_on = on.get("p50_latency_s") if on else None
        hit_rate = on.get("hit_rate") if on else 0.0
        hits = on.get("hits") if on else 0
        misses = on.get("misses") if on else 0

        cells_out = [
            f"{conc:>6}",
            f"{rps_off:>10.2f}" if rps_off else f"{'-':>10}",
            f"{rps_on:>10.2f}" if rps_on else f"{'-':>10}",
            f"{speedup:>9.2f}x" if speedup else f"{'-':>9}",
            f"{p50_off:>12.1f}" if p50_off else f"{'-':>12}",
            f"{p50_on:>12.1f}" if p50_on else f"{'-':>12}",
            (
                f"{hit_rate * 100:>11.2f}%"
                if (on and (hits + misses) > 0)
                else f"{'-':>13}"
            ),
            f"{hits:>10.0f}" if hits else f"{'-':>10}",
            f"{misses:>10.0f}" if misses else f"{'-':>10}",
        ]
        print(" | ".join(cells_out))

    print()
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: tokenizer_cache_aggregate.py <sweep_root_dir>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(Path(sys.argv[1])))
