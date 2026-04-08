#!/usr/bin/env python3
"""Parse [PERF] lines from sweep logs to produce a component-wise TTFT breakdown."""

import re
import statistics
import sys
from collections import defaultdict
from typing import Dict, List


def parse_perf_lines(log_path: str) -> Dict[str, List[float]]:
    """Extract [PERF] timing values from a log file.

    Returns dict mapping component name -> list of ms values.
    """
    components: Dict[str, List[float]] = defaultdict(list)

    with open(log_path) as f:
        for line in f:
            if "[PERF]" not in line:
                continue

            # [PERF] image_decode: 120ms src=data:image/jpeg;base64,
            m = re.search(r"\[PERF\]\s+image_decode:\s+([\d.]+)ms", line)
            if m:
                components["image_decode"].append(float(m.group(1)))
                continue

            # [PERF] _process_multimodal HF processor: 400ms for ...
            m = re.search(
                r"\[PERF\]\s+_process_multimodal HF processor:\s+([\d.]+)ms", line
            )
            if m:
                components["hf_processor"].append(float(m.group(1)))
                continue

            # [PERF] render_chat_request: 520ms for ...
            m = re.search(r"\[PERF\]\s+render_chat_request:\s+([\d.]+)ms", line)
            if m:
                components["render_chat_request"].append(float(m.group(1)))
                continue

            # [PERF] vision_encoder: 800ms reqs=[...]
            m = re.search(r"\[PERF\]\s+vision_encoder:\s+([\d.]+)ms", line)
            if m:
                components["vision_encoder"].append(float(m.group(1)))
                continue

            # [PERF] extract_mm_data: 50ms req=...
            m = re.search(r"\[PERF\]\s+extract_mm_data:\s+([\d.]+)ms", line)
            if m:
                components["extract_mm_data"].append(float(m.group(1)))
                continue

            # [PERF] nixl_read: 20ms alloc: 0.1ms shape=[...]
            m = re.search(r"\[PERF\]\s+nixl_read:\s+([\d.]+)ms", line)
            if m:
                components["nixl_read"].append(float(m.group(1)))
                continue

            # [PERF] rust_image: fetch=80ms decode=30ms
            m = re.search(
                r"\[PERF\]\s+rust_image: fetch=([\d.]+)ms decode=([\d.]+)ms", line
            )
            if m:
                components["rust_image_fetch"].append(float(m.group(1)))
                components["rust_image_decode"].append(float(m.group(2)))
                continue

            # [PERF] rust_fetch_decode_register: nixl=5ms total=120ms
            m = re.search(
                r"\[PERF\]\s+rust_fetch_decode_register: nixl=([\d.]+)ms total=([\d.]+)ms",
                line,
            )
            if m:
                components["rust_nixl_register"].append(float(m.group(1)))
                components["rust_per_image_total"].append(float(m.group(2)))
                continue

            # [PERF] rust_mm_gather: 500ms
            m = re.search(r"\[PERF\]\s+rust_mm_gather:\s+([\d.]+)ms", line)
            if m:
                components["rust_mm_gather"].append(float(m.group(1)))
                continue

    return dict(components)


def print_stats(components: Dict[str, List[float]], label: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(
        f"{'Component':<30s} {'Count':>6s} {'Avg':>8s} {'P50':>8s} {'P90':>8s} {'P99':>8s}"
    )
    print("-" * 70)

    for name in sorted(components.keys()):
        vals = components[name]
        if not vals:
            continue
        avg = statistics.mean(vals)
        p50 = statistics.median(vals)
        p90 = sorted(vals)[int(len(vals) * 0.9)] if len(vals) >= 10 else max(vals)
        p99 = sorted(vals)[int(len(vals) * 0.99)] if len(vals) >= 100 else max(vals)
        print(
            f"{name:<30s} {len(vals):>6d} {avg:>7.0f}ms {p50:>7.0f}ms {p90:>7.0f}ms {p99:>7.0f}ms"
        )


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: parse_ttft_breakdown.py <log_file> [log_file2 ...]")
        sys.exit(1)

    for log_path in sys.argv[1:]:
        components = parse_perf_lines(log_path)
        if not components:
            print(f"No [PERF] lines found in {log_path}")
            continue
        print_stats(components, log_path)


if __name__ == "__main__":
    main()
