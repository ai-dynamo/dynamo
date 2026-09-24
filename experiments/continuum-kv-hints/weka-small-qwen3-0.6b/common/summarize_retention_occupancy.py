# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize worker-local retained G1 occupancy over one benchmark window."""

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

OCCUPANCY_RE = re.compile(
    r"KV retention occupancy: reason=(?P<reason>\w+) "
    r"retained_blocks=(?P<retained>\d+) total_blocks=(?P<total>\d+) "
    r"retained_fraction=(?P<fraction>[0-9.]+) "
    r"event_time_seconds=(?P<timestamp>[0-9.]+)"
)
PHASE_WINDOW_RE = re.compile(
    r"baseline_start_ns=(?P<start>\d+), baseline_end_ns=(?P<end>\d+)"
)


@dataclass(frozen=True)
class Sample:
    timestamp: float
    retained_blocks: int
    total_blocks: int
    retained_fraction: float
    reason: str


def parse_samples(worker_log: Path) -> list[Sample]:
    samples = []
    for line in worker_log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = OCCUPANCY_RE.search(line)
        if match is None:
            continue
        samples.append(
            Sample(
                timestamp=float(match["timestamp"]),
                retained_blocks=int(match["retained"]),
                total_blocks=int(match["total"]),
                retained_fraction=float(match["fraction"]),
                reason=match["reason"],
            )
        )
    return sorted(samples, key=lambda sample: sample.timestamp)


def parse_aiperf_window(aiperf_log: Path) -> tuple[float, float]:
    matches = list(
        PHASE_WINDOW_RE.finditer(
            aiperf_log.read_text(encoding="utf-8", errors="replace")
        )
    )
    if not matches:
        raise ValueError(f"profiling window not found in {aiperf_log}")
    match = matches[-1]
    return int(match["start"]) / 1e9, int(match["end"]) / 1e9


def weighted_quantile(segments: list[tuple[float, float]], quantile: float) -> float:
    total_duration = sum(duration for _, duration in segments)
    if total_duration <= 0:
        return 0.0
    threshold = total_duration * quantile
    cumulative = 0.0
    for value, duration in sorted(segments):
        cumulative += duration
        if cumulative >= threshold:
            return value
    return max(value for value, _ in segments)


def summarize(
    samples: list[Sample], start: float, end: float
) -> tuple[dict[str, object], list[Sample]]:
    if end <= start:
        raise ValueError("benchmark end must be after benchmark start")

    current = Sample(start, 0, 0, 0.0, "benchmark_start")
    for sample in samples:
        if sample.timestamp > start:
            break
        current = Sample(
            start,
            sample.retained_blocks,
            sample.total_blocks,
            sample.retained_fraction,
            "benchmark_start",
        )

    timeline = [current]
    segments: list[tuple[float, float]] = []
    cursor = start
    reasons: Counter[str] = Counter()
    for sample in samples:
        if sample.timestamp <= start or sample.timestamp >= end:
            continue
        segments.append((current.retained_fraction, sample.timestamp - cursor))
        current = sample
        timeline.append(sample)
        cursor = sample.timestamp
        reasons[sample.reason] += 1
    segments.append((current.retained_fraction, end - cursor))
    timeline.append(
        Sample(
            end,
            current.retained_blocks,
            current.total_blocks,
            current.retained_fraction,
            "benchmark_end",
        )
    )

    duration = end - start
    mean = sum(value * seconds for value, seconds in segments) / duration
    peak_sample = max(timeline, key=lambda sample: sample.retained_fraction)
    summary: dict[str, object] = {
        "benchmark_duration_seconds": duration,
        "peak_retained_blocks": peak_sample.retained_blocks,
        "total_blocks": peak_sample.total_blocks,
        "peak_retained_fraction": peak_sample.retained_fraction,
        "time_weighted_mean_retained_fraction": mean,
        "time_weighted_p50_retained_fraction": weighted_quantile(segments, 0.50),
        "time_weighted_p95_retained_fraction": weighted_quantile(segments, 0.95),
        "final_retained_blocks": current.retained_blocks,
        "final_retained_fraction": current.retained_fraction,
        "occupancy_change_count": len(timeline) - 2,
        "change_reasons": dict(sorted(reasons.items())),
    }
    return summary, timeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-log", type=Path, required=True)
    parser.add_argument("--aiperf-log", type=Path)
    parser.add_argument("--benchmark-start", type=Path)
    parser.add_argument("--benchmark-end", type=Path)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--series-output", type=Path, required=True)
    args = parser.parse_args()

    if args.aiperf_log is not None:
        start, end = parse_aiperf_window(args.aiperf_log)
    elif args.benchmark_start is not None and args.benchmark_end is not None:
        start = float(args.benchmark_start.read_text().strip())
        end = float(args.benchmark_end.read_text().strip())
    else:
        parser.error(
            "provide --aiperf-log or both --benchmark-start and --benchmark-end"
        )
    summary, timeline = summarize(parse_samples(args.worker_log), start, end)
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n")
    with args.series_output.open("w", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "elapsed_seconds",
                "event_time_seconds",
                "retained_blocks",
                "total_blocks",
                "retained_fraction",
                "reason",
            ),
        )
        writer.writeheader()
        for sample in timeline:
            writer.writerow(
                {
                    "elapsed_seconds": sample.timestamp - start,
                    "event_time_seconds": sample.timestamp,
                    "retained_blocks": sample.retained_blocks,
                    "total_blocks": sample.total_blocks,
                    "retained_fraction": sample.retained_fraction,
                    "reason": sample.reason,
                }
            )

    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
