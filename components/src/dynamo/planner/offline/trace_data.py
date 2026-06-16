# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from collections import defaultdict
from typing import Any, Dict, List


def extract_metrics_from_mooncake(
    dataset: str, throughput_adjustment_interval_seconds: int
) -> List[Dict[str, Any]]:
    """
    Extract metrics from mooncake-style JSONL data.

    Args:
        dataset: Path to the JSONL file containing mooncake trace data
        throughput_adjustment_interval_seconds: Time interval to group requests

    Returns:
        List of dictionaries (ordered by time) containing metrics for each interval:
        - interval_start: Start time of the interval (in seconds)
        - request_count: Total number of requests in the interval
        - avg_isl: Average input sequence length
        - avg_osl: Average output sequence length

        Intervals with no requests *between* the first and last active interval
        are emitted as empty windows (request_count=0, avg_isl=0, avg_osl=0) so
        gaps in traffic are preserved. This matches the planner's online
        throughput loop, which feeds zero-traffic windows to the load
        predictors; collapsing the gaps would let warmup diverge from live
        behavior. Leading and trailing empty intervals are still omitted (the
        series starts at the first interval with traffic and ends at the last),
        matching the predictors' leading-idle skip.
    """
    # Read and parse JSONL data from file
    records = []
    with open(dataset, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    interval_groups = defaultdict(list)

    for record in records:
        timestamp_ms = record["timestamp"]
        timestamp_sec = timestamp_ms / 1000
        interval_start = (
            int(timestamp_sec // throughput_adjustment_interval_seconds)
            * throughput_adjustment_interval_seconds
        )
        interval_groups[interval_start].append(record)

    # Compute metrics for each interval. Walk every interval from the first to
    # the last active one (step = interval), filling gaps with empty windows so
    # zero-traffic intervals are preserved (matching the online throughput loop).
    metrics: List[Dict[str, Any]] = []
    if not interval_groups:
        return metrics

    sorted_starts = sorted(interval_groups.keys())
    for interval_start in range(
        sorted_starts[0],
        sorted_starts[-1] + 1,
        throughput_adjustment_interval_seconds,
    ):
        records_in_interval = interval_groups.get(interval_start, [])

        # Calculate metrics
        request_count = len(records_in_interval)

        # Calculate average ISL and OSL
        total_isl = sum(record["input_length"] for record in records_in_interval)
        total_osl = sum(record["output_length"] for record in records_in_interval)

        avg_isl = total_isl / request_count if request_count > 0 else 0
        avg_osl = total_osl / request_count if request_count > 0 else 0

        metrics.append(
            {
                "interval_start": interval_start,
                "request_count": request_count,
                "avg_isl": avg_isl,
                "avg_osl": avg_osl,
            }
        )

    return metrics
