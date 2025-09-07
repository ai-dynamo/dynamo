# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from collections import defaultdict
from typing import Any, Dict, List


def extract_metrics_from_mooncake(
    dataset: str, adjustment_interval: int
) -> List[Dict[str, Any]]:
    """
    Extract metrics from mooncake-style JSONL data.

    Args:
        dataset: Path to the JSONL file containing mooncake trace data
        adjustment_interval: Time interval in seconds to group requests

    Returns:
        List of dictionaries containing metrics for each interval:
        - interval_start: Start time of the interval (in seconds)
        - request_count: Total number of requests in the interval
        - avg_isl: Average input sequence length
        - avg_osl: Average output sequence length
    """
    # Read and parse JSONL data from file
    records = []
    with open(dataset, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Group records by adjustment interval
    interval_groups = defaultdict(list)

    for record in records:
        timestamp_ms = record["timestamp"]
        # Convert milliseconds to seconds and find the interval
        timestamp_sec = timestamp_ms / 1000
        interval_start = int(timestamp_sec // adjustment_interval) * adjustment_interval
        interval_groups[interval_start].append(record)

    # Compute metrics for each interval
    metrics = []

    for interval_start in sorted(interval_groups.keys()):
        records_in_interval = interval_groups[interval_start]

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
