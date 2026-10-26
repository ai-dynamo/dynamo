# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runs inside one region's load-generator Job.

Installs aiperf and fetches the dataset and tokenizer first, so every region
starts aiperf at the shared ``START_AT``. aiperf then rebuilds the traces,
which takes a few minutes and varies by region, so the measurement window is
absolute: ``WINDOW_OFFSET_S`` after ``START_AT``, for ``MEASURE_S``. Prints one
``BENCH_SUMMARY`` line: client metrics over that window, and how many requests
each local pool frontend served.
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

RECORDS = Path("/work/artifacts/profile_export.jsonl")


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def metric(record, name):
    value = (record.get("metrics") or {}).get(name)
    if isinstance(value, dict):
        value = value.get("value")
    if isinstance(value, list):
        value = sum(value) / len(value) if value else None
    return float(value) if isinstance(value, (int, float)) else None


def window_summary(records, start, end):
    """Client metrics for requests that finished in ``[start, end)``, epoch
    seconds. Every region uses the same window, so they measure while all of
    them are loading the fleet."""
    profiling = [
        record
        for record in records
        if (record.get("metadata") or {}).get("phase_kind", "profiling") == "profiling"
    ]
    starts = [
        record["metadata"]["request_start_ns"]
        for record in profiling
        if record["metadata"].get("request_start_ns")
    ]
    if not starts:
        return {"completed_requests": 0, "profiling_errors": 0}
    measure_s = end - start
    finished = [
        record
        for record in profiling
        if record["metadata"].get("request_end_ns")
        and start <= record["metadata"]["request_end_ns"] / 1e9 < end
        and not record.get("error")
    ]

    def stats(name):
        values = [v for v in (metric(r, name) for r in finished) if v is not None]
        return {"p50": percentile(values, 0.5), "p99": percentile(values, 0.99)}

    output_tokens = sum(metric(r, "output_sequence_length") or 0 for r in finished)
    input_tokens = [metric(r, "input_sequence_length") for r in finished]
    input_tokens = [v for v in input_tokens if v is not None]
    return {
        "window": [start, end],
        # Profiling must already be under way when the window opens.
        "profiling_started": min(starts) / 1e9,
        "completed_requests": len(finished),
        "profiling_errors": sum(1 for r in profiling if r.get("error")),
        "requests_per_second": len(finished) / measure_s,
        "output_tokens_per_second": output_tokens / measure_s,
        "mean_input_tokens": sum(input_tokens) / len(input_tokens) if input_tokens else None,
        "ttft_ms": stats("time_to_first_token"),
        "itl_ms": stats("inter_token_latency"),
        "request_latency_ms": stats("request_latency"),
    }


def requests_served(url):
    """Total requests a Dynamo frontend has counted, from its Prometheus page."""
    text = urllib.request.urlopen(url, timeout=10).read().decode()
    total = 0.0
    for line in text.splitlines():
        if line.startswith("dynamo_frontend_requests_total"):
            total += float(line.rsplit(" ", 1)[1])
    return total


def main():
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir",
         f"aiperf=={os.environ['AIPERF_VERSION']}"],
        check=True,
    )
    from huggingface_hub import snapshot_download

    snapshot_download(os.environ["DATASET_REPO"], repo_type="dataset")
    snapshot_download(
        os.environ["TOKENIZER"],
        allow_patterns=["*.json", "*.txt", "*.model", "*.tiktoken"],
    )
    delay = float(os.environ["START_AT"]) - time.time()
    print(f"BENCH ready; starting in {delay:.0f}s", flush=True)
    if delay > 0:
        time.sleep(delay)

    pools = dict(
        entry.split("=", 1) for entry in os.environ["METRICS_URLS"].split(",") if entry
    )
    before = {stem: requests_served(url) for stem, url in pools.items()}
    began = time.time()
    code = subprocess.run(["aiperf", "profile", "--config", "/bench/aiperf.json"]).returncode
    after = {stem: requests_served(url) for stem, url in pools.items()}

    records = []
    if RECORDS.exists():
        with RECORDS.open() as handle:
            for line in handle:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    window_start = float(os.environ["START_AT"]) + float(os.environ["WINDOW_OFFSET_S"])
    summary = {
        "aiperf_exit": code,
        "began": began,
        "client": window_summary(
            records, window_start, window_start + float(os.environ["MEASURE_S"])
        ),
        "pool_requests": {stem: after[stem] - before[stem] for stem in pools},
    }
    print("BENCH_SUMMARY " + json.dumps(summary, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
