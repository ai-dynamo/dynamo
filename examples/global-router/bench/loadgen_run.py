# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runs inside one region's load-generator Job.

Installs aiperf and fetches the dataset and tokenizer first, so every region
starts aiperf at the shared ``START_AT``. aiperf then rebuilds the traces,
which takes a few minutes and varies by region, so the measurement window is
absolute: ``WINDOW_OFFSET_S`` after ``START_AT``, for ``MEASURE_S``. Prints one
``BENCH_SUMMARY`` line: client metrics over that window, and how many requests
each local pool frontend served.

With ``DAY_S`` set, the Job replays one compressed day instead. A one-request
run first leaves the rebuilt traces in aiperf's cache, so the day itself
starts within seconds of ``START_AT`` in every region. The whole day is
measured. ``BENCH_SUMMARY`` covers the day, with the Global Ingress's routing
counts, and ``BENCH_HOURLY`` breaks the day into 24 simulated hours.
"""

import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from collections import Counter
from pathlib import Path

RECORDS = Path("/work/artifacts/profile_export.jsonl")
HOURS = 24
# Routing counts summed over the day and per hour.
ROUTER_COUNTS = ("requests", "in_region", "with_session", "later_turns", "same_pool")


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


def error_kind(error):
    """A short label for an aiperf record's error: its code and type."""
    if isinstance(error, dict):
        parts = (error.get("code"), error.get("type"))
        return " ".join(str(part) for part in parts if part is not None)
    return str(error)[:80]


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
        "error_kinds": dict(
            Counter(error_kind(r["error"]) for r in profiling if r.get("error"))
        ),
        "requests_per_second": len(finished) / measure_s,
        "output_tokens_per_second": output_tokens / measure_s,
        "mean_input_tokens": sum(input_tokens) / len(input_tokens)
        if input_tokens
        else None,
        "ttft_ms": stats("time_to_first_token"),
        "itl_ms": stats("inter_token_latency"),
        "request_latency_ms": stats("request_latency"),
    }


def hourly_summary(records, day_start, hour_s):
    """Per simulated hour, by when each request was sent: requests sent,
    requests completed, errors, and TTFT p50/p99 of the completed ones."""
    hours = [{"sent": 0, "completed": 0, "errors": 0, "ttft": []} for _ in range(HOURS)]
    for record in records:
        metadata = record.get("metadata") or {}
        if metadata.get("phase_kind", "profiling") != "profiling":
            continue
        if not metadata.get("request_start_ns"):
            continue
        hour = int((metadata["request_start_ns"] / 1e9 - day_start) // hour_s)
        if not 0 <= hour < HOURS:
            continue
        bucket = hours[hour]
        bucket["sent"] += 1
        if record.get("error"):
            bucket["errors"] += 1
            continue
        bucket["completed"] += 1
        ttft = metric(record, "time_to_first_token")
        if ttft is not None:
            bucket["ttft"].append(ttft)
    for bucket in hours:
        ttft = bucket.pop("ttft")
        bucket["ttft_p50"] = percentile(ttft, 0.5)
        bucket["ttft_p99"] = percentile(ttft, 0.99)
    return hours


def router_deltas(snapshots):
    """Differences between consecutive Global Ingress stats snapshots: the
    routing counts, and the requests routed to each pool."""
    deltas = []
    for earlier, later in zip(snapshots, snapshots[1:]):
        delta = {key: later[key] - earlier[key] for key in ROUTER_COUNTS}
        routed = {pool["name"]: pool["routed"] for pool in earlier["pools"]}
        delta["pools"] = {
            pool["name"]: pool["routed"] - routed.get(pool["name"], 0)
            for pool in later["pools"]
        }
        deltas.append(delta)
    return deltas


def read_stats(url, snapshots):
    try:
        snapshots.append(json.load(urllib.request.urlopen(url, timeout=10)))
    except OSError as error:
        print(f"BENCH cannot read {url}: {error}", flush=True)


def poll_stats(url, interval_s, snapshots, stop):
    """Appends a stats snapshot now and every ``interval_s`` until ``stop``."""
    while True:
        read_stats(url, snapshots)
        if stop.wait(interval_s):
            return


def requests_served(url):
    """Total requests a Dynamo frontend has counted, from its Prometheus page."""
    text = urllib.request.urlopen(url, timeout=10).read().decode()
    total = 0.0
    for line in text.splitlines():
        if line.startswith("dynamo_frontend_requests_total"):
            total += float(line.rsplit(" ", 1)[1])
    return total


def read_records():
    records = []
    if RECORDS.exists():
        with RECORDS.open() as handle:
            for line in handle:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def main():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-cache-dir",
            f"aiperf=={os.environ['AIPERF_VERSION']}",
        ],
        check=True,
    )
    from huggingface_hub import snapshot_download

    snapshot_download(os.environ["DATASET_REPO"], repo_type="dataset")
    snapshot_download(
        os.environ["TOKENIZER"],
        allow_patterns=["*.json", "*.txt", "*.model", "*.tiktoken"],
    )
    day_s = float(os.environ["DAY_S"]) if "DAY_S" in os.environ else None
    if day_s:
        prepared = subprocess.run(
            ["aiperf", "profile", "--config", "/bench/prepare.json"]
        ).returncode
        print(f"BENCH prepared the traces (aiperf exit {prepared})", flush=True)
    delay = float(os.environ["START_AT"]) - time.time()
    print(f"BENCH ready; starting in {delay:.0f}s", flush=True)
    if delay > 0:
        time.sleep(delay)

    pools = dict(
        entry.split("=", 1) for entry in os.environ["METRICS_URLS"].split(",") if entry
    )
    before = {stem: requests_served(url) for stem, url in pools.items()}
    snapshots, stop, poller = [], threading.Event(), None
    if day_s:
        poller = threading.Thread(
            target=poll_stats,
            args=(os.environ["STATS_URL"], day_s / HOURS, snapshots, stop),
            daemon=True,
        )
        poller.start()
    began = time.time()
    code = subprocess.run(
        ["aiperf", "profile", "--config", "/bench/aiperf.json"]
    ).returncode
    after = {stem: requests_served(url) for stem, url in pools.items()}
    if poller:
        stop.set()
        poller.join()
        read_stats(os.environ["STATS_URL"], snapshots)

    records = read_records()
    summary = {
        "aiperf_exit": code,
        "began": began,
        "pool_requests": {stem: after[stem] - before[stem] for stem in pools},
    }
    if day_s:
        starts = [
            record["metadata"]["request_start_ns"] / 1e9
            for record in records
            if (record.get("metadata") or {}).get("request_start_ns")
        ]
        day_start = min(starts) if starts else began
        end = day_start + day_s + float(os.environ["GRACE_S"])
        summary["client"] = window_summary(records, day_start, end)
        summary["start_skew_s"] = day_start - float(os.environ["START_AT"])
        hourly = {
            "client": hourly_summary(records, day_start, day_s / HOURS),
            "router": router_deltas(snapshots[: HOURS + 1]),
        }
        if len(snapshots) >= 2:
            summary["router"] = router_deltas([snapshots[0], snapshots[-1]])[0]
        print("BENCH_HOURLY " + json.dumps(hourly, separators=(",", ":")), flush=True)
    else:
        window_start = float(os.environ["START_AT"]) + float(
            os.environ["WINDOW_OFFSET_S"]
        )
        summary["client"] = window_summary(
            records, window_start, window_start + float(os.environ["MEASURE_S"])
        )
    print("BENCH_SUMMARY " + json.dumps(summary, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
