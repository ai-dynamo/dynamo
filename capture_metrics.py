#!/usr/bin/env python3
"""Background metrics capture for TRT-LLM disagg workers.

Runs inside the compute node container, polls /metrics and /perf_metrics
from worker ports every N seconds, writes JSONL.

Usage:
    python capture_metrics.py --endpoints localhost:8001,localhost:8003 \
        --output-dir /path/to/metrics/ --interval 2
"""

import argparse
import json
import os
import signal
import time
from datetime import datetime

try:
    import urllib.error
    import urllib.request
except ImportError:
    pass


def fetch_json(url, timeout=5):
    """Fetch JSON from URL, return None on failure."""
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def fetch_text(url, timeout=5):
    """Fetch text from URL, return None on failure."""
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode()
    except Exception:
        return None


def parse_prometheus(text):
    """Parse prometheus text format into dict of metric_name -> value."""
    if not text:
        return {}
    metrics = {}
    for line in text.strip().split("\n"):
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                metrics[parts[0]] = float(parts[1])
            except ValueError:
                metrics[parts[0]] = parts[1]
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoints", required=True, help="Comma-separated host:port list"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for JSONL files"
    )
    parser.add_argument(
        "--interval", type=float, default=2.0, help="Poll interval seconds"
    )
    parser.add_argument(
        "--labels",
        default="",
        help="Comma-separated labels for each endpoint (e.g. ctx,gen)",
    )
    args = parser.parse_args()

    endpoints = args.endpoints.split(",")
    labels = (
        args.labels.split(",")
        if args.labels
        else [f"worker{i}" for i in range(len(endpoints))]
    )
    os.makedirs(args.output_dir, exist_ok=True)

    # Open output files
    files = {}
    for i, (ep, label) in enumerate(zip(endpoints, labels)):
        fpath = os.path.join(args.output_dir, f"{label}_metrics.jsonl")
        files[ep] = open(fpath, "a")
        print(f"Capturing {ep} -> {fpath}")

    running = True

    def stop(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    print(f"Metrics capture started, interval={args.interval}s, endpoints={endpoints}")
    poll_count = 0

    while running:
        ts = time.time()
        ts_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

        for ep, label in zip(endpoints, labels):
            # Try /metrics (prometheus format)
            prom_text = fetch_text(f"http://{ep}/metrics", timeout=3)
            prom_data = parse_prometheus(prom_text) if prom_text else {}

            # Try /perf_metrics (JSON)
            perf_data = fetch_json(f"http://{ep}/perf_metrics", timeout=3)

            # Try /health for basic status
            health = fetch_json(f"http://{ep}/health", timeout=2)

            entry = {
                "ts": ts,
                "ts_str": ts_str,
                "worker": label,
                "endpoint": ep,
                "prometheus": prom_data,
                "perf_metrics": perf_data,
                "health": health,
            }

            try:
                files[ep].write(json.dumps(entry) + "\n")
                files[ep].flush()
            except Exception:
                pass

        poll_count += 1
        if poll_count % 30 == 0:  # Log every 60s at 2s interval
            sample_keys = list(prom_data.keys())[:5] if prom_data else ["empty"]
            print(
                f"[{ts_str}] Poll #{poll_count}, sample prometheus keys: {sample_keys}"
            )

        time.sleep(args.interval)

    # Close files
    for f in files.values():
        f.close()
    print(f"Metrics capture stopped after {poll_count} polls")


if __name__ == "__main__":
    main()
