#!/usr/bin/env python3
"""
capture_metrics.py — poll Prometheus /metrics endpoints on a fixed interval and
append one JSONL record per scrape to <label>_metrics.jsonl in --output-dir.

Designed to run as a background sidecar from the dynamo.trtllm benchx scripts.
Stdlib-only (urllib, threading, json, signal, argparse) so it works on the
sbatch host without any venv or container.

Per-line record format:
    {"ts": 1714944000.123, "endpoint": "host:port", "metrics": {"name{labels}": value, ...}}
On scrape failure, a record with "error" is written instead of "metrics" so the
stream stays continuous and you can see when the worker was unreachable.
"""
import argparse
import json
import os
import re
import signal
import sys
import threading
import time
import urllib.error
import urllib.request

# Match a Prometheus sample line: name{labels} value [timestamp]
# Captures: name, optional {labels} (kept literal incl. braces), value
_SAMPLE_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)(?:\s+\d+)?\s*$"
)


def parse_metrics(text):
    """Parse a Prometheus text exposition payload into {name{labels}: float}."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _SAMPLE_RE.match(line)
        if not m:
            continue
        name, labels, value = m.group(1), m.group(2) or "", m.group(3)
        try:
            v = float(value)
        except ValueError:
            continue
        out[f"{name}{labels}"] = v
    return out


def scrape_loop(endpoint, label, output_dir, interval, stop_event):
    url = f"http://{endpoint}/metrics"
    out_path = os.path.join(output_dir, f"{label}_metrics.jsonl")
    print(
        f"[{label}] -> {out_path} (poll {url} every {interval}s)",
        file=sys.stderr,
        flush=True,
    )

    timeout = max(1.0, float(interval))
    with open(out_path, "a", buffering=1) as f:
        next_due = time.monotonic()
        while not stop_event.is_set():
            ts = time.time()
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "capture_metrics/1"}
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    body = resp.read().decode("utf-8", errors="replace")
                record = {
                    "ts": ts,
                    "endpoint": endpoint,
                    "metrics": parse_metrics(body),
                }
            except (urllib.error.URLError, OSError, TimeoutError) as e:
                record = {"ts": ts, "endpoint": endpoint, "error": str(e)}
                print(f"[{label}] scrape error: {e}", file=sys.stderr, flush=True)

            f.write(json.dumps(record, separators=(",", ":")) + "\n")

            next_due += interval
            sleep_for = next_due - time.monotonic()
            if sleep_for <= 0:
                # Fell behind (slow scrape) — reset clock instead of bursting.
                next_due = time.monotonic()
                continue
            stop_event.wait(timeout=sleep_for)


def main():
    p = argparse.ArgumentParser(
        description="Poll Prometheus /metrics endpoints to per-label JSONL files."
    )
    p.add_argument(
        "--endpoints",
        required=True,
        help="Comma-separated host:port list (e.g. 'h1:8081,h2:8082').",
    )
    p.add_argument(
        "--labels",
        required=True,
        help="Comma-separated label list (1:1 with --endpoints).",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory for <label>_metrics.jsonl files (created if missing).",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Seconds between scrapes (default 2.0).",
    )
    args = p.parse_args()

    endpoints = [e.strip() for e in args.endpoints.split(",") if e.strip()]
    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    if len(endpoints) != len(labels):
        print(
            f"ERROR: --endpoints has {len(endpoints)} entries but --labels has {len(labels)}",
            file=sys.stderr,
        )
        sys.exit(2)
    if not endpoints:
        print("ERROR: --endpoints is empty", file=sys.stderr)
        sys.exit(2)

    os.makedirs(args.output_dir, exist_ok=True)

    stop_event = threading.Event()

    def _sig(sig, _frame):
        print(
            f"capture_metrics: caught signal {sig}, stopping...",
            file=sys.stderr,
            flush=True,
        )
        stop_event.set()

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    threads = []
    for ep, lbl in zip(endpoints, labels):
        t = threading.Thread(
            target=scrape_loop,
            args=(ep, lbl, args.output_dir, args.interval, stop_event),
            name=f"scrape-{lbl}",
            daemon=False,
        )
        t.start()
        threads.append(t)

    print(
        f"capture_metrics: started {len(threads)} scrape threads, interval={args.interval}s",
        file=sys.stderr,
        flush=True,
    )

    for t in threads:
        t.join()

    print("capture_metrics: all threads stopped", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
