#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scrape Prometheus ``/metrics`` of the frontend and the workers during a live run.

    scrape_metrics.py loop --target frontend=http://h:18000/metrics --target w0=http://h:19100/metrics \\
        --interval 10 --out series.jsonl
    scrape_metrics.py once --target ... --tag cell-start --out marks.jsonl

Each line is one scrape of one target: ``{"t_unix_ns", "tag", "target", "ok", "samples"}``, where
``samples`` maps ``name{labels}`` to its value for every ``vllm:*`` and ``dynamo_*`` series except
histogram buckets. ``loop`` runs until SIGTERM or SIGINT and flushes every line, so a killed run
keeps what it scraped. Standard library only: it runs from the AIPerf environment.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
import urllib.request
from pathlib import Path

PREFIXES = ("vllm:", "dynamo_")


def parse(text: str) -> dict[str, float]:
    samples: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#") or not line.startswith(PREFIXES):
            continue
        name, _, value = line.rpartition(" ")
        if "_bucket{" in name or name.endswith("_bucket"):
            continue
        try:
            samples[name] = float(value)
        except ValueError:
            continue
    return samples


def scrape(name: str, url: str, tag: str | None, timeout: float) -> dict:
    record: dict = {"t_unix_ns": time.time_ns(), "tag": tag, "target": name}
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            record["samples"] = parse(response.read().decode("utf-8", "replace"))
        record["ok"] = True
    # Recorded per scrape: a dead target must not stop the series.
    except Exception as exc:
        record["ok"] = False
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def scrape_all(targets: list[tuple[str, str]], tag: str | None, timeout: float):
    for name, url in targets:
        yield scrape(name, url, tag, timeout)


def parse_target(text: str) -> tuple[str, str]:
    name, sep, url = text.partition("=")
    if not sep or not name or not url.startswith("http"):
        raise argparse.ArgumentTypeError(
            f"--target needs name=http://..., got {text!r}"
        )
    return name, url


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("mode", choices=("loop", "once"))
    parser.add_argument("--target", type=parse_target, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    with args.out.open("a") as handle:
        while True:
            started = time.monotonic()
            for record in scrape_all(args.target, args.tag, args.timeout):
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            if args.mode == "once" or stop.wait(
                max(0.0, args.interval - (time.monotonic() - started))
            ):
                return 0


if __name__ == "__main__":
    sys.exit(main())
