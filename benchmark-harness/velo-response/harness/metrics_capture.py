#!/usr/bin/env python3
"""Capture a Prometheus endpoint once per second without process-table scanning."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import signal
import time
import urllib.request
from pathlib import Path
from typing import Any


STOP = False


def stop(_signum: int, _frame: Any) -> None:
    global STOP
    STOP = True


def fetch(url: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=0.75) as response:
            return {
                "ok": True,
                "text": response.read().decode(errors="replace"),
            }
    except Exception as error:
        return {"ok": False, "error": f"{type(error).__name__}: {error}"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic()
    with args.output.open("w", encoding="utf-8") as handle:
        while not STOP:
            sample = {
                "timestamp": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
                "monotonic_seconds": time.monotonic(),
                "role": args.role,
                "metrics": fetch(args.url),
            }
            handle.write(json.dumps(sample, sort_keys=True) + "\n")
            handle.flush()
            deadline += 1.0
            time.sleep(max(0.0, deadline - time.monotonic()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
