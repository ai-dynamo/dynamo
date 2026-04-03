#!/usr/bin/env python3
"""
Collect RSS for a process.

Usage:
  python3 collect_rss.py --pid <PID>
  python3 collect_rss.py --pid <PID> --output rss.csv
"""

import argparse
import csv
import signal
import time
from pathlib import Path


def get_rss(pid: int):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024  # KB to bytes
    except (OSError, ValueError, IndexError):
        return None
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("rss.csv"))
    parser.add_argument("--interval", type=float, default=0.1)
    args = parser.parse_args()

    running = True

    def stop(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    start = time.time()

    with args.output.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["elapsed_s", "rss_bytes"])

        while running:
            rss = get_rss(args.pid)
            if rss is None:
                print(f"Process {args.pid} gone")
                break

            elapsed = time.time() - start
            w.writerow([f"{elapsed:.3f}", rss])
            f.flush()

            time.sleep(args.interval)

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
