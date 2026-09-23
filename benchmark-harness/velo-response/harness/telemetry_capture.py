#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture one-second process, thread, network, TCP, and Prometheus snapshots."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import signal
import socket
import time
import urllib.request
from pathlib import Path
from typing import Any


STOP = False


def stop(_signum: int, _frame: object) -> None:
    global STOP
    STOP = True


def timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def cpu_allowed(pid: int) -> str:
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("Cpus_allowed_list:"):
            return line.split(":", 1)[1].strip()
    return ""


def process_record(path: Path) -> dict[str, Any] | None:
    pid = int(path.name)
    try:
        stat = (path / "stat").read_text()
        comm, tail = stat.rsplit(") ", 1)
        fields = tail.split()
        return {
            "pid": pid,
            "ppid": int(fields[1]),
            "pgid": os.getpgid(pid),
            "sid": os.getsid(pid),
            "comm": comm.split("(", 1)[1],
            "cmdline": (path / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                errors="replace"
            ).strip(),
            "user_ticks": int(fields[11]),
            "system_ticks": int(fields[12]),
            "rss_pages": int(fields[21]),
            "thread_count": int(fields[17]),
            "cpus_allowed_list": cpu_allowed(pid),
        }
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, IndexError):
        return None


def thread_records(pid: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in Path(f"/proc/{pid}/task").glob("[0-9]*"):
        try:
            stat = (path / "stat").read_text()
            comm, tail = stat.rsplit(") ", 1)
            fields = tail.split()
            schedstat = (path / "schedstat").read_text().split()
            try:
                wchan = (path / "wchan").read_text().strip()
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                wchan = None
            records.append(
                {
                    "tid": int(path.name),
                    "comm": comm.split("(", 1)[1],
                    "state": fields[0],
                    "user_ticks": int(fields[11]),
                    "system_ticks": int(fields[12]),
                    "processor": int(fields[36]),
                    "cpus_allowed_list": cpu_allowed(int(path.name)),
                    "sched_runtime_ns": int(schedstat[0]),
                    "sched_runqueue_wait_ns": int(schedstat[1]),
                    "sched_timeslices": int(schedstat[2]),
                    "wchan": wchan,
                }
            )
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, IndexError):
            continue
    return records


def network_counters() -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for line in Path("/proc/net/dev").read_text().splitlines()[2:]:
        interface, raw = line.split(":", 1)
        values = [int(value) for value in raw.split()]
        result[interface.strip()] = {
            "rx_bytes": values[0],
            "rx_packets": values[1],
            "rx_errors": values[2],
            "rx_dropped": values[3],
            "tx_bytes": values[8],
            "tx_packets": values[9],
            "tx_errors": values[10],
            "tx_dropped": values[11],
        }
    return result


def tcp_counters() -> dict[str, int]:
    lines = Path("/proc/net/snmp").read_text().splitlines()
    for index in range(len(lines) - 1):
        if lines[index].startswith("Tcp:") and lines[index + 1].startswith("Tcp:"):
            names = lines[index].split()[1:]
            values = lines[index + 1].split()[1:]
            return {name: int(value) for name, value in zip(names, values)}
    return {}


def fetch_metrics(url: str | None) -> dict[str, Any] | None:
    if url is None:
        return None
    try:
        with urllib.request.urlopen(url, timeout=0.75) as response:
            body = response.read().decode(errors="replace")
        return {"ok": True, "text": body}
    except Exception as error:
        return {"ok": False, "error": f"{type(error).__name__}: {error}"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tracked-pgid", action="append", default=[])
    parser.add_argument("--metrics-url")
    args = parser.parse_args()

    tracked: dict[str, int] = {}
    for item in args.tracked_pgid:
        name, raw = item.split(":", 1)
        tracked[name] = int(raw)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    next_sample = time.monotonic()
    with args.output.open("w", encoding="utf-8") as handle:
        while not STOP:
            processes = [
                record
                for path in Path("/proc").glob("[0-9]*")
                if (record := process_record(path)) is not None
            ]
            tracked_processes: dict[str, list[dict[str, Any]]] = {}
            tracked_threads: dict[str, list[dict[str, Any]]] = {}
            for name, pgid in tracked.items():
                members = [record for record in processes if record["pgid"] == pgid]
                tracked_processes[name] = members
                tracked_threads[name] = [
                    {"pid": record["pid"], "threads": thread_records(record["pid"])}
                    for record in members
                ]
            sample = {
                "timestamp": timestamp(),
                "monotonic_seconds": time.monotonic(),
                "host": socket.gethostname(),
                "role": args.role,
                "clock_ticks_per_second": os.sysconf("SC_CLK_TCK"),
                "page_size": os.sysconf("SC_PAGE_SIZE"),
                "loadavg": os.getloadavg(),
                "processes": processes,
                "tracked_processes": tracked_processes,
                "tracked_threads": tracked_threads,
                "network": network_counters(),
                "tcp": tcp_counters(),
                "metrics": fetch_metrics(args.metrics_url),
            }
            handle.write(json.dumps(sample, sort_keys=True) + "\n")
            handle.flush()
            next_sample += 1.0
            time.sleep(max(0.0, next_sample - time.monotonic()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
