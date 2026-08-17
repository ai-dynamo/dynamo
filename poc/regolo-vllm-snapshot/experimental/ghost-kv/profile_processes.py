#!/usr/bin/env python3
"""Run one command while aggregating Linux per-thread restore activity."""

import argparse
import json
import os
import pathlib
import subprocess
import threading
import time


HZ = os.sysconf("SC_CLK_TCK")
PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")


def read_thread(pid, tid):
    root = pathlib.Path("/proc") / str(pid)
    fields = (root / "task" / str(tid) / "stat").read_text().split()
    status = (root / "task" / str(tid) / "status").read_text().splitlines()
    switches = {
        line.split(":", 1)[0]: int(line.split()[1])
        for line in status
        if line.startswith(("voluntary_ctxt_switches:", "nonvoluntary_ctxt_switches:"))
    }
    schedstat = [int(value) for value in (root / "task" / str(tid) / "schedstat").read_text().split()]
    return {
        "pid": pid,
        "tid": tid,
        "start": int(fields[21]),
        "comm": fields[1].removeprefix("(").removesuffix(")"),
        "cpu_ticks": int(fields[13]) + int(fields[14]),
        "minflt": int(fields[9]),
        "majflt": int(fields[11]),
        "rss_bytes": int(fields[23]) * PAGE_SIZE,
        "run_ns": schedstat[0],
        "wait_ns": schedstat[1],
        "voluntary": switches.get("voluntary_ctxt_switches", 0),
        "involuntary": switches.get("nonvoluntary_ctxt_switches", 0),
    }


def snapshot():
    result = {}
    for proc in pathlib.Path("/proc").glob("[0-9]*"):
        try:
            pid = int(proc.name)
            command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            for task in (proc / "task").glob("[0-9]*"):
                row = read_thread(pid, int(task.name))
                row["command"] = command
                result[(row["tid"], row["start"])] = row
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--interval", default=0.05, type=float)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command or args.command[0] != "--":
        parser.error("command must follow --")

    aggregates = {}
    peaks = {}
    samples = 0
    stopped = threading.Event()
    previous = snapshot()

    def collect():
        nonlocal previous, samples
        while not stopped.wait(args.interval):
            current = snapshot()
            samples += 1
            for key, row in current.items():
                old = previous.get(key)
                if old is None:
                    continue
                delta = {field: row[field] - old[field] for field in (
                    "cpu_ticks", "minflt", "majflt", "run_ns", "wait_ns", "voluntary", "involuntary"
                )}
                if any(value < 0 for value in delta.values()):
                    continue
                entry = aggregates.setdefault(key, {
                    "pid": row["pid"], "tid": row["tid"], "comm": row["comm"],
                    "command": row["command"], **{field: 0 for field in delta},
                })
                for field, value in delta.items():
                    entry[field] += value
                peaks[key] = max(peaks.get(key, 0), row["rss_bytes"])
            previous = current

    thread = threading.Thread(target=collect, daemon=True)
    started = time.monotonic()
    thread.start()
    completed = subprocess.run(args.command[1:], capture_output=True, text=True)
    stopped.set()
    thread.join()
    duration = time.monotonic() - started
    rows = []
    for key, row in aggregates.items():
        row["cpu_s"] = row.pop("cpu_ticks") / HZ
        row["rss_peak_bytes"] = peaks.get(key, 0)
        rows.append(row)
    rows.sort(key=lambda row: (row["cpu_s"], row["minflt"], row["wait_ns"]), reverse=True)
    result = {
        "command": args.command[1:], "returncode": completed.returncode,
        "duration_s": duration, "interval_s": args.interval, "samples": samples,
        "stdout": completed.stdout, "stderr": completed.stderr, "threads": rows[:100],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if completed.returncode:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
