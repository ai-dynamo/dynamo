# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample a process's CPU time once per second, and summarize a run.

Usage:
  cpu_sampler.py PID OUT.jsonl
      Each line: {"t", "utime", "stime", "rss_kb", "threads", "groups": {name: [utime, stime, n]}}
      with times in clock ticks. Thread names lose trailing numbers ("ZMQbg/IO/3" -> "ZMQbg/IO").
  cpu_sampler.py summarize RUN_DIR
      Print the bench's measure-phase results and each process's CPU over that phase.
"""

import glob
import json
import os
import re
import sys
import time


def read_stat(path):
    with open(path) as f:
        data = f.read()
    close = data.rindex(")")
    comm = data[data.index("(") + 1 : close]
    fields = data[close + 2 :].split()
    return comm, int(fields[11]), int(fields[12])


def sample(pid, out_path):
    proc = f"/proc/{pid}"
    with open(out_path, "w") as out:
        next_tick = time.time()
        while os.path.exists(proc):
            try:
                now = time.time()
                _, utime, stime = read_stat(f"{proc}/stat")
                groups = {}
                tids = os.listdir(f"{proc}/task")
                for tid in tids:
                    try:
                        comm, tu, ts = read_stat(f"{proc}/task/{tid}/stat")
                    except (FileNotFoundError, ProcessLookupError):
                        continue
                    group = groups.setdefault(re.sub(r"[-/]?\d+$", "", comm), [0, 0, 0])
                    group[0] += tu
                    group[1] += ts
                    group[2] += 1
                rss_kb = 0
                with open(f"{proc}/status") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            rss_kb = int(line.split()[1])
                line = {
                    "t": now,
                    "utime": utime,
                    "stime": stime,
                    "rss_kb": rss_kb,
                    "threads": len(tids),
                    "groups": groups,
                }
                out.write(json.dumps(line, separators=(",", ":")) + "\n")
                out.flush()
            except (FileNotFoundError, ProcessLookupError):
                break
            next_tick += 1.0
            time.sleep(max(0.0, next_tick - time.time()))


def phase_window(run_dir):
    """Measure-phase start and end, from the phase markers the bench touches."""
    bench = os.path.join(run_dir, "bench")
    start = os.path.getmtime(os.path.join(bench, "phase.measure"))
    for name in ("phase.profile", "phase.done"):
        path = os.path.join(bench, name)
        if os.path.exists(path):
            return start, os.path.getmtime(path)
    raise FileNotFoundError("no phase marker after phase.measure")


def cpu_over(path, start, end):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    inside = [row for row in rows if start <= row["t"] <= end]
    if len(inside) < 2:
        return None
    first, last = inside[0], inside[-1]
    secs = last["t"] - first["t"]
    ticks = os.sysconf("SC_CLK_TCK")
    return {
        "cores": (last["utime"] + last["stime"] - first["utime"] - first["stime"])
        / ticks
        / secs,
        "sys_cores": (last["stime"] - first["stime"]) / ticks / secs,
        "threads": max(row["threads"] for row in inside),
        "rss_gib": max(row["rss_kb"] for row in inside) / 2**20,
    }


def summarize(run_dir):
    summary = json.load(open(os.path.join(run_dir, "bench", "summary.json")))
    measure = summary["phases"]["measure"]
    print(f"measure phase ({measure['secs']:.0f} s):")
    print(
        f"  requests/s {measure['requests_per_s']:.1f}"
        f"  hit ratio predicted {measure['predicted_hit_ratio']:.3f}"
        f" / engine {measure['engine_hit_ratio']:.3f}"
        f"  KV stored blocks/s {measure['kv_stored_blocks_per_s']:,.0f}"
        f"  error fraction {measure['error_fraction']:.1e}"
    )
    for name, endpoint in measure["endpoints"].items():
        errors = endpoint["errors"]
        if isinstance(errors, dict):
            errors = sum(errors.values())
        print(
            f"  {name}: {endpoint['calls_per_s']:.1f}/s"
            f"  p50 {endpoint['p50_ms']:.2f} ms  p99 {endpoint['p99_ms']:.2f} ms"
            f"  errors {errors}"
        )
    start, end = phase_window(run_dir)
    for path in sorted(glob.glob(os.path.join(run_dir, "*.cpu.jsonl"))):
        usage = cpu_over(path, start, end)
        if usage is None:
            continue
        name = os.path.basename(path).removesuffix(".cpu.jsonl")
        print(
            f"  {name}: {usage['cores']:.2f} cores ({usage['sys_cores']:.2f} sys)"
            f"  threads {usage['threads']}  RSS {usage['rss_gib']:.2f} GiB"
        )


if __name__ == "__main__":
    if sys.argv[1] == "summarize":
        summarize(sys.argv[2])
    else:
        sample(*sys.argv[1:3])
