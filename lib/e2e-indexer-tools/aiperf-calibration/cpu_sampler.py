#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sample CPU use of process trees (stdlib only, Linux /proc).

Usage: cpu_sampler.py --out FILE --interval 1 LABEL=PID [LABEL=PID ...]
Writes one JSON line per interval: {"t", "node_busy_cores", "<label>": cores,
"<label>.by_name": {name: cores}}. Stops when every root is gone or on SIGTERM.
Names come from comm, refined for Python processes by the AIPerf service in argv.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import time

HZ = os.sysconf("SC_CLK_TCK")


def children_map() -> dict[int, list[int]]:
    kids: dict[int, list[int]] = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat", "rb") as f:
                s = f.read()
        except OSError:
            continue
        rest = s[s.rfind(b")") + 2 :].split()
        kids.setdefault(int(rest[1]), []).append(int(d))
    return kids


def tree(root: int, kids: dict[int, list[int]]) -> list[int]:
    out, stack = [], [root]
    while stack:
        p = stack.pop()
        out.append(p)
        stack.extend(kids.get(p, []))
    return out


def cpu_ticks(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            s = f.read()
    except OSError:
        return None
    rest = s[s.rfind(b")") + 2 :].split()
    return int(rest[11]) + int(rest[12])


_names: dict[int, str] = {}


def name_of(pid: int) -> str:
    n = _names.get(pid)
    if n is not None:
        return n
    try:
        with open(f"/proc/{pid}/comm") as f:
            n = f.read().strip()
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            argv = f.read().split(b"\0")
    except OSError:
        return "?"
    joined = b" ".join(argv).decode(errors="replace")
    for key in (
        "worker",
        "record_processor",
        "dataset_manager",
        "timing_manager",
        "system_controller",
        "records_manager",
        "worker_manager",
    ):
        if key in joined:
            n = f"{n}:{key}"
            break
    _names[pid] = n
    return n


def node_busy() -> tuple[int, int]:
    with open("/proc/stat") as f:
        v = [int(x) for x in f.readline().split()[1:]]
    idle = v[3] + v[4]
    return sum(v) - idle, sum(v)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument("roots", nargs="+")
    a = p.parse_args()
    roots = {lab: int(pid) for lab, pid in (r.split("=") for r in a.roots)}
    stop = False

    def on_term(*_: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    last: dict[int, int] = {}
    kids0 = children_map()
    for root in roots.values():
        for pid in tree(root, kids0):
            cur = cpu_ticks(pid)
            if cur is not None:
                last[pid] = cur
    last_t = time.monotonic()
    last_busy = node_busy()
    with open(a.out, "w") as out:
        while not stop:
            time.sleep(a.interval)
            now = time.monotonic()
            dt = now - last_t
            kids = children_map()
            rec: dict[str, object] = {"t": round(time.time(), 3)}
            busy = node_busy()
            ncpu = os.cpu_count() or 1
            rec["node_busy_cores"] = round(
                (busy[0] - last_busy[0]) / max(1, busy[1] - last_busy[1]) * ncpu, 2
            )
            last_busy = busy
            alive = False
            for lab, root in roots.items():
                if not os.path.exists(f"/proc/{root}"):
                    continue
                alive = True
                total = 0.0
                by: dict[str, float] = {}
                for pid in tree(root, kids):
                    cur = cpu_ticks(pid)
                    if cur is None:
                        continue
                    # A pid first seen after the baseline pass started inside this interval.
                    d = max(0, cur - last.get(pid, 0))
                    last[pid] = cur
                    c = d / HZ / dt
                    total += c
                    nm = name_of(pid)
                    by[nm] = by.get(nm, 0.0) + c
                rec[lab] = round(total, 3)
                rec[lab + ".by_name"] = {
                    k: round(v, 3) for k, v in by.items() if v >= 0.005
                }
                rec[lab + ".nproc"] = len(by)
            last_t = now
            out.write(json.dumps(rec) + "\n")
            out.flush()
            if not alive:
                break


if __name__ == "__main__":
    main()
