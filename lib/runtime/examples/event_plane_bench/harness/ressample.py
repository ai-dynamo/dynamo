#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DIS-2172 event-plane RESOURCE-COST sampler (CHANGE #3).

A single lightweight per-node snapshot of the event-plane's resource footprint,
taken mid measurement-window so we can quantify the cost difference between the
brokerless ZMQ direct mesh (O(p×s) TCP connections, no broker) and NATS
(O(p+s) connections to a central broker that itself burns CPU/RSS).

Runs on the NODE HOST (NOT inside the pyxis container): enroot/pyxis run their
processes in the host PID namespace and share the host net namespace, so a
host-level `ss`/`/proc` scan sees every mocker / subscriber / nats-server on the
node regardless of container boundaries. Emits one small JSON snapshot to
--out (appended-as-a-line is the caller's job; we just write one object).

Captured PER NODE:
  - established_tcp : count of `ss -tn state established` rows (the headline
    O(p×s) vs O(p+s) connection-count difference).
  - per-process {cpu_pct, rss_kb, fds, nproc} for the key process classes:
      mocker            (python -m dynamo.mocker)
      sub               (event_plane_bench_sub)
      nats              (nats-server)
      frontend          (python -m dynamo.frontend)
      zmq_broker        (zmq_broker.py)
    nproc = number of matched processes (e.g. p mockers can land on one node);
    cpu_pct / rss_kb / fds are SUMMED across them. nats RSS/CPU during NATS
    cells vs ~idle during ZMQ cells is the broker-cost quantification.

Cheap by design: one `ss` call + two quick /proc passes (≈CPU sample interval)
per invocation. Must NOT perturb the measurement, so keep the interval short.
"""
import argparse
import glob
import json
import os
import subprocess
import time

# Process classes -> substring matched against /proc/<pid>/cmdline (NUL-joined).
# Order matters only for reporting; matching is independent per class.
CLASSES = {
    "mocker": ["dynamo.mocker"],
    "sub": ["event_plane_bench_sub"],
    "nats": ["nats-server"],
    "frontend": ["dynamo.frontend"],
    "zmq_broker": ["zmq_broker.py"],
    "loadgen": ["loadgen.py"],
}

CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


def _established_tcp():
    """Count of established TCP connections on this node (host net ns)."""
    for argv in (["ss", "-tn", "state", "established"], ["ss", "-tn"]):
        try:
            out = subprocess.check_output(argv, text=True, stderr=subprocess.DEVNULL)
        except Exception:
            continue
        lines = [ln for ln in out.splitlines() if ln.strip()]
        # `ss -tn` (no state filter) prints a header row; `state established`
        # does not. Drop a leading header if present.
        if lines and lines[0].split()[:1] == ["State"]:
            lines = lines[1:]
        if argv[-1] != "established":
            lines = [ln for ln in lines if "ESTAB" in ln]
        return len(lines)
    return None


def _proc_cmdline(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\x00", b" ").decode("utf-8", "replace")
    except Exception:
        return ""


def _proc_stat(pid):
    """Return (utime+stime ticks, rss_kb) or None."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            parts = f.read().split()
        # field 14 utime, 15 stime (1-indexed); comm in parens may contain
        # spaces, so anchor after the closing ')'.
        rparen = " ".join(parts).rfind(")")
        # Simpler: split on ')' once; fields after are space-clean.
        with open(f"/proc/{pid}/stat") as f:
            raw = f.read()
        after = raw[raw.rfind(")") + 1:].split()
        # after[0]=state ... utime=after[11], stime=after[12] (0-indexed here)
        utime = int(after[11])
        stime = int(after[12])
    except Exception:
        return None
    rss_kb = 0
    try:
        with open(f"/proc/{pid}/status") as f:
            for ln in f:
                if ln.startswith("VmRSS:"):
                    rss_kb = int(ln.split()[1])
                    break
    except Exception:
        pass
    return (utime + stime, rss_kb)


def _proc_fds(pid):
    try:
        return len(os.listdir(f"/proc/{pid}/fd"))
    except Exception:
        return 0


def _classify(cmd):
    for cls, needles in CLASSES.items():
        if any(n in cmd for n in needles):
            return cls
    return None


def sample(cpu_interval=0.3):
    pids = []
    for d in glob.glob("/proc/[0-9]*"):
        pid = d.rsplit("/", 1)[-1]
        cmd = _proc_cmdline(pid)
        cls = _classify(cmd) if cmd else None
        if cls:
            pids.append((pid, cls))

    # First CPU/RSS pass.
    t0 = time.time()
    first = {}
    for pid, cls in pids:
        st = _proc_stat(pid)
        if st is not None:
            first[pid] = st
    time.sleep(cpu_interval)
    elapsed = time.time() - t0

    agg = {cls: {"nproc": 0, "cpu_pct": 0.0, "rss_kb": 0, "fds": 0}
           for cls in CLASSES}
    for pid, cls in pids:
        st1 = _proc_stat(pid)
        st0 = first.get(pid)
        if st1 is None or st0 is None:
            continue
        ticks = max(0, st1[0] - st0[0])
        cpu_pct = 100.0 * (ticks / CLK_TCK) / elapsed if elapsed > 0 else 0.0
        a = agg[cls]
        a["nproc"] += 1
        a["cpu_pct"] += round(cpu_pct, 1)
        a["rss_kb"] += st1[1]
        a["fds"] += _proc_fds(pid)
    for cls in agg:
        agg[cls]["cpu_pct"] = round(agg[cls]["cpu_pct"], 1)

    return {
        "established_tcp": _established_tcp(),
        "procs": agg,
        "cpu_interval_s": round(elapsed, 3),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="write one JSON snapshot here")
    ap.add_argument("--node", default=os.environ.get("SLURMD_NODENAME", ""),
                    help="node label for this snapshot")
    ap.add_argument("--transport", default="", help="cell transport (nats/zmq/...)")
    ap.add_argument("--cell", default="", help="cell idx")
    ap.add_argument("--snap", default="", help="snapshot label (e.g. mid)")
    ap.add_argument("--cpu-interval", type=float, default=0.3,
                    help="CPU sampling interval (s); keep short to stay cheap")
    a = ap.parse_args()

    snap = sample(a.cpu_interval)
    snap.update(node=a.node or "?", transport=a.transport, cell=a.cell,
                snap=a.snap, ts=time.time())
    with open(a.out, "w") as f:
        json.dump(snap, f)
    # Also echo to stdout so the orchestrator's log captures it for debugging.
    print(json.dumps(snap))


if __name__ == "__main__":
    main()
