#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Recompute throughput/latency from aiperf's raw profile_export.jsonl.

WHY: aiperf's finalizer can hang for minutes (or effectively deadlock) on large
runs while "processing records". Don't wait for profile_export_aiperf.json --
the raw per-request JSONL is written incrementally and has everything you need.
Kill the finalizer and run this instead.

Usage: python3 extract_throughput.py <artifact_dir_or_jsonl> [--conc N]

Each JSONL line: {"metadata": {...request_start_ns, request_end_ns,
benchmark_phase, was_cancelled...}, "metrics": {request_latency, time_to_first_token, ...}}
"""
import json
import os
import statistics as st
import sys


def find_jsonl(p):
    if p.endswith(".jsonl"):
        return p
    for root, _, files in os.walk(p):
        if "profile_export.jsonl" in files:
            return os.path.join(root, "profile_export.jsonl")
    sys.exit(f"no profile_export.jsonl under {p}")


def main():
    path = find_jsonl(sys.argv[1])
    conc = None
    if "--conc" in sys.argv:
        conc = int(sys.argv[sys.argv.index("--conc") + 1])
    S, E, lat, ttft = [], [], [], []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        m = r.get("metadata") or {}
        if m.get("benchmark_phase") != "profiling" or m.get("was_cancelled"):
            continue
        start = m.get("request_start_ns")
        end = m.get("request_end_ns")
        if start is None or end is None:
            continue
        S.append(start)
        E.append(end)
        mt = r.get("metrics") or {}
        request_latency = mt.get("request_latency") or {}
        time_to_first_token = mt.get("time_to_first_token") or {}
        if "value" in request_latency:
            lat.append(request_latency["value"])
        if "value" in time_to_first_token:
            ttft.append(time_to_first_token["value"])
    if not S:
        sys.exit("no completed profiling-phase records")
    wall = (max(E) - min(S)) / 1e9
    tput = len(S) / wall
    print(f"file: {path}")
    print(f"completed reqs : {len(S)}")
    print(f"wall (1st start->last end): {wall:.1f}s")
    print(f"THROUGHPUT     : {tput:.1f} req/s")
    if lat:
        print(
            f"request_latency: mean {st.mean(lat):.0f}ms  p50 {st.median(lat):.0f}ms  "
            f"p99 {sorted(lat)[int(0.99*len(lat))]:.0f}ms"
        )
    if ttft:
        print(
            f"TTFT           : p50 {st.median(ttft):.0f}ms  p99 {sorted(ttft)[int(0.99*len(ttft))]:.0f}ms"
        )
    if conc and lat:
        # Little's law sanity check (closed loop): tput ~= concurrency / latency
        print(
            f"Little's law   : {conc}/{st.mean(lat)/1000:.2f}s = {conc/(st.mean(lat)/1000):.0f} req/s "
            f"(vs measured {tput:.0f})"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main()
