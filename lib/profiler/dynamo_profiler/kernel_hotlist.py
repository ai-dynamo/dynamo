# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kernel hotlist analyzer.

Ranks CUDA kernels by total time and attributes each to NVTX stages.
Detects TP imbalance via coefficient of variation across shards.
"""

from __future__ import annotations

import json
import logging
import statistics
import sys
from collections import defaultdict
from typing import Optional

from . import proto_reader

log = logging.getLogger("kernel_hotlist")


def analyze(merged_trace: str, top_n: int = 20) -> dict:
    log.info("Loading %s", merged_trace)
    tracks: dict[int, proto_reader.TrackInfo] = {}
    open_slices: dict[int, list] = defaultdict(list)
    by_kernel = defaultdict(lambda: {
        "total_ns": 0, "count": 0,
        "stages": defaultdict(int),
        "by_shard": defaultdict(lambda: {"total_ns": 0, "count": 0}),
        "min_dur_ns": float("inf"), "max_dur_ns": 0,
    })

    n_kernels = 0
    for raw in proto_reader.iter_packets(merged_trace):
        event, track = proto_reader.parse_packet(raw)
        if track is not None:
            tracks[track.uuid] = track
        elif event is not None:
            if event.event_type == 1:
                open_slices[event.track_uuid].append(event)
            elif event.event_type == 2:
                if open_slices[event.track_uuid]:
                    begin = open_slices[event.track_uuid].pop()
                    track_info = tracks.get(event.track_uuid)
                    if track_info and " Compute" in track_info.name:
                        duration_ns = event.timestamp_ns - begin.timestamp_ns
                        if duration_ns <= 0:
                            continue
                        kname = begin.name
                        stage = begin.annotations.get("stage", "<no_stage>")
                        process_name = ""
                        if track_info.parent_uuid and track_info.parent_uuid in tracks:
                            process_name = tracks[track_info.parent_uuid].process_name
                        shard_id = f"{process_name}/{track_info.name}"

                        agg = by_kernel[kname]
                        agg["total_ns"] += duration_ns
                        agg["count"] += 1
                        agg["stages"][stage] += duration_ns
                        agg["by_shard"][shard_id]["total_ns"] += duration_ns
                        agg["by_shard"][shard_id]["count"] += 1
                        agg["min_dur_ns"] = min(agg["min_dur_ns"], duration_ns)
                        agg["max_dur_ns"] = max(agg["max_dur_ns"], duration_ns)
                        n_kernels += 1

    log.info("Aggregated %d kernel invocations across %d unique kernels",
             n_kernels, len(by_kernel))

    sorted_kernels = sorted(by_kernel.items(), key=lambda x: -x[1]["total_ns"])
    total_kernel_time_ns = sum(v["total_ns"] for _, v in sorted_kernels)

    out_kernels = []
    for kname, agg in sorted_kernels[:top_n]:
        pct = agg["total_ns"] / total_kernel_time_ns * 100 if total_kernel_time_ns else 0

        shards = list(agg["by_shard"].values())
        cv_pct = None
        if len(shards) > 1:
            shard_avgs = [s["total_ns"] / s["count"] for s in shards if s["count"] > 0]
            if len(shard_avgs) > 1:
                mean = statistics.mean(shard_avgs)
                if mean > 0:
                    cv_pct = statistics.stdev(shard_avgs) / mean * 100

        out_kernels.append({
            "kernel": kname,
            "total_ms": agg["total_ns"] / 1e6,
            "count": agg["count"],
            "avg_us": agg["total_ns"] / agg["count"] / 1e3 if agg["count"] else 0,
            "min_us": agg["min_dur_ns"] / 1e3 if agg["min_dur_ns"] != float("inf") else 0,
            "max_us": agg["max_dur_ns"] / 1e3,
            "pct_of_total": pct,
            "per_stage_ms": {
                s: ns / 1e6 for s, ns in sorted(agg["stages"].items(), key=lambda x: -x[1])[:5]
            },
            "by_shard_ms": {k: v["total_ns"] / 1e6 for k, v in agg["by_shard"].items()},
            "shard_imbalance_cv_pct": cv_pct,
        })

    return {
        "top_kernels": out_kernels,
        "total_kernel_time_ms": total_kernel_time_ns / 1e6,
        "total_kernel_invocations": n_kernels,
    }


def diff_with_baseline(current: dict, baseline: dict) -> list[dict]:
    base_by_name = {k["kernel"]: k for k in baseline["top_kernels"]}
    cur_by_name = {k["kernel"]: k for k in current["top_kernels"]}

    diffs = []
    for name in set(base_by_name) | set(cur_by_name):
        cur = cur_by_name.get(name, {"total_ms": 0, "count": 0})
        base = base_by_name.get(name, {"total_ms": 0, "count": 0})
        delta_ms = cur["total_ms"] - base["total_ms"]
        if abs(delta_ms) < 10:
            continue
        diffs.append({
            "kernel": name,
            "baseline_ms": base["total_ms"],
            "current_ms": cur["total_ms"],
            "delta_ms": delta_ms,
            "pct_change": (delta_ms / base["total_ms"] * 100 if base["total_ms"] else None),
        })
    diffs.sort(key=lambda x: -abs(x["delta_ms"]))
    return diffs


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-trace", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--baseline")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    result = analyze(args.merged_trace, top_n=args.top_n)
    if args.baseline:
        with open(args.baseline) as f:
            result["diff_vs_baseline"] = diff_with_baseline(result, json.load(f))
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    sys.exit(main() or 0)
