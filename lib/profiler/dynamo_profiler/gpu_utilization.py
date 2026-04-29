# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU utilization gap analyzer.

Reads merged Perfetto traces and computes per-device utilization,
time-on-GPU breakdown by NVTX stage, and idle gap attribution.
"""

from __future__ import annotations

import json
import logging
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from . import proto_reader

log = logging.getLogger("gpu_util")


@dataclass
class IdleGap:
    start_ns: int
    duration_ns: int
    likely_cause: Optional[str] = None
    likely_cause_track: Optional[str] = None
    likely_cause_overlap_ns: int = 0


@dataclass
class GpuTrackReport:
    process_name: str
    track_name: str
    capture_start_ns: int
    capture_end_ns: int
    capture_duration_ns: int
    busy_ns: int
    idle_ns: int
    utilization: float
    busy_by_stage: dict[str, int] = field(default_factory=dict)
    biggest_idle_gaps: list[IdleGap] = field(default_factory=list)
    suspicious_gaps_ns: int = 0


def analyze(merged_trace: str | list[str]) -> dict:
    if isinstance(merged_trace, str):
        merged_trace = [merged_trace]

    tracks: dict[int, proto_reader.TrackInfo] = {}
    slices_by_track: dict[int, list] = defaultdict(list)
    open_slices: dict[int, list] = defaultdict(list)
    interned_names: dict = {}

    overall_min_ts = None
    overall_max_ts = None

    for trace_file in merged_trace:
        log.info("Loading %s", trace_file)
        open_slices.clear()
        for raw in proto_reader.iter_packets(trace_file):
            event, track = proto_reader.parse_packet(raw, interned_names)
            if track is not None:
                tracks[track.uuid] = track
            elif event is not None:
                if overall_min_ts is None or event.timestamp_ns < overall_min_ts:
                    overall_min_ts = event.timestamp_ns
                if overall_max_ts is None or event.timestamp_ns > overall_max_ts:
                    overall_max_ts = event.timestamp_ns
                if event.event_type == 1:
                    open_slices[event.track_uuid].append(event)
                elif event.event_type == 2:
                    if open_slices[event.track_uuid]:
                        begin = open_slices[event.track_uuid].pop()
                        slices_by_track[event.track_uuid].append({
                            "start_ns": begin.timestamp_ns,
                            "end_ns": event.timestamp_ns,
                            "name": begin.name,
                            "annotations": begin.annotations,
                        })

    for t in tracks.values():
        if t.parent_uuid and t.parent_uuid in tracks:
            t.process_name = tracks[t.parent_uuid].process_name or tracks[t.parent_uuid].name

    gpu_tracks = {
        uuid: t for uuid, t in tracks.items()
        if " Compute" in t.name and t.name.startswith("GPU")
    }
    log.info("GPU compute tracks: %d", len(gpu_tracks))

    reports = []
    for uuid, gpu_track in gpu_tracks.items():
        gpu_slices = slices_by_track.get(uuid, [])
        if not gpu_slices:
            continue
        gpu_intervals = proto_reader.merge_intervals(
            [(s["start_ns"], s["end_ns"]) for s in gpu_slices]
        )
        capture_start = overall_min_ts
        capture_end = overall_max_ts
        idle_intervals = proto_reader.invert_intervals(gpu_intervals, capture_start, capture_end)
        busy_ns = sum(e - s for s, e in gpu_intervals)
        idle_ns = sum(e - s for s, e in idle_intervals)
        capture_duration = capture_end - capture_start

        busy_by_stage: dict[str, int] = defaultdict(int)
        for s in gpu_slices:
            stage = s["annotations"].get("stage", "<no_stage>")
            busy_by_stage[stage] += s["end_ns"] - s["start_ns"]

        same_process_tracks = [
            other_uuid for other_uuid, other_t in tracks.items()
            if other_t.process_name == gpu_track.process_name
            and other_uuid != uuid
            and " Compute" not in other_t.name
            and "Memcpy" not in other_t.name
        ]
        cpu_slices_for_attribution = []
        for ouid in same_process_tracks:
            for sl in slices_by_track.get(ouid, []):
                cpu_slices_for_attribution.append((
                    sl["start_ns"], sl["end_ns"], sl["name"],
                    tracks[ouid].name, sl["annotations"],
                ))
        cpu_slices_for_attribution.sort()

        big_gaps = sorted(idle_intervals, key=lambda x: -(x[1] - x[0]))[:10]
        idle_gaps = []
        suspicious_gaps_ns = 0
        for start, end in big_gaps:
            if end - start < 1_000_000:
                continue
            gap = IdleGap(start_ns=start, duration_ns=end - start)
            best_overlap_ns = 0
            best_slice = None
            for ssn, sen, sname, tname, _ in cpu_slices_for_attribution:
                overlap = max(0, min(sen, end) - max(ssn, start))
                if overlap > best_overlap_ns:
                    best_overlap_ns = overlap
                    best_slice = (sname, tname)
            if best_slice and best_overlap_ns > (end - start) * 0.5:
                gap.likely_cause = best_slice[0]
                gap.likely_cause_track = best_slice[1]
                gap.likely_cause_overlap_ns = best_overlap_ns
            else:
                suspicious_gaps_ns += end - start
            idle_gaps.append(gap)

        reports.append(GpuTrackReport(
            process_name=gpu_track.process_name,
            track_name=gpu_track.name,
            capture_start_ns=capture_start,
            capture_end_ns=capture_end,
            capture_duration_ns=capture_duration,
            busy_ns=busy_ns,
            idle_ns=idle_ns,
            utilization=busy_ns / capture_duration if capture_duration else 0,
            busy_by_stage=dict(busy_by_stage),
            biggest_idle_gaps=idle_gaps,
            suspicious_gaps_ns=suspicious_gaps_ns,
        ))

    return {
        "gpu_track_reports": [_track_report_to_dict(r) for r in reports],
        "summary": {
            "n_gpu_tracks": len(reports),
            "avg_utilization": (
                statistics.mean([r.utilization for r in reports]) if reports else 0
            ),
            "min_utilization": min((r.utilization for r in reports), default=0),
            "min_utilization_track": min(
                reports, key=lambda r: r.utilization
            ).track_name if reports else None,
        },
    }


def _track_report_to_dict(r: GpuTrackReport) -> dict:
    return {
        "process_name": r.process_name,
        "track_name": r.track_name,
        "capture_duration_ms": r.capture_duration_ns / 1e6,
        "busy_ms": r.busy_ns / 1e6,
        "idle_ms": r.idle_ns / 1e6,
        "utilization": r.utilization,
        "busy_by_stage_ms": {k: v / 1e6 for k, v in r.busy_by_stage.items()},
        "biggest_idle_gaps": [
            {
                "duration_ms": g.duration_ns / 1e6,
                "likely_cause": g.likely_cause,
                "likely_cause_track": g.likely_cause_track,
                "overlap_pct": (
                    g.likely_cause_overlap_ns / g.duration_ns * 100
                    if g.duration_ns else 0
                ),
            }
            for g in r.biggest_idle_gaps
        ],
        "suspicious_gaps_ms": r.suspicious_gaps_ns / 1e6,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-trace", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--print", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    result = analyze(args.merged_trace)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    if getattr(args, "print"):
        for r in result["gpu_track_reports"]:
            print(f"{r['process_name']} / {r['track_name']}: {r['utilization']*100:.1f}%")


if __name__ == "__main__":
    sys.exit(main() or 0)
