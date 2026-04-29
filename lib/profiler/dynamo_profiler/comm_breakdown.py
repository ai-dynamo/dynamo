# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Communication breakdown analyzer.

Analyzes NCCL, NiXL, and NATS communication events from merged Perfetto
traces to compute per-category bandwidth, latency, and NATS propagation lag.
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

log = logging.getLogger("comm_breakdown")


@dataclass
class CommOpStats:
    op_name: str
    category: str
    count: int = 0
    total_duration_ns: int = 0
    total_bytes: int = 0
    durations_ns: list = field(default_factory=list)
    per_host: dict = field(default_factory=lambda: defaultdict(int))


@dataclass
class NatsPair:
    msg_id: str
    pub_host: str
    pub_ts: int
    sub_host: str
    sub_ts: int
    lag_ns: int


def classify_track(name: str) -> Optional[str]:
    lo = name.lower()
    if "nccl" in lo:
        return "nccl"
    if "nixl" in lo:
        return "nixl"
    if "nats" in lo:
        return "nats"
    return None


def analyze(merged_trace: str) -> dict:
    log.info("Loading %s", merged_trace)
    tracks: dict[int, proto_reader.TrackInfo] = {}
    open_slices: dict[int, list] = defaultdict(list)
    completed_slices: list = []
    instant_events: list = []

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
                    completed_slices.append({
                        "start_ns": begin.timestamp_ns,
                        "end_ns": event.timestamp_ns,
                        "name": begin.name,
                        "annotations": {**begin.annotations, **event.annotations},
                        "track_uuid": begin.track_uuid,
                    })
            elif event.event_type == 3:
                instant_events.append(event)

    for t in tracks.values():
        if t.parent_uuid and t.parent_uuid in tracks:
            t.process_name = tracks[t.parent_uuid].process_name or tracks[t.parent_uuid].name

    comm_tracks: dict[int, str] = {}
    for uuid, t in tracks.items():
        cat = classify_track(t.name)
        if cat:
            comm_tracks[uuid] = cat

    ops: dict[str, CommOpStats] = {}
    for sl in completed_slices:
        cat = comm_tracks.get(sl["track_uuid"])
        if cat is None:
            lo = sl["name"].lower()
            if "nccl" in lo:
                cat = "nccl"
            elif "nixl" in lo:
                cat = "nixl"
            elif "nats" in lo:
                cat = "nats"
            else:
                continue

        op_name = sl["name"]
        key = f"{cat}:{op_name}"
        if key not in ops:
            ops[key] = CommOpStats(op_name=op_name, category=cat)
        stat = ops[key]
        stat.count += 1
        dur = sl["end_ns"] - sl["start_ns"]
        stat.total_duration_ns += dur
        stat.durations_ns.append(dur)
        b = sl["annotations"].get("bytes")
        if b is not None:
            try:
                stat.total_bytes += int(b)
            except (ValueError, TypeError):
                pass
        t = tracks.get(sl["track_uuid"])
        stat.per_host[t.process_name if t else "unknown"] += dur

    nats_pubs: dict[str, tuple[str, int]] = {}
    nats_pairs: list[NatsPair] = []

    for ev in instant_events:
        t = tracks.get(ev.track_uuid)
        cat = comm_tracks.get(ev.track_uuid) or classify_track(ev.name)
        if cat != "nats":
            continue
        msg_id = ev.annotations.get("msg_id")
        if not msg_id:
            continue
        host = t.process_name if t else "unknown"
        lo = ev.name.lower()
        if "pub" in lo or "send" in lo:
            nats_pubs[msg_id] = (host, ev.timestamp_ns)
        elif "sub" in lo or "recv" in lo or "receive" in lo:
            if msg_id in nats_pubs:
                pub_host, pub_ts = nats_pubs[msg_id]
                if pub_host != host:
                    nats_pairs.append(NatsPair(
                        msg_id=msg_id, pub_host=pub_host, pub_ts=pub_ts,
                        sub_host=host, sub_ts=ev.timestamp_ns,
                        lag_ns=ev.timestamp_ns - pub_ts,
                    ))

    for sl in completed_slices:
        cat = comm_tracks.get(sl["track_uuid"]) or classify_track(sl["name"])
        if cat != "nats":
            continue
        msg_id = sl["annotations"].get("msg_id")
        if not msg_id:
            continue
        t = tracks.get(sl["track_uuid"])
        host = t.process_name if t else "unknown"
        lo = sl["name"].lower()
        if "pub" in lo or "send" in lo:
            nats_pubs[msg_id] = (host, sl["start_ns"])
        elif "sub" in lo or "recv" in lo:
            if msg_id in nats_pubs:
                pub_host, pub_ts = nats_pubs[msg_id]
                if pub_host != host:
                    nats_pairs.append(NatsPair(
                        msg_id=msg_id, pub_host=pub_host, pub_ts=pub_ts,
                        sub_host=host, sub_ts=sl["start_ns"],
                        lag_ns=sl["start_ns"] - pub_ts,
                    ))

    per_category = {}
    for cat in ("nccl", "nixl", "nats"):
        cat_ops = [s for s in ops.values() if s.category == cat]
        if not cat_ops:
            per_category[cat] = {
                "total_duration_ms": 0, "total_bytes": 0,
                "avg_bandwidth_gbps": 0, "op_count": 0, "operations": [],
            }
            continue

        total_dur = sum(s.total_duration_ns for s in cat_ops)
        total_bytes = sum(s.total_bytes for s in cat_ops)
        total_count = sum(s.count for s in cat_ops)
        bw_gbps = (total_bytes * 8) / total_dur if total_dur > 0 and total_bytes > 0 else 0

        op_list = []
        for s in sorted(cat_ops, key=lambda x: -x.total_duration_ns):
            durations = sorted(s.durations_ns)
            n = len(durations)
            op_list.append({
                "op_name": s.op_name,
                "count": s.count,
                "total_duration_ms": s.total_duration_ns / 1e6,
                "avg_duration_us": (s.total_duration_ns / s.count / 1000) if s.count else 0,
                "p50_duration_us": durations[n // 2] / 1000 if n else 0,
                "p99_duration_us": durations[int(n * 0.99)] / 1000 if n else 0,
                "total_bytes": s.total_bytes,
                "bandwidth_gbps": (
                    (s.total_bytes * 8) / s.total_duration_ns
                    if s.total_duration_ns > 0 and s.total_bytes > 0 else 0
                ),
                "per_host_ms": {k: v / 1e6 for k, v in s.per_host.items()},
            })

        per_category[cat] = {
            "total_duration_ms": total_dur / 1e6,
            "total_bytes": total_bytes,
            "avg_bandwidth_gbps": bw_gbps,
            "op_count": total_count,
            "operations": op_list,
        }

    nats_propagation = {"pairs_analyzed": 0, "causality_violations": 0}
    if nats_pairs:
        lags = [p.lag_ns for p in nats_pairs]
        violations = [p for p in nats_pairs if p.lag_ns < 0]
        nats_propagation = {
            "pairs_analyzed": len(nats_pairs),
            "causality_violations": len(violations),
            "min_lag_ns": min(lags),
            "max_lag_ns": max(lags),
            "median_lag_ns": int(statistics.median(lags)),
            "p99_lag_us": sorted(lags)[int(len(lags) * 0.99)] / 1000 if lags else 0,
        }

    return {
        "per_category": per_category,
        "nats_propagation": nats_propagation,
        "summary": {
            "total_comm_duration_ms": sum(c["total_duration_ms"] for c in per_category.values()),
            "total_comm_bytes": sum(c["total_bytes"] for c in per_category.values()),
            "dominant_category": max(
                per_category.items(), key=lambda x: x[1]["total_duration_ms"],
            )[0] if any(c["total_duration_ms"] > 0 for c in per_category.values()) else None,
        },
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-trace", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    result = analyze(args.merged_trace)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
