#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize response-stream packet multiplexing from Quinn qlog captures.

Input may be qlog JSON text sequences (one object per line) or a regular JSON
document. The output is deliberately small and stable so a locked benchmark can
include it directly in comparison.json and enforce the >=50% multi-stream gate.

Quinn 0.11's built-in packet_sent events omit the frames array. For those
captures, pass the worker's short diagnostic JSON log with --quinn-trace-log.
Quinn emits every STREAM frame inside a per-packet `send` span, so the fallback
groups exact stream IDs by that span. Without either source of frame details,
the analyzer fails the gate instead of inferring sharing from aggregate counts.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def _objects(path: Path) -> Iterable[Any]:
    text = path.read_text(encoding="utf-8")
    try:
        yield json.loads(text)
        return
    except json.JSONDecodeError:
        pass

    for line_number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line == "\x1e":
            continue
        if line.startswith("\x1e"):
            line = line[1:]
        try:
            yield json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid qlog JSON: {error}") from error


def _walk(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _stream_ids(packet: dict[str, Any]) -> set[str]:
    data = packet.get("data")
    frames = data.get("frames", []) if isinstance(data, dict) else []
    stream_ids: set[str] = set()
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        frame_type = str(frame.get("frame_type", frame.get("frameType", ""))).lower()
        if frame_type != "stream":
            continue
        stream_id = frame.get("stream_id", frame.get("streamId"))
        if stream_id is not None:
            stream_ids.add(str(stream_id))
    return stream_ids


def _trace_packet_streams(paths: list[Path]) -> tuple[dict[str, set[str]], int]:
    packets: dict[str, set[str]] = {}
    stream_frame_events = 0
    for path_index, path in enumerate(paths):
        for root in _objects(path):
            for event in _walk(root):
                target = str(event.get("target", ""))
                message = str(event.get("message", "")).upper()
                if not target.startswith("quinn_proto::") or message != "STREAM":
                    continue
                if str(event.get("span_name", "")) != "send":
                    continue
                stream_id = event.get("id")
                if stream_id is None:
                    continue
                span_id = event.get("span_id")
                if span_id is not None:
                    packet_key = f"{path_index}:span:{span_id}"
                else:
                    packet_number = event.get("pn")
                    if packet_number is None:
                        continue
                    packet_key = (
                        f"{path_index}:pn:{event.get('space', 'unknown')}:{packet_number}"
                    )
                packets.setdefault(packet_key, set()).add(str(stream_id))
                stream_frame_events += 1
    return packets, stream_frame_events


def summarize(paths: list[Path], trace_paths: list[Path]) -> dict[str, Any]:
    histogram: Counter[int] = Counter()
    packet_count = 0
    packet_sent_events = 0
    packet_sent_events_with_frame_details = 0
    for path in paths:
        for root in _objects(path):
            for event in _walk(root):
                name = str(event.get("name", "")).lower()
                if not name.endswith("packet_sent"):
                    continue
                packet_sent_events += 1
                data = event.get("data")
                if isinstance(data, dict) and isinstance(data.get("frames"), list):
                    packet_sent_events_with_frame_details += 1
                stream_ids = _stream_ids(event)
                if not stream_ids:
                    continue
                packet_count += 1
                histogram[len(stream_ids)] += 1

    frame_detail_source = "qlog" if packet_count else None
    trace_frame_events = 0
    if not packet_count and trace_paths:
        trace_packets, trace_frame_events = _trace_packet_streams(trace_paths)
        for stream_ids in trace_packets.values():
            packet_count += 1
            histogram[len(stream_ids)] += 1
        if packet_count:
            frame_detail_source = "quinn_trace_log"

    multi_stream_packets = sum(
        count for stream_count, count in histogram.items() if stream_count >= 2
    )
    percentage = (
        100.0 * multi_stream_packets / packet_count if packet_count else 0.0
    )
    return {
        "qlog_files": [str(path) for path in paths],
        "quinn_trace_log_files": [str(path) for path in trace_paths],
        "packet_sent_events": packet_sent_events,
        "packet_sent_events_with_frame_details": packet_sent_events_with_frame_details,
        "quinn_trace_stream_frame_events": trace_frame_events,
        "response_carrying_packets": packet_count,
        "multi_response_stream_packets": multi_stream_packets,
        "multi_response_stream_packet_percentage": percentage,
        "distinct_response_streams_per_packet_histogram": {
            str(key): histogram[key] for key in sorted(histogram)
        },
        "passes_50_percent_gate": packet_count > 0 and percentage >= 50.0,
        "frame_detail_available": frame_detail_source is not None,
        "frame_detail_source": frame_detail_source,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("qlog", nargs="+", type=Path)
    parser.add_argument("--quinn-trace-log", nargs="*", default=[], type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = summarize(args.qlog, args.quinn_trace_log)
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
