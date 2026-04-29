# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Perfetto protobuf wire-format reader.

Parses .pftrace.gz files produced by the Rust TraceWriter (dynamo-sysprofile)
or by nsys_to_perfetto.py. Extracts track descriptors and track events
(slices, instants, counters) with debug annotations.

No external dependencies — uses raw protobuf wire format parsing.
"""

from __future__ import annotations

import gzip
import struct
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterator, Optional


# -- Protobuf wire-format primitives ------------------------------------------

def read_varint(buf, off: int) -> tuple[int, int]:
    result, shift = 0, 0
    while True:
        b = buf[off]
        off += 1
        result |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            return result, off
        shift += 7


def read_tag(buf, off: int) -> tuple[int, int, int]:
    tag, off = read_varint(buf, off)
    return tag >> 3, tag & 0x07, off


def skip_field(buf, off: int, wire_type: int) -> int:
    if wire_type == 0:
        _, off = read_varint(buf, off)
    elif wire_type == 1:
        off += 8
    elif wire_type == 2:
        ln, off = read_varint(buf, off)
        off += ln
    elif wire_type == 5:
        off += 4
    return off


# -- Data types ---------------------------------------------------------------

@dataclass
class TrackInfo:
    uuid: int
    parent_uuid: Optional[int] = None
    name: str = ""
    process_name: str = ""
    pid: int = 0
    tid: int = 0
    is_process: bool = False


@dataclass
class Event:
    timestamp_ns: int
    track_uuid: int
    event_type: int  # 1=SLICE_BEGIN 2=SLICE_END 3=INSTANT 4=COUNTER
    name: str = ""
    name_iid: int = 0
    annotations: dict = field(default_factory=dict)


@dataclass
class Slice:
    stage: str
    traceparent: str
    run_id: str
    start_ns: int
    end_ns: int
    duration_ns: int
    track_uuid: int
    process_name: str
    thread_name: str
    pid: int = 0
    tid: int = 0


@dataclass
class ParsedTrace:
    source_file: str
    slices: list[Slice]
    instants: list[Event]
    tracks: dict[int, TrackInfo]


# -- Packet-level parsing -----------------------------------------------------

def _parse_track_descriptor(buf) -> TrackInfo:
    track = TrackInfo(uuid=0)
    off = 0
    while off < len(buf):
        field_num, wire_type, off = read_tag(buf, off)
        if field_num == 1 and wire_type == 0:
            track.uuid, off = read_varint(buf, off)
        elif field_num == 5 and wire_type == 0:
            track.parent_uuid, off = read_varint(buf, off)
        elif field_num == 2 and wire_type == 2:
            ln, off = read_varint(buf, off)
            track.name = bytes(buf[off:off + ln]).decode("utf-8", errors="replace")
            off += ln
        elif field_num == 3 and wire_type == 2:
            ln, off = read_varint(buf, off)
            track.is_process = True
            _parse_process_descriptor(buf[off:off + ln], track)
            off += ln
        elif field_num == 4 and wire_type == 2:
            ln, off = read_varint(buf, off)
            _parse_thread_descriptor(buf[off:off + ln], track)
            off += ln
        else:
            off = skip_field(buf, off, wire_type)
    return track


def _parse_process_descriptor(buf, track: TrackInfo):
    off = 0
    while off < len(buf):
        field_num, wire_type, off = read_tag(buf, off)
        if field_num == 1 and wire_type == 0:
            track.pid, off = read_varint(buf, off)
        elif field_num == 6 and wire_type == 2:
            ln, off = read_varint(buf, off)
            track.process_name = bytes(buf[off:off + ln]).decode("utf-8", errors="replace")
            off += ln
        else:
            off = skip_field(buf, off, wire_type)


def _parse_thread_descriptor(buf, track: TrackInfo):
    off = 0
    while off < len(buf):
        field_num, wire_type, off = read_tag(buf, off)
        if field_num == 1 and wire_type == 0:
            track.pid, off = read_varint(buf, off)
        elif field_num == 2 and wire_type == 0:
            track.tid, off = read_varint(buf, off)
        else:
            off = skip_field(buf, off, wire_type)


def _parse_debug_annotation(buf, out: dict):
    name = ""
    value = None
    off = 0
    while off < len(buf):
        field_num, wire_type, off = read_tag(buf, off)
        if field_num == 10 and wire_type == 2:
            ln, off = read_varint(buf, off)
            name = bytes(buf[off:off + ln]).decode("utf-8", errors="replace")
            off += ln
        elif field_num == 6 and wire_type == 2:
            ln, off = read_varint(buf, off)
            value = bytes(buf[off:off + ln]).decode("utf-8", errors="replace")
            off += ln
        elif field_num == 4 and wire_type == 0:
            value, off = read_varint(buf, off)
        elif field_num == 5 and wire_type == 1:
            value = struct.unpack_from("<d", buf, off)[0]
            off += 8
        else:
            off = skip_field(buf, off, wire_type)
    if name:
        out[name] = value


def _parse_track_event(buf, timestamp_ns: int) -> Event:
    event = Event(timestamp_ns=timestamp_ns, track_uuid=0, event_type=0)
    off = 0
    while off < len(buf):
        field_num, wire_type, off = read_tag(buf, off)
        if field_num == 9 and wire_type == 0:
            event.event_type, off = read_varint(buf, off)
        elif field_num == 11 and wire_type == 0:
            event.track_uuid, off = read_varint(buf, off)
        elif field_num == 10 and wire_type == 0:
            event.name_iid, off = read_varint(buf, off)
        elif field_num == 23 and wire_type == 2:
            ln, off = read_varint(buf, off)
            event.name = bytes(buf[off:off + ln]).decode("utf-8", errors="replace")
            off += ln
        elif field_num == 4 and wire_type == 2:
            ln, off = read_varint(buf, off)
            _parse_debug_annotation(buf[off:off + ln], event.annotations)
            off += ln
        else:
            off = skip_field(buf, off, wire_type)
    return event


def parse_packet(
    raw: bytes,
    interned_names: dict | None = None,
) -> tuple[Optional[Event], Optional[TrackInfo]]:
    """Parse a single TracePacket. Returns (event, track_info).

    Pass a mutable dict as *interned_names* to accumulate interned event
    names across packets and auto-resolve ``name_iid`` → ``event.name``.
    """
    buf = memoryview(raw)
    off = 0
    timestamp_ns = None
    seq_id = 0
    track_event_bytes = None
    track_descriptor_bytes = None
    interned_data_bytes = None

    while off < len(buf):
        field_num, wire_type, off = read_tag(buf, off)
        if field_num == 8 and wire_type == 0:
            timestamp_ns, off = read_varint(buf, off)
        elif field_num == 10 and wire_type == 0:
            seq_id, off = read_varint(buf, off)
        elif field_num == 11 and wire_type == 2:
            ln, off = read_varint(buf, off)
            track_event_bytes = buf[off:off + ln]
            off += ln
        elif field_num == 60 and wire_type == 2:
            ln, off = read_varint(buf, off)
            track_descriptor_bytes = buf[off:off + ln]
            off += ln
        elif field_num == 12 and wire_type == 2:
            ln, off = read_varint(buf, off)
            interned_data_bytes = buf[off:off + ln]
            off += ln
        else:
            off = skip_field(buf, off, wire_type)

    if interned_data_bytes is not None and interned_names is not None:
        _parse_interned_data(interned_data_bytes, seq_id, interned_names)

    if track_descriptor_bytes is not None:
        track = _parse_track_descriptor(track_descriptor_bytes)
        return None, track
    if track_event_bytes is not None and timestamp_ns is not None:
        event = _parse_track_event(track_event_bytes, timestamp_ns)
        if interned_names is not None and event.name_iid and not event.name:
            event.name = interned_names.get((seq_id, event.name_iid), "")
        return event, None
    return None, None


# -- File-level reading -------------------------------------------------------

def iter_packets(path: str) -> Iterator[bytes]:
    """Iterate over raw TracePacket bytes from a .pftrace[.gz] file."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        data = f.read()
    buf = memoryview(data)
    off = 0
    while off < len(buf):
        field_num, wire_type, off = read_tag(buf, off)
        if field_num != 1 or wire_type != 2:
            off = skip_field(buf, off, wire_type)
            continue
        ln, off = read_varint(buf, off)
        yield bytes(buf[off:off + ln])
        off += ln


def read_trace(path: str) -> ParsedTrace:
    """Read a .pftrace[.gz] file and return all slices and tracks.

    Mirrors the Rust `reader::read_trace` API. Handles interned event names
    and resolves track parentage for process/thread names on slices.
    """
    import os
    source = os.path.basename(path)
    tracks: dict[int, TrackInfo] = {}
    interned_names: dict[tuple[int, int], str] = {}
    open_slices: dict[int, list] = defaultdict(list)
    slices: list[Slice] = []
    instants: list[Event] = []
    seq_id = 0

    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        data = f.read()

    buf = memoryview(data)
    off = 0
    while off < len(buf):
        field_num, wire_type, off = read_tag(buf, off)
        if field_num != 1 or wire_type != 2:
            off = skip_field(buf, off, wire_type)
            continue
        ln, off = read_varint(buf, off)
        pkt = buf[off:off + ln]
        off += ln

        _process_packet_for_slices(
            pkt, tracks, interned_names, open_slices, slices, instants
        )

    _resolve_names(slices, instants, tracks)
    return ParsedTrace(source_file=source, slices=slices, instants=instants, tracks=tracks)


def _process_packet_for_slices(
    pkt,
    tracks: dict[int, TrackInfo],
    interned_names: dict[tuple[int, int], str],
    open_slices: dict[int, list],
    slices: list[Slice],
    instants: list[Event],
):
    poff = 0
    timestamp = 0
    seq_id = 0
    td_bytes = None
    te_bytes = None
    id_bytes = None

    while poff < len(pkt):
        fn, wt, poff = read_tag(pkt, poff)
        if fn == 8 and wt == 0:
            timestamp, poff = read_varint(pkt, poff)
        elif fn == 10 and wt == 0:
            seq_id, poff = read_varint(pkt, poff)
        elif fn == 60 and wt == 2:
            ln, poff = read_varint(pkt, poff)
            td_bytes = pkt[poff:poff + ln]
            poff += ln
        elif fn == 11 and wt == 2:
            ln, poff = read_varint(pkt, poff)
            te_bytes = pkt[poff:poff + ln]
            poff += ln
        elif fn == 12 and wt == 2:
            ln, poff = read_varint(pkt, poff)
            id_bytes = pkt[poff:poff + ln]
            poff += ln
        else:
            poff = skip_field(pkt, poff, wt)

    if td_bytes is not None:
        t = _parse_track_descriptor(td_bytes)
        tracks[t.uuid] = t

    if id_bytes is not None:
        _parse_interned_data(id_bytes, seq_id, interned_names)

    if te_bytes is not None:
        _handle_track_event(
            te_bytes, timestamp, seq_id, interned_names,
            open_slices, slices, instants,
        )


def _parse_interned_data(buf, seq_id: int, names: dict[tuple[int, int], str]):
    off = 0
    while off < len(buf):
        fn, wt, off = read_tag(buf, off)
        if fn == 2 and wt == 2:
            ln, off = read_varint(buf, off)
            entry = buf[off:off + ln]
            iid, name = _parse_interned_string(entry)
            if iid > 0:
                names[(seq_id, iid)] = name
            off += ln
        else:
            off = skip_field(buf, off, wt)


def _parse_interned_string(buf) -> tuple[int, str]:
    iid = 0
    name = ""
    off = 0
    while off < len(buf):
        fn, wt, off = read_tag(buf, off)
        if fn == 1 and wt == 0:
            iid, off = read_varint(buf, off)
        elif fn == 2 and wt == 2:
            ln, off = read_varint(buf, off)
            name = bytes(buf[off:off + ln]).decode("utf-8", errors="replace")
            off += ln
        else:
            off = skip_field(buf, off, wt)
    return iid, name


def _handle_track_event(
    buf, timestamp: int, seq_id: int,
    interned_names: dict[tuple[int, int], str],
    open_slices: dict[int, list],
    slices: list[Slice],
    instants: list[Event],
):
    track_uuid = 0
    event_type = 0
    name_iid = 0
    traceparent = ""
    run_id = ""

    off = 0
    while off < len(buf):
        fn, wt, off = read_tag(buf, off)
        if fn == 11 and wt == 0:
            track_uuid, off = read_varint(buf, off)
        elif fn == 9 and wt == 0:
            event_type, off = read_varint(buf, off)
        elif fn == 10 and wt == 0:
            name_iid, off = read_varint(buf, off)
        elif fn == 4 and wt == 2:
            ln, off = read_varint(buf, off)
            anns: dict = {}
            _parse_debug_annotation(buf[off:off + ln], anns)
            if "traceparent" in anns:
                traceparent = anns["traceparent"]
            if "dynamo.run_id" in anns:
                run_id = anns["dynamo.run_id"]
            off += ln
        else:
            off = skip_field(buf, off, wt)

    stage_name = interned_names.get((seq_id, name_iid), "")

    if event_type == 1:  # SLICE_BEGIN
        open_slices.setdefault(track_uuid, []).append({
            "stage": stage_name,
            "traceparent": traceparent,
            "run_id": run_id,
            "start_ns": timestamp,
            "track_uuid": track_uuid,
        })
    elif event_type == 2:  # SLICE_END
        stack = open_slices.get(track_uuid, [])
        if stack:
            pending = stack.pop()
            duration = timestamp - pending["start_ns"]
            slices.append(Slice(
                stage=pending["stage"],
                traceparent=pending["traceparent"],
                run_id=pending["run_id"],
                start_ns=pending["start_ns"],
                end_ns=timestamp,
                duration_ns=duration,
                track_uuid=pending["track_uuid"],
                process_name="",
                thread_name="",
            ))
    elif event_type == 3:  # INSTANT
        instants.append(Event(
            timestamp_ns=timestamp,
            track_uuid=track_uuid,
            event_type=3,
            name=stage_name,
            annotations={"traceparent": traceparent, "dynamo.run_id": run_id},
        ))


def _resolve_names(
    slices: list[Slice],
    instants: list[Event],
    tracks: dict[int, TrackInfo],
):
    for sl in slices:
        track = tracks.get(sl.track_uuid)
        if track:
            sl.thread_name = track.name
            sl.tid = track.tid
            parent = tracks.get(track.parent_uuid)
            if parent:
                sl.process_name = parent.process_name or parent.name
                sl.pid = parent.pid


# -- Utilities ----------------------------------------------------------------

def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping (start, end) intervals. Returns sorted, merged."""
    if not intervals:
        return []
    intervals.sort()
    out = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def invert_intervals(
    busy: list[tuple[int, int]], capture_start: int, capture_end: int
) -> list[tuple[int, int]]:
    """Return idle intervals = capture window minus busy intervals."""
    idle = []
    cursor = capture_start
    for start, end in busy:
        if start > cursor:
            idle.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < capture_end:
        idle.append((cursor, capture_end))
    return idle
