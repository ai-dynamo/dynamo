# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""nsys SQLite -> Perfetto protobuf converter.

Designed for Dynamo distributed profiling where:
  - LLMs are sharded across multiple GPUs per process (TP/PP).
  - Multiple processes per host, multiple hosts per cluster.
  - Each capture has a PTP offset to align with cluster wall-clock.
  - NVTX ranges follow convention: "stage=<name> req=<trace_id> [k=v ...]"

Output is a Perfetto-protobuf trace file (gzipped optional). Open in
ui.perfetto.dev or merge with peers via the Rust merger.

Usage:
    python -m dynamo_profiler nsys-convert \
        --sqlite capture.sqlite \
        --component engine-prefill-0 \
        --host node-17 \
        --ptp-offset-ns 12345678 \
        --output engine-prefill-0.pftrace

To produce the SQLite from an .nsys-rep:
    nsys export --type=sqlite --output capture.sqlite capture.nsys-rep
"""

from __future__ import annotations

import gzip
import hashlib
import logging
import re
import sqlite3
import struct
import sys
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("nsys2pftrace")


# ----- protobuf wire encoding helpers -----

def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _tag(field: int, wire_type: int) -> bytes:
    return _varint((field << 3) | wire_type)


def _wire_varint(field: int, value: int) -> bytes:
    return _tag(field, 0) + _varint(value)


def _wire_string(field: int, value: str) -> bytes:
    data = value.encode("utf-8")
    return _tag(field, 2) + _varint(len(data)) + data


def _wire_bytes(field: int, value: bytes) -> bytes:
    return _tag(field, 2) + _varint(len(value)) + value


def _wire_double(field: int, value: float) -> bytes:
    return _tag(field, 1) + struct.pack("<d", value)


def _wire_msg(field: int, msg: bytes) -> bytes:
    return _tag(field, 2) + _varint(len(msg)) + msg


# ----- Perfetto message builders -----

class PerfettoWriter:
    """Streaming writer that appends one TracePacket at a time to disk."""

    CLOCK_PTP = 64

    def __init__(self, path: str, gzip_output: bool = True):
        if gzip_output:
            self.f = gzip.open(path, "wb", compresslevel=3)
        else:
            self.f = open(path, "wb")
        self.sequence_id = 1

    def close(self):
        self.f.close()

    def _write_packet(self, packet_body: bytes):
        self.f.write(_wire_bytes(1, packet_body))

    def emit_clock_snapshot(self, primary_trace_clock_ns: int, ptp_clock_ns: int):
        builtin_boottime_clock = (
            _wire_varint(1, 6)
            + _wire_varint(2, primary_trace_clock_ns)
        )
        ptp_clock = (
            _wire_varint(1, self.CLOCK_PTP)
            + _wire_varint(2, ptp_clock_ns)
        )
        snapshot = (
            _wire_msg(1, builtin_boottime_clock)
            + _wire_msg(1, ptp_clock)
        )
        packet = _wire_msg(6, snapshot) + _wire_varint(10, self.sequence_id)
        self._write_packet(packet)

    def emit_process_descriptor(self, uuid: int, pid: int, name: str):
        process = _wire_varint(1, pid) + _wire_string(6, name)
        track_desc = (
            _wire_varint(1, uuid)
            + _wire_string(2, name)
            + _wire_msg(3, process)
        )
        packet = (
            _wire_msg(60, track_desc)
            + _wire_varint(10, self.sequence_id)
        )
        self._write_packet(packet)

    def emit_thread_descriptor(
        self, uuid: int, parent_uuid: int, pid: int, tid: int, name: str
    ):
        thread = (
            _wire_varint(1, pid)
            + _wire_varint(2, tid)
            + _wire_string(5, name)
        )
        track_desc = (
            _wire_varint(1, uuid)
            + _wire_varint(5, parent_uuid)
            + _wire_string(2, name)
            + _wire_msg(4, thread)
        )
        packet = _wire_msg(60, track_desc) + _wire_varint(10, self.sequence_id)
        self._write_packet(packet)

    def emit_track_descriptor(
        self, uuid: int, parent_uuid: int, name: str
    ):
        track_desc = (
            _wire_varint(1, uuid)
            + _wire_varint(5, parent_uuid)
            + _wire_string(2, name)
        )
        packet = _wire_msg(60, track_desc) + _wire_varint(10, self.sequence_id)
        self._write_packet(packet)

    def emit_counter_descriptor(
        self, uuid: int, parent_uuid: int, name: str, unit_name: str = ""
    ):
        counter = _wire_string(6, unit_name) if unit_name else b""
        track_desc = (
            _wire_varint(1, uuid)
            + _wire_varint(5, parent_uuid)
            + _wire_string(2, name)
            + _wire_msg(8, counter)
        )
        packet = _wire_msg(60, track_desc) + _wire_varint(10, self.sequence_id)
        self._write_packet(packet)

    def emit_slice_begin(
        self,
        ts_ns: int,
        track_uuid: int,
        name: str,
        category: str = "",
        annotations: Optional[dict] = None,
    ):
        event = (
            _wire_varint(9, 1)
            + _wire_string(23, name)
            + _wire_varint(11, track_uuid)
        )
        if category:
            event += _wire_string(22, category)
        if annotations:
            for k, v in annotations.items():
                event += _wire_msg(4, _build_debug_annotation(k, v))
        packet = (
            _wire_varint(8, ts_ns)
            + _wire_varint(58, self.CLOCK_PTP)
            + _wire_msg(11, event)
            + _wire_varint(10, self.sequence_id)
        )
        self._write_packet(packet)

    def emit_slice_end(self, ts_ns: int, track_uuid: int):
        event = _wire_varint(9, 2) + _wire_varint(11, track_uuid)
        packet = (
            _wire_varint(8, ts_ns)
            + _wire_varint(58, self.CLOCK_PTP)
            + _wire_msg(11, event)
            + _wire_varint(10, self.sequence_id)
        )
        self._write_packet(packet)

    def emit_instant(
        self,
        ts_ns: int,
        track_uuid: int,
        name: str,
        annotations: Optional[dict] = None,
    ):
        event = (
            _wire_varint(9, 3)
            + _wire_string(23, name)
            + _wire_varint(11, track_uuid)
        )
        if annotations:
            for k, v in annotations.items():
                event += _wire_msg(4, _build_debug_annotation(k, v))
        packet = (
            _wire_varint(8, ts_ns)
            + _wire_varint(58, self.CLOCK_PTP)
            + _wire_msg(11, event)
            + _wire_varint(10, self.sequence_id)
        )
        self._write_packet(packet)

    def emit_counter(self, ts_ns: int, track_uuid: int, value: float):
        event = (
            _wire_varint(9, 4)
            + _wire_varint(11, track_uuid)
            + _wire_double(44, value)
        )
        packet = (
            _wire_varint(8, ts_ns)
            + _wire_varint(58, self.CLOCK_PTP)
            + _wire_msg(11, event)
            + _wire_varint(10, self.sequence_id)
        )
        self._write_packet(packet)


def _build_debug_annotation(name: str, value) -> bytes:
    out = _wire_string(10, name)
    if isinstance(value, bool):
        out += _wire_varint(4, 1 if value else 0)
    elif isinstance(value, int):
        out += _wire_varint(4, value)
    elif isinstance(value, float):
        out += _wire_double(5, value)
    else:
        out += _wire_string(6, str(value))
    return out


# ----- nsys SQLite reading -----

@dataclass
class NvtxRange:
    start_ns: int
    end_ns: int
    text: str
    global_tid: int
    trace_id: Optional[str]
    stage: Optional[str]


TRACE_ID_RE = re.compile(r"\breq[=:]([A-Za-z0-9_\-]+)")
STAGE_RE = re.compile(r"\bstage[=:]([A-Za-z0-9_\-\.]+)")


def _extract_trace_id(text: str) -> Optional[str]:
    m = TRACE_ID_RE.search(text or "")
    return m.group(1) if m else None


def _extract_stage(text: str) -> Optional[str]:
    m = STAGE_RE.search(text or "")
    return m.group(1) if m else None


def unpack_global_tid(global_tid: int) -> tuple[int, int]:
    pid = (global_tid // 0x1000000) % 0x1000000
    tid = global_tid % 0x1000000
    return pid, tid


def stable_uuid(*parts: str) -> int:
    h = hashlib.sha256(("|".join(parts)).encode("utf-8")).digest()
    val = int.from_bytes(h[:8], "big") | 1
    return val


# ----- Per-thread NVTX index for fast enclosing-range lookup -----

class NvtxIndex:
    def __init__(self):
        self._per_tid: dict[int, tuple[list[int], list[NvtxRange]]] = {}

    def build(self, ranges: list[NvtxRange]):
        by_tid: dict[int, list[NvtxRange]] = defaultdict(list)
        for r in ranges:
            by_tid[r.global_tid].append(r)
        for tid, rs in by_tid.items():
            rs.sort(key=lambda r: r.start_ns)
            starts = [r.start_ns for r in rs]
            self._per_tid[tid] = (starts, rs)

    def innermost_at(self, tid: int, ts: int) -> Optional[NvtxRange]:
        entry = self._per_tid.get(tid)
        if not entry:
            return None
        starts, rs = entry
        idx = bisect_right(starts, ts) - 1
        while idx >= 0:
            r = rs[idx]
            if r.end_ns >= ts:
                return r
            idx -= 1
        return None


# ----- The converter -----

class NsysToPerfettoConverter:
    """Reads an nsys SQLite export and emits a Perfetto-protobuf trace."""

    def __init__(
        self,
        sqlite_path: str,
        output_path: str,
        component: str,
        host: str,
        ptp_offset_ns: int,
        gzip_output: bool = True,
    ):
        self.conn = sqlite3.connect(sqlite_path)
        self.conn.row_factory = sqlite3.Row
        self.writer = PerfettoWriter(output_path, gzip_output=gzip_output)
        self.component = component
        self.host = host
        self.ptp_offset = ptp_offset_ns
        self.process_pid = stable_uuid("pid", host, component) % (2**31)
        self.process_uuid = stable_uuid("proc", host, component)
        self._gpu_compute_track: dict[int, int] = {}
        self._gpu_memcpy_tracks: dict[tuple[int, str], int] = {}
        self._gpu_metric_tracks: dict[tuple[int, str], int] = {}
        self._thread_tracks: dict[int, int] = {}
        self._nvtx_app_track: int = 0
        self._nccl_track: int = 0
        self.nvtx_index = NvtxIndex()

    def convert(self):
        self._emit_initial_descriptors()
        self._build_nvtx_index()
        self._convert_nvtx_ranges()
        self._convert_kernels()
        self._convert_memcpys()
        self._convert_nccl()
        self._convert_gpu_metrics()
        self.writer.close()
        log.info("Conversion complete")

    def _emit_initial_descriptors(self):
        self.writer.emit_process_descriptor(
            uuid=self.process_uuid,
            pid=self.process_pid,
            name=f"{self.host}/{self.component}",
        )
        self._nvtx_app_track = stable_uuid("nvtx_app", self.host, self.component)
        self.writer.emit_track_descriptor(
            uuid=self._nvtx_app_track,
            parent_uuid=self.process_uuid,
            name="NVTX (Application)",
        )
        self._nccl_track = stable_uuid("nccl", self.host, self.component)
        self.writer.emit_track_descriptor(
            uuid=self._nccl_track,
            parent_uuid=self.process_uuid,
            name="NCCL Collectives",
        )

    def _ensure_gpu_compute_track(self, device_id: int) -> int:
        if device_id not in self._gpu_compute_track:
            uuid = stable_uuid("gpu_compute", self.host, self.component, str(device_id))
            self._gpu_compute_track[device_id] = uuid
            self.writer.emit_track_descriptor(
                uuid=uuid,
                parent_uuid=self.process_uuid,
                name=f"GPU{device_id} Compute",
            )
        return self._gpu_compute_track[device_id]

    def _ensure_gpu_memcpy_track(self, device_id: int, kind: str) -> int:
        key = (device_id, kind)
        if key not in self._gpu_memcpy_tracks:
            uuid = stable_uuid("gpu_memcpy", self.host, self.component,
                               str(device_id), kind)
            self._gpu_memcpy_tracks[key] = uuid
            self.writer.emit_track_descriptor(
                uuid=uuid,
                parent_uuid=self.process_uuid,
                name=f"GPU{device_id} Memcpy {kind}",
            )
        return self._gpu_memcpy_tracks[key]

    def _ensure_gpu_metric_track(self, device_id: int, metric: str, unit: str) -> int:
        key = (device_id, metric)
        if key not in self._gpu_metric_tracks:
            uuid = stable_uuid("gpu_metric", self.host, self.component,
                               str(device_id), metric)
            self._gpu_metric_tracks[key] = uuid
            self.writer.emit_counter_descriptor(
                uuid=uuid,
                parent_uuid=self.process_uuid,
                name=f"GPU{device_id} {metric}",
                unit_name=unit,
            )
        return self._gpu_metric_tracks[key]

    def _ensure_thread_track(self, global_tid: int, thread_name: str = "") -> int:
        if global_tid in self._thread_tracks:
            return self._thread_tracks[global_tid]
        pid, tid = unpack_global_tid(global_tid)
        uuid = stable_uuid("thread", self.host, self.component, str(global_tid))
        self._thread_tracks[global_tid] = uuid
        self.writer.emit_thread_descriptor(
            uuid=uuid,
            parent_uuid=self.process_uuid,
            pid=self.process_pid,
            tid=tid,
            name=thread_name or f"tid_{tid}",
        )
        return uuid

    def _build_nvtx_index(self):
        ranges: list[NvtxRange] = []
        cur = self.conn.execute("""
            SELECT n.start AS start_ns, n.end AS end_ns,
                   COALESCE(ns.value, n.text) AS text,
                   n.globalTid AS global_tid
            FROM NVTX_EVENTS n
            LEFT JOIN StringIds ns ON n.textId = ns.id
            WHERE n.eventType = 59
              AND n.start IS NOT NULL AND n.end IS NOT NULL
        """)
        for row in cur:
            text = row["text"] or ""
            ranges.append(NvtxRange(
                start_ns=row["start_ns"],
                end_ns=row["end_ns"],
                text=text,
                global_tid=row["global_tid"],
                trace_id=_extract_trace_id(text),
                stage=_extract_stage(text),
            ))
        self.nvtx_index.build(ranges)
        self._nvtx_ranges = ranges
        log.info("Indexed %d NVTX ranges across %d threads",
                 len(ranges), len(set(r.global_tid for r in ranges)))

    def _convert_nvtx_ranges(self):
        for r in self._nvtx_ranges:
            ann = {}
            if r.trace_id:
                ann["trace_id"] = r.trace_id
            if r.stage:
                ann["stage"] = r.stage
            ann["raw_text"] = r.text
            self.writer.emit_slice_begin(
                ts_ns=r.start_ns + self.ptp_offset,
                track_uuid=self._nvtx_app_track,
                name=r.stage or r.text,
                category="nvtx",
                annotations=ann,
            )
            self.writer.emit_slice_end(
                ts_ns=r.end_ns + self.ptp_offset,
                track_uuid=self._nvtx_app_track,
            )

    def _convert_kernels(self):
        cur = self.conn.execute("""
            SELECT k.start AS start_ns, k.end AS end_ns,
                   sk.value AS kernel_name,
                   k.deviceId AS device_id,
                   k.streamId AS stream_id,
                   r.globalTid AS launch_tid,
                   r.start AS launch_start_ns
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds sk ON k.shortName = sk.id
            LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME r
                ON k.correlationId = r.correlationId
            ORDER BY k.start
        """)
        n = 0
        for row in cur:
            device_id = row["device_id"] or 0
            track = self._ensure_gpu_compute_track(device_id)
            nvtx = None
            if row["launch_tid"] is not None and row["launch_start_ns"] is not None:
                nvtx = self.nvtx_index.innermost_at(
                    row["launch_tid"], row["launch_start_ns"]
                )
            ann = {
                "device": device_id,
                "stream": row["stream_id"] or 0,
            }
            if nvtx:
                if nvtx.trace_id:
                    ann["trace_id"] = nvtx.trace_id
                if nvtx.stage:
                    ann["stage"] = nvtx.stage
            self.writer.emit_slice_begin(
                ts_ns=row["start_ns"] + self.ptp_offset,
                track_uuid=track,
                name=row["kernel_name"],
                category="cuda_kernel",
                annotations=ann,
            )
            self.writer.emit_slice_end(
                ts_ns=row["end_ns"] + self.ptp_offset,
                track_uuid=track,
            )
            n += 1
        log.info("Converted %d CUDA kernels", n)

    def _convert_memcpys(self):
        kind_map = {1: "H2D", 2: "D2H", 8: "D2D", 9: "H2H"}
        cur = self.conn.execute("""
            SELECT start AS start_ns, end AS end_ns,
                   deviceId AS device_id, copyKind, bytes
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            ORDER BY start
        """)
        n = 0
        for row in cur:
            device_id = row["device_id"] or 0
            kind = kind_map.get(row["copyKind"], f"kind{row['copyKind']}")
            track = self._ensure_gpu_memcpy_track(device_id, kind)
            self.writer.emit_slice_begin(
                ts_ns=row["start_ns"] + self.ptp_offset,
                track_uuid=track,
                name=f"memcpy {kind} {row['bytes']}B",
                category="cuda_memcpy",
                annotations={"bytes": row["bytes"], "kind": kind},
            )
            self.writer.emit_slice_end(
                ts_ns=row["end_ns"] + self.ptp_offset,
                track_uuid=track,
            )
            n += 1
        log.info("Converted %d memcpys", n)

    def _convert_nccl(self):
        tables = {
            row[0] for row in self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "NCCL_API_EVENTS" not in tables and "NCCL_EVENTS" not in tables:
            log.info("No NCCL tables in capture, skipping")
            return
        try:
            cur = self.conn.execute("""
                SELECT start AS start_ns, end AS end_ns,
                       sn.value AS op_name,
                       globalTid
                FROM NCCL_API_EVENTS n
                LEFT JOIN StringIds sn ON n.opNameId = sn.id
                ORDER BY start
            """)
            n = 0
            for row in cur:
                ann = {"op": row["op_name"] or "unknown"}
                self.writer.emit_slice_begin(
                    ts_ns=row["start_ns"] + self.ptp_offset,
                    track_uuid=self._nccl_track,
                    name=row["op_name"] or "nccl",
                    category="nccl",
                    annotations=ann,
                )
                self.writer.emit_slice_end(
                    ts_ns=row["end_ns"] + self.ptp_offset,
                    track_uuid=self._nccl_track,
                )
                n += 1
            log.info("Converted %d NCCL events", n)
        except sqlite3.OperationalError as e:
            log.warning("NCCL table present but schema unexpected: %s", e)

    def _convert_gpu_metrics(self):
        tables = {
            row[0] for row in self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "GPU_METRICS" not in tables:
            log.info("No GPU_METRICS table, skipping")
            return
        try:
            cur = self.conn.execute("""
                SELECT m.timestamp AS ts_ns, m.typeId, m.value, m.deviceId
                FROM GPU_METRICS m
                ORDER BY m.timestamp
            """)
            metric_names = self._load_gpu_metric_names()
            n = 0
            for row in cur:
                metric = metric_names.get(row["typeId"], f"metric_{row['typeId']}")
                device_id = row["deviceId"] or 0
                unit = "%"
                if "bandwidth" in metric.lower() or "bytes" in metric.lower():
                    unit = "GB/s"
                elif "freq" in metric.lower() or "clock" in metric.lower():
                    unit = "MHz"
                elif "memory" in metric.lower():
                    unit = "MiB"
                track = self._ensure_gpu_metric_track(device_id, metric, unit)
                self.writer.emit_counter(
                    ts_ns=row["ts_ns"] + self.ptp_offset,
                    track_uuid=track,
                    value=float(row["value"]),
                )
                n += 1
            log.info("Converted %d GPU metric samples", n)
        except sqlite3.OperationalError as e:
            log.warning("GPU_METRICS schema unexpected: %s", e)

    def _load_gpu_metric_names(self) -> dict[int, str]:
        names: dict[int, str] = {}
        try:
            cur = self.conn.execute(
                "SELECT id, value FROM TARGET_INFO_GPU_METRICS"
            )
            for row in cur:
                names[row[0]] = row[1]
        except sqlite3.OperationalError:
            pass
        return names


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Convert nsys SQLite export to Perfetto protobuf"
    )
    parser.add_argument("--sqlite", required=True, help="nsys SQLite export")
    parser.add_argument("--component", required=True,
                        help="Logical component name, e.g. engine-prefill-0")
    parser.add_argument("--host", required=True,
                        help="Host where capture was taken")
    parser.add_argument("--ptp-offset-ns", type=int, default=0,
                        help="Add this many ns to every timestamp to align "
                             "to PTP master clock")
    parser.add_argument("--output", required=True, help="Output .pftrace file")
    parser.add_argument("--no-gzip", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    converter = NsysToPerfettoConverter(
        sqlite_path=args.sqlite,
        output_path=args.output,
        component=args.component,
        host=args.host,
        ptp_offset_ns=args.ptp_offset_ns,
        gzip_output=not args.no_gzip,
    )
    converter.convert()


if __name__ == "__main__":
    sys.exit(main() or 0)
