# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Full offline inspection of native artifacts, run by the bundle job only.

The capture path checks presence (artifacts.py, nsys.check_report); parsing,
hashing, recording bounds and required-activity checks live here so the serving
frontend never imports or runs them.
"""

import hashlib
import json
import sqlite3
import time
import zlib
from contextlib import closing
from decimal import Decimal
from pathlib import Path

import ijson

from .models import Config
from .native import Layout, kineto_blocks, open_trace
from .nsys import open_export, report_bounds

CPU_TABLES = ("COMPOSITE_EVENTS", "SCHED_EVENTS", "OSRT_API", "NVTX_EVENTS")
CUDA_TABLES = (
    "CUPTI_ACTIVITY_KIND_KERNEL",
    "CUPTI_ACTIVITY_KIND_MEMCPY",
    "CUPTI_ACTIVITY_KIND_MEMSET",
)


def sha256_file(path: Path, deadline: float | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Artifact validation deadline expired")
            digest.update(chunk)
    return digest.hexdigest()


def file_state(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


class BoundedReader:
    def __init__(self, stream, limit: int, deadline: float):
        self.stream, self.limit, self.deadline, self.bytes = stream, limit, deadline, 0
        self.digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Artifact validation deadline expired")
        data = self.stream.read(size if 0 <= size <= 1 << 22 else 1 << 22)
        self.bytes += len(data)
        if self.bytes > self.limit:
            raise ValueError("Trace exceeds uncompressed byte limit")
        self.digest.update(data)
        return data


def recording_bounds(base_time_ns: int | None, markers: list[dict]) -> dict:
    """Report native recorder bounds without claiming synchronized clocks."""
    if base_time_ns is None or len(markers) != 1:
        return {
            "status": "unknown",
            "reason": "missing_or_ambiguous_native_profiler_bounds",
        }
    timestamp, duration = values = markers[0].get("ts"), markers[0].get("dur")
    if (
        any(type(v) not in (int, Decimal) or not Decimal(v).is_finite() for v in values)
        or duration < 0
    ):
        return {"status": "unknown", "reason": "invalid_native_profiler_bounds"}
    start = base_time_ns + round(timestamp * 1000)
    duration_ns = round(duration * 1000)
    return {
        "status": "observed",
        "clock": "worker_trace_unix_ns_unattested",
        "start_unix_ns": start,
        "end_unix_ns": start + duration_ns,
        "duration_ns": duration_ns,
    }


def is_marker(event: dict) -> bool:
    """The profiler's own recording-window event."""
    return (
        event.get("cat") == "Trace"
        and event.get("ph") == "X"
        and isinstance(event.get("name"), str)
        and event["name"].startswith("PyTorch Profiler")
    )


def scan_blocks(reader) -> tuple:
    """Check and count Kineto's writer layout without decoding every event.

    The header, the trailer and the block structure are checked here and the
    stream is hashed and size-limited; each event's JSON is decoded during
    timeline conversion, which reports a malformed event as a failed target.
    Only profiler-marker blocks are decoded (with Decimal, for exact ns
    bounds). Raises Layout to fall back to the full token parser.
    """
    blocks = kineto_blocks(reader)
    base = next(blocks).get("baseTimeNanoseconds")
    events = kernels = 0
    markers = []
    for block in blocks:
        events += 1
        if '"cat": "kernel"' in block:
            kernels += 1
        elif '"PyTorch Profiler' in block and len(markers) < 2:
            try:
                event = json.loads(block, parse_float=Decimal)
            except ValueError as exc:
                raise Layout(str(exc)) from exc
            if is_marker(event):
                markers.append(event)
    return True, events, kernels, base if type(base) is int else None, markers


FIELDS = {f"traceEvents.item.{key}" for key in ("name", "cat", "ph", "ts", "dur")}


def scan_tokens(reader) -> tuple:
    """Count events with the ijson token parser (any valid trace layout)."""
    events = kernels = 0
    has_events, base_time_ns, markers, current = False, None, [], {}
    for prefix, event, value in ijson.parse(reader):
        if (prefix, event) == ("traceEvents", "start_array"):
            has_events = True
        elif (prefix, event) == ("traceEvents.item", "start_map"):
            events += 1
            current = {}
        elif prefix == "traceEvents.item" and event == "end_map":
            if is_marker(current) and len(markers) < 2:
                markers.append(current)
        elif prefix in FIELDS and event in ("string", "number"):
            current[prefix.rsplit(".", 1)[1]] = value
            if prefix == "traceEvents.item.cat" and value == "kernel":
                kernels += 1
        elif (
            prefix == "baseTimeNanoseconds" and event == "number" and type(value) is int
        ):
            base_time_ns = value
    return has_events, events, kernels, base_time_ns, markers


def inspect_trace(path: Path, config: Config, deadline: float) -> dict:
    try:
        return scan_trace(path, config, deadline, scan_blocks)
    except Layout:  # not Kineto's own writer layout: parse tokens instead
        return scan_trace(path, config, deadline, scan_tokens)


def scan_trace(path: Path, config: Config, deadline: float, scan) -> dict:
    before = file_state(path)
    with open_trace(path) as stream:
        reader = BoundedReader(stream, config.max_trace_bytes, deadline)
        has_events, events, kernels, base_time_ns, markers = scan(reader)
        # Consuming the whole stream also verifies the gzip trailer and CRC.
        if file_state(path) != before:
            raise ValueError("Trace changed during validation")
        if not has_events or not events:
            raise ValueError("Trace has no traceEvents objects")
        return {
            "location": str(path),
            "bytes": before[0],
            "uncompressed_bytes": reader.bytes,
            "sha256_uncompressed": reader.digest.hexdigest(),
            "events": events,
            "cuda_kernels": kernels,
            "validation": "valid",
            "recording_bounds": recording_bounds(base_time_ns, markers),
        }


def inspect_report(
    report: Path, evidence: dict, config: Config, deadline: float
) -> dict:
    """Validate the automatic export and hash its associated native report."""
    paths = (report, report.with_suffix(".sqlite"))
    before = [file_state(path) for path in paths]
    _, _, conn, tables = open_export(report, config)
    with closing(conn):
        conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 10000)
        if conn.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise ValueError("Nsight SQLite export is corrupt")
        counts = {}
        for table in CPU_TABLES:
            conditions = ["cpuCycles = 1"] if table == "COMPOSITE_EVENTS" else []
            parameters = []
            if evidence.get("pid"):
                conditions.append("((globalTid >> 24) & 16777215) = ?")
                parameters.append(evidence["pid"])
            where = " WHERE " + " AND ".join(conditions) if conditions else ""
            query = f'SELECT COUNT(*) FROM "{table}"{where}'
            counts[table] = (
                conn.execute(query, parameters).fetchone()[0] if table in tables else 0
            )
        if evidence["required_activity"] == "cpu_events" and not (
            counts["COMPOSITE_EVENTS"] or counts["SCHED_EVENTS"] or counts["OSRT_API"]
        ):
            raise ValueError("Frontend report contains no CPU activity beyond NVTX")
        # One scan per CUDA table yields both its count and its GPU processes.
        processes = set()
        for table in CUDA_TABLES:
            counts[table] = 0
            if table not in tables:
                continue
            query = f'SELECT globalPid, COUNT(*) FROM "{table}" GROUP BY globalPid'
            for pid, count in conn.execute(query):
                counts[table] += count
                if pid is not None:
                    processes.add(pid)
        gpu_processes = sorted(processes)
        if evidence["required_activity"] == "cuda_events":
            if not gpu_processes:
                raise ValueError(
                    "Worker report contains no CUDA kernel or memory activity"
                )
            expected = evidence.get("expected_gpu_processes", 0)
            if expected and len(gpu_processes) != expected:
                raise ValueError(
                    "Worker CUDA process count differs from configured rank count"
                )
        bounds = report_bounds(conn, tables)
        diagnostics = []
        if "DIAGNOSTIC_EVENT" in tables:
            query = "SELECT text FROM DIAGNOSTIC_EVENT WHERE severity >= 2 LIMIT 16"
            diagnostics = [row[0][:512] for row in conn.execute(query)]
    artifacts = [
        {
            "location": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path, deadline),
        }
        for path in paths
    ]
    if before != [file_state(path) for path in paths]:
        raise ValueError("Nsight export is still changing")
    return {
        "status": "verified",
        "coverage": "observed_required_activity",
        "artifacts": artifacts,
        "events": counts,
        "cpu_sampling_status": (
            "observed" if counts["COMPOSITE_EVENTS"] else "not_observed"
        ),
        "context_switch_status": (
            "observed" if counts["SCHED_EVENTS"] else "not_observed"
        ),
        "gpu_processes": gpu_processes,
        "recording_bounds": bounds,
        "diagnostics": diagnostics,
        "rank_identity": "unattested",
        "association": "new_report_in_dedicated_launch_prefix",
        "validation_scope": "SQLite integrity and required activity; native report transport hash",
    }


def inspect_nsys(item: dict, evidence: dict, config: Config) -> dict:
    """Full offline inspection of a target's Nsight report and SQLite export."""
    report = next(
        (
            Path(a["location"])
            for a in item.get("artifacts", [])
            if a["location"].endswith(".nsys-rep")
        ),
        None,
    )
    if report is None:
        return {"status": "invalid", "error": "no Nsight report"}
    try:
        deadline = time.monotonic() + config.artifact_validation_timeout_s
        return inspect_report(report, evidence, config, deadline)
    except (OSError, sqlite3.Error, ValueError, TimeoutError) as exc:
        return {"status": "invalid", "error": str(exc)}


OFFLINE = Config("unused", "unused", artifact_validation_timeout_s=3600)


def inspect_rank(artifact: dict, config: Config = OFFLINE) -> dict:
    """Full inspection of one Kineto rank trace; one bundle pool task each."""
    path = Path(artifact["location"])
    try:
        result = inspect_trace(
            path, config, time.monotonic() + config.artifact_validation_timeout_s
        )
    except (
        OSError,
        EOFError,
        zlib.error,
        ijson.JSONError,
        ValueError,
        TimeoutError,
    ) as exc:
        result = {"location": str(path), "validation": "invalid", "error": str(exc)}
    return {**result, "rank_id": artifact.get("rank_id")}


def kineto_result(item: dict, artifacts: list[dict]) -> dict:
    ok = artifacts and all(a["validation"] == "valid" for a in artifacts)
    status = item.get("status") if ok else "invalid"
    return {
        "status": "verified" if status == "present" else status,
        "artifacts": artifacts,
    }


def inspect_kineto(item: dict, config: Config = OFFLINE) -> dict:
    """Full offline inspection of a target's Kineto rank traces."""
    return kineto_result(
        item, [inspect_rank(a, config) for a in item.get("artifacts", [])]
    )


def inspect_target(item: dict, evidence: dict) -> dict:
    """Full offline inspection of one target's native artifacts (bundle job)."""
    if evidence.get("collector") == "nsys":
        return inspect_nsys(item, evidence, OFFLINE)
    if evidence.get("collector") == "torch_profiler":
        return inspect_kineto(item, OFFLINE)
    return {
        "status": item.get("status", "unverified"),
        "reason": "no offline inspector",
    }
