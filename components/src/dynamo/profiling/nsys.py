# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-process Nsight triggers and automatic report discovery; no CLI control."""

import asyncio
import contextlib
import os
import re
import sqlite3
import time
from contextlib import closing
from pathlib import Path

# Optional profiling extra; ordinary Torch deployments do not require NVTX.
try:
    import nvtx
except ImportError:
    nvtx = None

from .collectors import CollectionError, TorchCollector, process_epoch, selected_workers
from .models import Config, Plan, RuntimeSnapshot


def report_paths(prefix: Path) -> list[Path]:
    pattern = re.compile(rf"{re.escape(prefix.name)}(?:\.\d+)?\.nsys-rep")
    return [path for path in prefix.parent.iterdir() if pattern.fullmatch(path.name)]


def prepare_reports(evidence: dict) -> None:
    prefix = Path(evidence["output_prefix"])
    if not prefix.parent.is_dir():
        raise CollectionError("Nsight output directory is not mounted")
    # Each prepared launch must have its own prefix and a single controller.
    evidence["previous_reports"] = [str(path) for path in report_paths(prefix)]


class NsysFrontendCollector:
    """Hold a process range across awaits; no thread-local push/pop range."""

    def __init__(self, runtime: RuntimeSnapshot, prefix: str):
        self.runtime = runtime
        self.range_handle = None
        self.starting: asyncio.Future | None = None
        self.evidence = {
            "target_id": "frontend",
            "process_epoch": runtime.frontend_epoch,
            "pid": runtime.frontend_pid,
            "collector": "nsys",
            "output_prefix": prefix,
            "trigger": "nvtx:ROI@app",
            "required_activity": "cpu_events",
            "coverage": "unknown",
        }

    async def start(self) -> None:
        if nvtx is None:
            raise CollectionError("Install ai-dynamo[profiling] for NVTX capture")
        if (
            self.runtime.frontend_pid != os.getpid()
            or process_epoch(os.getpid()) != self.runtime.frontend_epoch
        ):
            raise CollectionError("NVTX frontend capture requires an embedded host")
        await asyncio.to_thread(prepare_reports, self.evidence)
        self.evidence["start_requested_at_ns"] = time.time_ns()
        # Nsight can block range transitions; keep them off the serving loop.
        # A start timeout cancels this await but not the thread, which may still
        # open the range; keep its future so stop() can close it.
        self.starting = asyncio.ensure_future(
            asyncio.to_thread(nvtx.start_range, message="ROI", domain="app")
        )
        self.range_handle = await asyncio.shield(self.starting)
        self.evidence.update(started_at_ns=time.time_ns(), coverage="recording")

    async def stop(self) -> None:
        if self.range_handle is None and self.starting is not None:
            with contextlib.suppress(Exception):  # a failed start opened nothing
                self.range_handle = await self.starting
        self.starting = None
        if self.range_handle is not None:
            self.evidence["stop_requested_at_ns"] = time.time_ns()
            await asyncio.to_thread(nvtx.end_range, self.range_handle)
            self.range_handle = None
            self.evidence.update(
                stopped_at_ns=time.time_ns(), coverage="export_pending"
            )


class NsysWorkerCollector(TorchCollector):
    """Reuse backend HTTP controls configured to call CUDA profiler APIs."""

    def __init__(self, worker, config: Config):
        super().__init__(worker, config)
        self.evidence.update(
            collector="nsys",
            output_prefix=worker.nsys_output_prefix,
            trigger="cudaProfilerApi",
            required_activity="cuda_events",
            expected_gpu_processes=len(worker.expected_rank_ids),
        )

    async def start(self) -> None:
        await asyncio.to_thread(prepare_reports, self.evidence)
        await super().start()


def nsys_collectors(
    config: Config, runtime: RuntimeSnapshot, plan: Plan, directory: Path
) -> list:
    workers = selected_workers(runtime, plan, {"sglang"}, activities=["CUDA_PROFILER"])
    return [
        NsysFrontendCollector(runtime, config.nsys_frontend_output_prefix)
        if worker is None
        else NsysWorkerCollector(worker, config)
        for worker in workers
    ]


def report_bounds(conn, tables: set) -> dict:
    """Native recording bounds on the host UTC clock, if the export has them."""
    unknown = {"status": "unknown"}
    if not {"ANALYSIS_DETAILS", "TARGET_INFO_SESSION_START_TIME"} <= tables:
        return unknown
    columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(TARGET_INFO_SESSION_START_TIME)")
    }
    if not {"utcEpochNs", "systemClockNs"} <= columns:
        return unknown  # older exports; the bundle derives bounds offline
    capture = conn.execute(
        "SELECT startTime, stopTime FROM ANALYSIS_DETAILS"
    ).fetchall()
    anchor = conn.execute(
        "SELECT utcEpochNs, systemClockNs FROM TARGET_INFO_SESSION_START_TIME"
    ).fetchall()
    if len(capture) != 1 or len(anchor) != 1:
        return unknown
    (start, end), (utc, clock) = capture[0], anchor[0]
    if not all(type(v) is int for v in (start, end, utc, clock)) or end <= start:
        return unknown
    return {
        "status": "observed",
        "clock": "nsight_host_unix_ns_unattested",
        "start_unix_ns": utc + start - clock,
        "end_unix_ns": utc + end - clock,
        "duration_ns": end - start,
    }


async def validate_reports(evidence: dict, config: Config) -> dict:
    if "previous_reports" not in evidence:
        return {
            "status": "invalid",
            "coverage": "unknown",
            "error": "Nsight capture was not prepared",
        }
    prefix = Path(evidence["output_prefix"])
    baseline = set(evidence["previous_reports"])
    deadline = time.monotonic() + config.artifact_validation_timeout_s
    error = "Automatic Nsight report/SQLite export has not arrived"
    while time.monotonic() < deadline:
        # The prefix is on shared storage: keep scans off the serving loop.
        try:
            result = await asyncio.to_thread(check_report, prefix, baseline, config)
        except (OSError, sqlite3.Error, ValueError, TimeoutError) as exc:
            error = str(exc)
        else:
            if result is not None:
                return result
        await asyncio.sleep(min(0.5, max(0, deadline - time.monotonic())))
    return {"status": "partial", "coverage": "partial", "error": error}


def open_export(report: Path, config: Config):
    """Open this report's finished automatic SQLite export read-only.

    Returns (paths, sizes, connection, tables); the caller closes the connection.
    """
    paths = (report, report.with_suffix(".sqlite"))
    sizes = [path.stat().st_size for path in paths]
    if any(size <= 0 or size > config.max_trace_bytes for size in sizes):
        raise ValueError("Nsight artifact is empty or exceeds the byte limit")
    conn = sqlite3.connect(paths[1].as_uri() + "?mode=ro", uri=True, timeout=0.1)
    try:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        # Nsight writes its total export time after exporting the event tables.
        # File existence or a momentarily stable size alone is not readiness.
        metadata = dict(conn.execute("SELECT name, value FROM META_DATA_EXPORT"))
        if (
            not metadata.get("EXPORT_TIME")
            or metadata.get("EXPORT_PARAM_INPUT_FILE") != report.name
        ):
            raise ValueError(
                "Nsight SQLite export is incomplete or belongs to another report"
            )
    except BaseException:
        conn.close()
        raise
    return paths, sizes, conn, tables


def check_report(prefix: Path, baseline: set[str], config: Config) -> dict | None:
    """Cheap readiness check on the capture path; None until a new report exists.

    Full inspection (integrity scan, required activity, GPU process count and
    hashes) runs offline in the bundle job (dynamo.profiling.inspection).
    """
    reports = [path for path in report_paths(prefix) if str(path) not in baseline]
    if len(reports) > 1:
        return {
            "status": "invalid",
            "coverage": "partial",
            "error": "Ambiguous Nsight reports for one window",
        }
    if not reports:
        return None
    paths, sizes, conn, tables = open_export(reports[0], config)
    with closing(conn):
        bounds = report_bounds(conn, tables)
    return {
        "status": "present",
        "coverage": "present",
        "artifacts": [
            {"location": str(path), "bytes": size} for path, size in zip(paths, sizes)
        ],
        "recording_bounds": bounds,
        "association": "new_report_in_dedicated_launch_prefix",
        "inspection": "deferred_to_bundle",
    }
