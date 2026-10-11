# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-process Nsight triggers and automatic report discovery; no CLI control."""

import asyncio
import hashlib
import os
import re
import sqlite3
import time
from contextlib import closing
from pathlib import Path

import msgspec

# Optional profiling extra; ordinary Torch deployments do not require NVTX.
try:
    import nvtx
except ImportError:
    nvtx = None

from .collectors import CollectionError, TorchCollector, process_epoch
from .models import Config, Plan, RuntimeSnapshot

CPU_TABLES = ("COMPOSITE_EVENTS", "SCHED_EVENTS", "OSRT_API", "NVTX_EVENTS")
CUDA_TABLES = (
    "CUPTI_ACTIVITY_KIND_KERNEL",
    "CUPTI_ACTIVITY_KIND_MEMCPY",
    "CUPTI_ACTIVITY_KIND_MEMSET",
)


def file_state(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


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
        prepare_reports(self.evidence)
        self.range_handle = nvtx.start_range(message="ROI", domain="app")
        self.evidence.update(started_at_ns=time.time_ns(), coverage="recording")

    async def stop(self) -> None:
        if self.range_handle is not None:
            self.evidence["stop_requested_at_ns"] = time.time_ns()
            nvtx.end_range(self.range_handle)
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
        prepare_reports(self.evidence)
        await super().start()


def nsys_collectors(
    config: Config, runtime: RuntimeSnapshot, plan: Plan, directory: Path
) -> list:
    collectors = []
    workers = {worker.target_id: worker for worker in runtime.workers}
    for target in plan.targets:
        if target.target_id == "frontend":
            collectors.append(
                NsysFrontendCollector(runtime, config.nsys_frontend_output_prefix)
            )
        else:
            worker = workers[target.target_id]
            if worker.backend == "sglang":
                worker = msgspec.structs.replace(
                    worker,
                    start_options={
                        **worker.start_options,
                        "activities": ["CUDA_PROFILER"],
                    },
                )
            collectors.append(NsysWorkerCollector(worker, config))
    return collectors


def inspect_report(
    report: Path, evidence: dict, config: Config, deadline: float
) -> dict:
    """Validate the automatic export and hash its associated native report."""
    paths = (report, report.with_suffix(".sqlite"))
    before = [file_state(path) for path in paths]
    if any(size <= 0 or size > config.max_trace_bytes for size, _ in before):
        raise ValueError("Nsight artifact is empty or exceeds the byte limit")
    with closing(
        sqlite3.connect(paths[1].as_uri() + "?mode=ro", uri=True, timeout=0.1)
    ) as conn:
        conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 10000)
        if conn.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise ValueError("Nsight SQLite export is corrupt")
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
        counts = {}
        for table in (*CPU_TABLES, *CUDA_TABLES):
            conditions = ["cpuCycles = 1"] if table == "COMPOSITE_EVENTS" else []
            parameters = []
            if evidence.get("pid") and table in CPU_TABLES:
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
        gpu_processes = sorted(
            {
                row[0]
                for table in CUDA_TABLES
                if counts[table]
                for row in conn.execute(f'SELECT DISTINCT globalPid FROM "{table}"')
                if row[0] is not None
            }
        )
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
        bounds = {"status": "unknown"}
        if {"ANALYSIS_DETAILS", "TARGET_INFO_SESSION_START_TIME"} <= tables:
            capture = conn.execute(
                "SELECT startTime, stopTime FROM ANALYSIS_DETAILS"
            ).fetchall()
            anchor = conn.execute(
                "SELECT utcEpochNs, systemClockNs FROM TARGET_INFO_SESSION_START_TIME"
            ).fetchall()
            if len(capture) == len(anchor) == 1:
                start, end = capture[0]
                utc, clock = anchor[0]
                if (
                    all(type(value) is int for value in (start, end, utc, clock))
                    and end > start
                ):
                    bounds = {
                        "status": "observed",
                        "clock": "nsight_host_unix_ns_unattested",
                        "start_unix_ns": utc + start - clock,
                        "end_unix_ns": utc + end - clock,
                        "duration_ns": end - start,
                    }
        diagnostics = (
            [
                row[0][:512]
                for row in conn.execute(
                    "SELECT text FROM DIAGNOSTIC_EVENT WHERE severity >= 2 LIMIT 16"
                )
            ]
            if "DIAGNOSTIC_EVENT" in tables
            else []
        )
    artifacts = []
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(65536), b""):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Nsight artifact validation deadline expired")
                digest.update(chunk)
        artifacts.append(
            {
                "location": str(path),
                "bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
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


async def validate_reports(evidence: dict, config: Config) -> dict:
    if "previous_reports" not in evidence:
        return {
            "status": "invalid",
            "coverage": "unknown",
            "error": "Nsight capture was not prepared",
        }
    prefix = Path(evidence["output_prefix"])
    baseline = set(evidence.get("previous_reports", []))
    deadline = time.monotonic() + config.artifact_validation_timeout_s
    error = "Automatic Nsight report/SQLite export has not arrived"
    while time.monotonic() < deadline:
        reports = [path for path in report_paths(prefix) if str(path) not in baseline]
        if len(reports) > 1:
            return {
                "status": "invalid",
                "coverage": "partial",
                "error": "Ambiguous Nsight reports for one window",
            }
        if reports:
            try:
                return await asyncio.to_thread(
                    inspect_report, reports[0], evidence, config, deadline
                )
            except (OSError, sqlite3.Error, ValueError, TimeoutError) as exc:
                error = str(exc)
        await asyncio.sleep(min(0.5, max(0, deadline - time.monotonic())))
    return {"status": "partial", "coverage": "partial", "error": error}
