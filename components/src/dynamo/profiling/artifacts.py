# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Artifact checks: a cheap presence check on the capture path, full inspection offline.

The coordinator only confirms that each expected native file exists and is
readable ("present"). Full parsing, hashing and recording bounds run in the
post-processing bundle job, off the frontend and the capture's critical path.
"""

import asyncio
import gzip
import hashlib
import re
import time
from decimal import Decimal
from pathlib import Path
from typing import Protocol

import ijson

from .models import Config
from .nsys import validate_reports


class ArtifactValidator(Protocol):
    async def validate(self, evidence: dict, config: Config) -> dict:
        ...


class BoundedReader:
    def __init__(self, stream, limit: int, deadline: float):
        self.stream = stream
        self.limit = limit
        self.deadline = deadline
        self.bytes = 0
        self.digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Artifact validation deadline expired")
        data = self.stream.read(min(size if size >= 0 else 65536, 65536))
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
    marker = markers[0]
    values = (marker.get("ts"), marker.get("dur"))
    if any(type(value) not in (int, Decimal) for value in values):
        return {"status": "unknown", "reason": "invalid_native_profiler_bounds"}
    timestamp, duration = (Decimal(value) for value in values)
    if not timestamp.is_finite() or not duration.is_finite() or duration < 0:
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


def inspect_trace(path: Path, config: Config, deadline: float) -> dict:
    before = path.stat()
    events = kernels = 0
    has_events = False
    base_time_ns = None
    markers = []
    current = {}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        reader = BoundedReader(stream, config.max_trace_bytes, deadline)
        for prefix, event, value in ijson.parse(reader):
            if (prefix, event) == ("traceEvents", "start_array"):
                has_events = True
            elif (prefix, event) == ("traceEvents.item", "start_map"):
                events += 1
                current = {}
            elif prefix == "traceEvents.item" and event == "end_map":
                if (
                    current.get("cat") == "Trace"
                    and current.get("ph") == "X"
                    and isinstance(current.get("name"), str)
                    and current["name"].startswith("PyTorch Profiler")
                    and len(markers) < 2
                ):
                    markers.append(current)
            elif prefix in (
                "traceEvents.item.name",
                "traceEvents.item.cat",
                "traceEvents.item.ph",
                "traceEvents.item.ts",
                "traceEvents.item.dur",
            ) and event in ("string", "number"):
                current[prefix.rsplit(".", 1)[1]] = value
                if prefix == "traceEvents.item.cat" and value == "kernel":
                    kernels += 1
            elif (
                prefix == "baseTimeNanoseconds"
                and event == "number"
                and type(value) is int
            ):
                base_time_ns = value
        # Consuming the whole stream also verifies the gzip trailer and CRC.
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Trace changed during validation")
        if not has_events or not events:
            raise ValueError("Trace has no traceEvents objects")
        return {
            "location": str(path),
            "bytes": after.st_size,
            "uncompressed_bytes": reader.bytes,
            "sha256_uncompressed": reader.digest.hexdigest(),
            "events": events,
            "cuda_kernels": kernels,
            "validation": "valid",
            "recording_bounds": recording_bounds(base_time_ns, markers),
        }


def rank_files(location: str, prefix: str) -> list[tuple[str, Path]]:
    """Kineto rank traces written under this session's unique prefix."""
    pattern = re.compile(rf"^{re.escape(prefix)}_(.+)\.\d+\.pt\.trace\.json(?:\.gz)?$")
    return sorted(
        (match.group(1), path)
        for path in Path(location).iterdir()
        if (match := pattern.fullmatch(path.name)) and path.is_file()
    )


def present(path: Path) -> str | None:
    """Return why a native trace is unusable, without parsing it."""
    size = path.stat().st_size
    if size <= 0:
        return "empty"
    if path.suffix == ".gz":
        with path.open("rb") as stream:
            if stream.read(2) != b"\x1f\x8b":
                return "not a gzip file"
    return None


class SharedFilesystemValidator:
    """Cheap checks only; full inspection is the bundle's job."""

    async def validate(self, evidence: dict, config: Config) -> dict:
        if evidence.get("collector") == "nsys":
            return await validate_reports(evidence, config)
        return await asyncio.to_thread(self._validate, evidence, config)

    def _validate(self, evidence: dict, config: Config) -> dict:
        location = evidence.get("artifact_location")
        prefix = evidence.get("artifact_prefix")
        if evidence.get("collector") != "torch_profiler" or not location or not prefix:
            return {
                "status": "unverified",
                "coverage": "unknown",
                "reason": "no_supported_accessible_store",
            }
        if not Path(location).is_dir():
            return {
                "status": "unverified",
                "coverage": "unknown",
                "reason": "artifact_store_not_mounted",
                "location": location,
            }
        try:
            found = rank_files(location, prefix)
            artifacts = []
            for rank, path in found:
                problem = present(path)
                artifacts.append(
                    {
                        "location": str(path),
                        "rank_id": rank,
                        "bytes": path.stat().st_size,
                        "validation": "present" if problem is None else "invalid",
                        **({"error": problem} if problem else {}),
                    }
                )
        except OSError as exc:
            return {"status": "invalid", "coverage": "partial", "error": str(exc)}
        ranks = [rank for rank, _ in found]
        expected = evidence.get("expected_rank_ids", [])
        missing = sorted(set(expected) - set(ranks))
        unexpected = sorted(set(ranks) - set(expected)) if expected else []
        duplicate = len(ranks) != len(set(ranks))
        invalid = any(artifact["validation"] != "present" for artifact in artifacts)
        status = (
            "invalid"
            if invalid or duplicate
            else "partial"
            if missing or unexpected or not artifacts
            else "present"
            if expected
            else "unverified"
        )
        return {
            "status": status,
            "coverage": "present"
            if status == "present"
            else "partial"
            if status in ("invalid", "partial")
            else "unknown",
            "artifacts": artifacts,
            "expected_rank_ids": expected,
            "observed_rank_ids": sorted(ranks),
            "missing_rank_ids": missing,
            "unexpected_rank_ids": unexpected,
            "duplicate_rank_files": duplicate,
            "association": "unique_session_prefix",
            "scope": "engine_rank_files",
            "inspection": "deferred_to_bundle",
            "excluded": "AsyncLLM CPU traces lack the native profile prefix",
        }


def inspect_kineto(item: dict, config: Config) -> dict:
    """Full offline inspection of a target's Kineto rank traces."""
    deadline = time.monotonic() + config.artifact_validation_timeout_s
    artifacts = []
    for artifact in item.get("artifacts", []):
        path = Path(artifact["location"])
        try:
            result = inspect_trace(path, config, deadline)
        except (OSError, EOFError, ijson.JSONError, ValueError, TimeoutError) as exc:
            result = {"location": str(path), "validation": "invalid", "error": str(exc)}
        artifacts.append({**result, "rank_id": artifact.get("rank_id")})
    ok = artifacts and all(a["validation"] == "valid" for a in artifacts)
    return {
        "status": "verified"
        if ok and item.get("status") == "present"
        else "invalid"
        if not ok
        else item.get("status"),
        "artifacts": artifacts,
    }
