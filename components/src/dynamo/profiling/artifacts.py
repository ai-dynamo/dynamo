# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared-filesystem Kineto validation; other stores implement the same protocol."""

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


class SharedFilesystemValidator:
    async def validate(self, evidence: dict, config: Config) -> dict:
        # Parsing can take seconds for native traces; never run it on the
        # frontend's serving loop. The reader enforces a cooperative deadline.
        return await asyncio.to_thread(self._validate, evidence.copy(), config)

    def _validate(self, evidence: dict, config: Config) -> dict:
        location = evidence.get("artifact_location")
        prefix = evidence.get("artifact_prefix")
        if evidence.get("collector") != "torch_profiler" or not location or not prefix:
            return {
                "status": "unverified",
                "coverage": "unknown",
                "reason": "no_supported_accessible_store",
            }
        directory = Path(location)
        if not directory.is_dir():
            return {
                "status": "unverified",
                "coverage": "unknown",
                "reason": "artifact_store_not_mounted",
                "location": location,
            }
        pattern = re.compile(
            rf"^{re.escape(prefix)}_(.+)\.\d+\.pt\.trace\.json(?:\.gz)?$"
        )
        deadline = time.monotonic() + config.artifact_validation_timeout_s
        artifacts = []
        ranks = []
        try:
            for path in directory.iterdir():
                if time.monotonic() >= deadline:
                    raise TimeoutError("Artifact enumeration deadline expired")
                match = pattern.fullmatch(path.name)
                if not match or not path.is_file():
                    continue
                rank = match.group(1)
                ranks.append(rank)
                try:
                    artifact = inspect_trace(path, config, deadline)
                except (
                    OSError,
                    EOFError,
                    ijson.JSONError,
                    ValueError,
                    TimeoutError,
                ) as exc:
                    artifact = {
                        "location": str(path),
                        "validation": "invalid",
                        "error": str(exc),
                    }
                artifacts.append({**artifact, "rank_id": rank})
        except (OSError, TimeoutError) as exc:
            return {
                "status": "invalid",
                "coverage": "partial",
                "artifacts": artifacts,
                "error": str(exc),
            }
        artifacts.sort(key=lambda artifact: artifact["location"])
        ranks.sort()
        expected = evidence.get("expected_rank_ids", [])
        missing = sorted(set(expected) - set(ranks))
        unexpected = sorted(set(ranks) - set(expected)) if expected else []
        duplicate = len(ranks) != len(set(ranks))
        invalid = any(artifact["validation"] != "valid" for artifact in artifacts)
        status = (
            "invalid"
            if invalid or duplicate
            else "partial"
            if missing or unexpected or not artifacts
            else "verified"
            if expected
            else "unverified"
        )
        return {
            "status": status,
            "coverage": "complete"
            if status == "verified"
            else "partial"
            if status in ("invalid", "partial")
            else "unknown",
            "artifacts": artifacts,
            "expected_rank_ids": expected,
            "observed_rank_ids": ranks,
            "missing_rank_ids": missing,
            "unexpected_rank_ids": unexpected,
            "duplicate_rank_files": duplicate,
            "association": "unique_session_prefix",
            "scope": "engine_rank_files",
            "excluded": "AsyncLLM CPU traces lack the native profile prefix",
        }
